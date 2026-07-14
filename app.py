"""
Eye Detection All-in-One Project
--------------------------------
Single-file project using:
Python, OpenCV, NumPy, Pandas, Matplotlib, Tkinter, Flask, SQLite, HTML, CSS, JavaScript.
Run in VS Code:
    python -m venv venv
    venv\\Scripts\\activate        # Windows
    pip install -r requirements.txt
    python app.py
Open: http://127.0.0.1:5000
Optional desktop GUI:
    python app.py --desktop
"""

import base64
import io
import os
import sqlite3
import sys
from datetime import datetime
from typing import Dict, List, Tuple

import cv2
import numpy as np
import pandas as pd
from flask import Flask, Response, jsonify, request, send_file

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    MATPLOTLIB_AVAILABLE = True
except Exception:
    plt = None
    MATPLOTLIB_AVAILABLE = False

APP_NAME = "AI Eye Detection System"
DB_FILE = "eye_detection.db"
UPLOAD_DIR = "processed_images"
os.makedirs(UPLOAD_DIR, exist_ok=True)

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 8 * 1024 * 1024

# Haar cascade files come with opencv-python package.
FACE_CASCADE = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
EYE_CASCADE = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_eye.xml")


def init_db() -> None:
    """Create SQLite table for detection history."""
    with sqlite3.connect(DB_FILE) as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS detections (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                source TEXT NOT NULL,
                faces INTEGER NOT NULL,
                eyes INTEGER NOT NULL,
                avg_eye_area REAL NOT NULL,
                image_path TEXT,
                created_at TEXT NOT NULL
            )
            """
        )
        conn.commit()


def save_detection(source: str, faces: int, eyes: int, avg_eye_area: float, image_path: str) -> None:
    """Save result row in SQLite."""
    with sqlite3.connect(DB_FILE) as conn:
        conn.execute(
            """
            INSERT INTO detections(source, faces, eyes, avg_eye_area, image_path, created_at)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (source, faces, eyes, float(avg_eye_area), image_path, datetime.now().strftime("%Y-%m-%d %H:%M:%S")),
        )
        conn.commit()


def read_history_df(limit: int = 50) -> pd.DataFrame:
    """Read history using Pandas."""
    init_db()
    with sqlite3.connect(DB_FILE) as conn:
        return pd.read_sql_query(
            "SELECT id, source, faces, eyes, avg_eye_area, image_path, created_at "
            "FROM detections ORDER BY id DESC LIMIT ?",
            conn,
            params=(limit,),
        )


def bytes_to_cv2_image(file_bytes: bytes) -> np.ndarray:
    """Convert uploaded image bytes to OpenCV BGR image."""
    np_buffer = np.frombuffer(file_bytes, dtype=np.uint8)
    image = cv2.imdecode(np_buffer, cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError("Invalid image. Please upload JPG, JPEG, or PNG file.")
    return image


def data_url_to_cv2_image(data_url: str) -> np.ndarray:
    """Convert browser camera dataURL to OpenCV BGR image."""
    if "," in data_url:
        data_url = data_url.split(",", 1)[1]
    image_bytes = base64.b64decode(data_url)
    return bytes_to_cv2_image(image_bytes)


def image_to_base64(image: np.ndarray) -> str:
    """Convert OpenCV BGR image to browser-friendly base64 JPEG."""
    ok, buffer = cv2.imencode(".jpg", image, [int(cv2.IMWRITE_JPEG_QUALITY), 90])
    if not ok:
        raise ValueError("Could not encode processed image.")
    encoded = base64.b64encode(buffer).decode("utf-8")
    return f"data:image/jpeg;base64,{encoded}"


def safe_resize(image: np.ndarray, width: int = 900) -> np.ndarray:
    """Resize large images for faster detection."""
    h, w = image.shape[:2]
    if w <= width:
        return image
    ratio = width / float(w)
    return cv2.resize(image, (width, int(h * ratio)), interpolation=cv2.INTER_AREA)


def detect_eyes(image: np.ndarray) -> Tuple[np.ndarray, Dict]:
    """
    Detect faces and eyes.
    Strategy:
    1. Detect faces.
    2. Detect eyes inside each face ROI.
    3. If no face is found, detect eyes in whole image.
    """
    image = safe_resize(image)
    output = image.copy()
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.equalizeHist(gray)

    faces = FACE_CASCADE.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(70, 70),
        flags=cv2.CASCADE_SCALE_IMAGE,
    )

    detected_eyes: List[Tuple[int, int, int, int]] = []
    face_count = len(faces)

    for (x, y, w, h) in faces:
        cv2.rectangle(output, (x, y), (x + w, y + h), (20, 180, 255), 2)
        roi_gray = gray[y : y + h, x : x + w]
        roi_color = output[y : y + h, x : x + w]
        eyes = EYE_CASCADE.detectMultiScale(
            roi_gray,
            scaleFactor=1.08,
            minNeighbors=8,
            minSize=(22, 22),
            maxSize=(int(w * 0.45), int(h * 0.35)),
        )

        # Keep only upper-half face detections because eyes are normally there.
        for (ex, ey, ew, eh) in eyes:
            if ey < h * 0.62:
                detected_eyes.append((x + ex, y + ey, ew, eh))
                cv2.rectangle(roi_color, (ex, ey), (ex + ew, ey + eh), (0, 255, 100), 2)
                cv2.putText(roi_color, "Eye", (ex, max(18, ey - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 100), 2)

    if face_count == 0:
        eyes = EYE_CASCADE.detectMultiScale(
            gray,
            scaleFactor=1.08,
            minNeighbors=7,
            minSize=(25, 25),
            flags=cv2.CASCADE_SCALE_IMAGE,
        )
        for (ex, ey, ew, eh) in eyes:
            detected_eyes.append((ex, ey, ew, eh))
            cv2.rectangle(output, (ex, ey), (ex + ew, ey + eh), (0, 255, 100), 2)
            cv2.putText(output, "Eye", (ex, max(18, ey - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 100), 2)

    eye_areas = [w * h for (_, _, w, h) in detected_eyes]
    avg_eye_area = float(np.mean(eye_areas)) if eye_areas else 0.0

    message = "Eyes detected successfully" if detected_eyes else "No eyes detected. Try front face, good light, clear image."
    cv2.putText(output, f"Faces: {face_count} | Eyes: {len(detected_eyes)}", (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 3)
    cv2.putText(output, f"Faces: {face_count} | Eyes: {len(detected_eyes)}", (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (30, 30, 30), 1)

    result = {
        "faces": int(face_count),
        "eyes": int(len(detected_eyes)),
        "avg_eye_area": round(avg_eye_area, 2),
        "message": message,
    }
    return output, result


def process_and_store(image: np.ndarray, source: str) -> Dict:
    """Run detection, save image, save database row, return API response."""
    processed, result = detect_eyes(image)
    filename = f"{source}_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}.jpg"
    path = os.path.join(UPLOAD_DIR, filename)
    cv2.imwrite(path, processed)
    save_detection(source, result["faces"], result["eyes"], result["avg_eye_area"], path)
    result["image"] = image_to_base64(processed)
    result["image_path"] = path
    return result


HTML_PAGE = r"""
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8" />
<meta name="viewport" content="width=device-width, initial-scale=1.0" />
<title>AI Eye Detection System</title>
<style>
:root{--bg:#0f172a;--card:#111c33;--text:#e5e7eb;--muted:#94a3b8;--brand:#38bdf8;--ok:#22c55e;--bad:#ef4444;}
*{box-sizing:border-box} body{margin:0;font-family:Arial,Helvetica,sans-serif;background:radial-gradient(circle at top,#1e3a8a,#0f172a 45%);color:var(--text)}
.header{padding:34px 18px;text-align:center}.header h1{margin:0;font-size:38px}.header p{color:var(--muted);font-size:17px}
.container{width:min(1150px,94%);margin:auto;display:grid;grid-template-columns:1fr 1fr;gap:20px}.card{background:rgba(17,28,51,.92);border:1px solid rgba(148,163,184,.22);border-radius:18px;padding:22px;box-shadow:0 16px 40px rgba(0,0,0,.25)}
.card h2{margin-top:0}.full{grid-column:1/-1}.btn{border:0;border-radius:12px;padding:12px 16px;margin:6px 6px 6px 0;background:var(--brand);color:#00111f;font-weight:700;cursor:pointer}.btn:hover{filter:brightness(1.08)}.btn.secondary{background:#334155;color:var(--text)}.btn.green{background:var(--ok)}.btn.red{background:var(--bad);color:white}
input[type=file]{width:100%;padding:14px;background:#0b1222;border:1px dashed #64748b;border-radius:12px;color:var(--text)} video,canvas,img{width:100%;border-radius:14px;background:#050816;border:1px solid rgba(148,163,184,.25)}
#resultImage{margin-top:12px}.stats{display:grid;grid-template-columns:repeat(3,1fr);gap:12px;margin-top:10px}.stat{background:#0b1222;padding:15px;border-radius:14px;text-align:center}.stat b{display:block;font-size:26px;color:var(--brand)}
.message{padding:12px;border-radius:12px;background:#0b1222;color:var(--muted);margin-top:12px}.tablewrap{overflow:auto;max-height:330px}table{width:100%;border-collapse:collapse;font-size:14px}th,td{border-bottom:1px solid rgba(148,163,184,.16);padding:10px;text-align:left}th{color:var(--brand)}
.footer{text-align:center;color:var(--muted);padding:25px}.hint{color:var(--muted);font-size:14px;line-height:1.5}.pill{display:inline-block;background:#0b1222;border:1px solid #334155;border-radius:999px;padding:6px 10px;margin:4px;color:#cbd5e1;font-size:13px}
@media(max-width:850px){.container{grid-template-columns:1fr}.header h1{font-size:29px}.stats{grid-template-columns:1fr}}
</style>
</head>
<body>
<section class="header">
  <h1>👁️ AI Eye Detection System</h1>
  <p>Python + OpenCV + NumPy + Pandas + Matplotlib + Flask + SQLite + HTML/CSS/JS</p>
  <span class="pill">Upload Image</span><span class="pill">Webcam Capture</span><span class="pill">History Database</span><span class="pill">Chart Report</span>
</section>
<main class="container">
  <section class="card">
    <h2>1) Upload Photo</h2>
    <p class="hint">JPG/PNG image upload kara. Front face + good light madhe detection best chalel.</p>
    <input id="imageInput" type="file" accept="image/*" />
    <button class="btn" onclick="uploadImage()">Detect Eyes</button>
  </section>
  <section class="card">
    <h2>2) Camera Detection</h2>
    <video id="video" autoplay playsinline></video>
    <canvas id="canvas" style="display:none"></canvas>
    <button class="btn green" onclick="startCamera()">Start Camera</button>
    <button class="btn" onclick="captureFrame()">Capture & Detect</button>
    <button class="btn red" onclick="stopCamera()">Stop</button>
  </section>
  <section class="card full">
    <h2>Detection Result</h2>
    <div class="stats">
      <div class="stat"><b id="faces">0</b>Faces</div>
      <div class="stat"><b id="eyes">0</b>Eyes</div>
      <div class="stat"><b id="area">0</b>Avg Eye Area</div>
    </div>
    <div id="msg" class="message">Result ithe show hoil.</div>
    <img id="resultImage" alt="Processed output will appear here" />
  </section>
  <section class="card">
    <h2>Detection History</h2>
    <button class="btn secondary" onclick="loadHistory()">Refresh History</button>
    <a class="btn secondary" href="/export.csv">Download CSV</a>
    <div class="tablewrap"><table><thead><tr><th>ID</th><th>Source</th><th>Faces</th><th>Eyes</th><th>Time</th></tr></thead><tbody id="historyBody"></tbody></table></div>
  </section>
  <section class="card">
    <h2>Matplotlib Chart</h2>
    <p class="hint">SQLite madhil detection history varun chart generate hoto.</p>
    <button class="btn secondary" onclick="reloadChart()">Reload Chart</button>
    <img id="chart" src="/chart.png" alt="History chart" />
  </section>
</main>
<div class="footer">Made for easy VS Code mini project demo.</div>
<script>
let stream=null;
async function uploadImage(){
  const input=document.getElementById('imageInput');
  if(!input.files.length){showMessage('Please select image first.');return;}
  const form=new FormData(); form.append('image',input.files[0]);
  showMessage('Processing uploaded image...');
  const res=await fetch('/detect_image',{method:'POST',body:form});
  const data=await res.json(); renderResult(data); loadHistory(); reloadChart();
}
async function startCamera(){
  try{stream=await navigator.mediaDevices.getUserMedia({video:true,audio:false}); document.getElementById('video').srcObject=stream; showMessage('Camera started.');}
  catch(e){showMessage('Camera permission denied or not available.');}
}
function stopCamera(){if(stream){stream.getTracks().forEach(t=>t.stop());stream=null;showMessage('Camera stopped.');}}
async function captureFrame(){
  const video=document.getElementById('video'); if(!stream){showMessage('Start camera first.');return;}
  const canvas=document.getElementById('canvas'); canvas.width=video.videoWidth; canvas.height=video.videoHeight;
  canvas.getContext('2d').drawImage(video,0,0); const image=canvas.toDataURL('image/jpeg',0.92);
  showMessage('Processing camera frame...');
  const res=await fetch('/detect_camera',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({image})});
  const data=await res.json(); renderResult(data); loadHistory(); reloadChart();
}
function renderResult(data){
  if(data.error){showMessage(data.error);return;}
  document.getElementById('faces').innerText=data.faces;
  document.getElementById('eyes').innerText=data.eyes;
  document.getElementById('area').innerText=data.avg_eye_area;
  document.getElementById('msg').innerText=data.message;
  document.getElementById('resultImage').src=data.image;
}
function showMessage(text){document.getElementById('msg').innerText=text;}
async function loadHistory(){
  const res=await fetch('/history'); const rows=await res.json(); const body=document.getElementById('historyBody'); body.innerHTML='';
  rows.forEach(r=>{body.innerHTML+=`<tr><td>${r.id}</td><td>${r.source}</td><td>${r.faces}</td><td>${r.eyes}</td><td>${r.created_at}</td></tr>`;});
}
function reloadChart(){document.getElementById('chart').src='/chart.png?ts='+Date.now();}
loadHistory();
</script>
</body>
</html>
"""


@app.route("/")
def home() -> Response:
    return Response(HTML_PAGE, mimetype="text/html")


@app.route("/detect_image", methods=["POST"])
def detect_image_route():
    try:
        if "image" not in request.files:
            return jsonify({"error": "No image file uploaded."}), 400
        image_file = request.files["image"]
        image = bytes_to_cv2_image(image_file.read())
        result = process_and_store(image, "upload")
        return jsonify(result)
    except Exception as exc:
        return jsonify({"error": str(exc)}), 500


@app.route("/detect_camera", methods=["POST"])
def detect_camera_route():
    try:
        payload = request.get_json(force=True)
        if not payload or "image" not in payload:
            return jsonify({"error": "No camera image received."}), 400
        image = data_url_to_cv2_image(payload["image"])
        result = process_and_store(image, "camera")
        return jsonify(result)
    except Exception as exc:
        return jsonify({"error": str(exc)}), 500


@app.route("/history")
def history_route():
    df = read_history_df(50)
    return jsonify(df.to_dict(orient="records"))


@app.route("/export.csv")
def export_csv_route():
    df = read_history_df(500)
    csv_buffer = io.StringIO()
    df.to_csv(csv_buffer, index=False)
    mem = io.BytesIO(csv_buffer.getvalue().encode("utf-8"))
    return send_file(mem, mimetype="text/csv", as_attachment=True, download_name="eye_detection_history.csv")


def create_placeholder_chart_bytes() -> bytes:
    """Create a simple PNG chart placeholder when Matplotlib is unavailable."""
    image = np.full((320, 640, 3), (12, 18, 32), dtype=np.uint8)
    cv2.putText(image, "Eye Detection History", (28, 60), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (56, 189, 248), 2)
    cv2.putText(image, "Chart view is using a fallback image.", (28, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (226, 232, 240), 1)
    cv2.putText(image, "Run detection to populate history data.", (28, 160), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (148, 163, 184), 1)
    cv2.rectangle(image, (40, 210), (600, 280), (34, 197, 94), 2)
    cv2.rectangle(image, (40, 210), (260, 280), (56, 189, 248), -1)
    cv2.rectangle(image, (280, 210), (470, 280), (34, 197, 94), -1)
    ok, buffer = cv2.imencode(".png", image)
    if not ok:
        raise RuntimeError("Unable to create chart image")
    return buffer.tobytes()


@app.route("/chart.png")
def chart_route():
    df = read_history_df(500)
    if not MATPLOTLIB_AVAILABLE or plt is None:
        buffer = io.BytesIO(create_placeholder_chart_bytes())
        buffer.seek(0)
        return send_file(buffer, mimetype="image/png")

    try:
        fig, ax = plt.subplots(figsize=(8, 4.2))
        if df.empty:
            ax.text(0.5, 0.5, "No detection data yet", ha="center", va="center", fontsize=14)
            ax.set_axis_off()
        else:
            df["created_at"] = pd.to_datetime(df["created_at"])
            df["date"] = df["created_at"].dt.strftime("%Y-%m-%d")
            grouped = df.groupby("date")[["faces", "eyes"]].sum().tail(7)
            grouped.plot(kind="bar", ax=ax)
            ax.set_title("Last Detection Summary")
            ax.set_xlabel("Date")
            ax.set_ylabel("Count")
            ax.tick_params(axis="x", rotation=25)
            ax.grid(True, axis="y", alpha=0.25)
        plt.tight_layout()
        buffer = io.BytesIO()
        fig.savefig(buffer, format="png", dpi=120)
        plt.close(fig)
        buffer.seek(0)
        return send_file(buffer, mimetype="image/png")
    except Exception:
        buffer = io.BytesIO(create_placeholder_chart_bytes())
        buffer.seek(0)
        return send_file(buffer, mimetype="image/png")


def run_desktop_gui() -> None:
    """Optional Tkinter desktop GUI for image file detection."""
    import tkinter as tk
    from tkinter import filedialog, messagebox

    root = tk.Tk()
    root.title(APP_NAME + " - Tkinter")
    root.geometry("460x240")

    info = tk.Label(root, text="Select image and run eye detection", font=("Arial", 14))
    info.pack(pady=18)

    result_label = tk.Label(root, text="No image selected", wraplength=420)
    result_label.pack(pady=10)

    def choose_image() -> None:
        path = filedialog.askopenfilename(filetypes=[("Images", "*.jpg *.jpeg *.png")])
        if not path:
            return
        image = cv2.imread(path)
        if image is None:
            messagebox.showerror("Error", "Could not read image.")
            return
        result = process_and_store(image, "tkinter")
        result_label.config(text=f"Faces: {result['faces']} | Eyes: {result['eyes']} | Saved: {result['image_path']}")
        messagebox.showinfo("Done", result["message"])

    button = tk.Button(root, text="Choose Image", command=choose_image, bg="#38bdf8", font=("Arial", 12, "bold"))
    button.pack(pady=12)
    root.mainloop()


if __name__ == "__main__":
    init_db()
    if "--desktop" in sys.argv:
        run_desktop_gui()
    else:
        print("Starting Eye Detection Web App...")
        print("Open this URL in browser: http://127.0.0.1:5000")
        app.run(debug=True, host="127.0.0.1", port=5000)
