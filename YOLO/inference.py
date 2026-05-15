from IPython.display import display, Javascript, Image
from google.colab.output import eval_js
from base64 import b64decode
import numpy as np
import cv2
from ultralytics import YOLO
import PIL.Image
import io, time

model = YOLO("/content/drive/MyDrive/face-recognition/runs/yolov8n_faces/weights/best.pt")

# ── Single JS block: camera + frame capture all in one persistent scope ────────
def setup_camera_js():
    display(Javascript('''
        // Global camera setup — persists across eval_js calls
        window._webcamStream  = null;
        window._webcamVideo   = null;
        window._cameraReady   = false;

        (async () => {
            // Clean up previous run
            const oldDiv = document.getElementById('yolo_cam_div');
            if (oldDiv) oldDiv.remove();
            if (window._webcamStream) {
                window._webcamStream.getTracks().forEach(t => t.stop());
            }

            const div = document.createElement('div');
            div.id = 'yolo_cam_div';
            document.body.appendChild(div);

            const video = document.createElement('video');
            video.id              = 'yolo_video';
            video.autoplay        = true;
            video.playsInline     = true;
            video.muted           = true;
            video.width           = 640;
            video.height          = 480;
            video.style.border    = '2px solid #4CAF50';
            div.appendChild(video);

            // Status text
            const status = document.createElement('p');
            status.id        = 'yolo_status';
            status.innerText = '⏳ Requesting camera...';
            div.appendChild(status);

            try {
                const stream = await navigator.mediaDevices.getUserMedia({
                    video: { width: 640, height: 480, facingMode: 'user' }
                });

                window._webcamStream = stream;
                window._webcamVideo  = video;
                video.srcObject      = stream;

                await new Promise((resolve, reject) => {
                    video.onloadedmetadata = resolve;
                    setTimeout(reject, 10000, 'metadata timeout');
                });

                await video.play();
                await new Promise(r => setTimeout(r, 2000)); // warm-up

                window._cameraReady    = true;
                status.innerText       = '✅ Camera ready — Python will now capture frames';
                status.style.color     = 'green';
                console.log('Camera ready:', video.videoWidth, 'x', video.videoHeight);

            } catch(err) {
                status.innerText   = '❌ Camera error: ' + err;
                status.style.color = 'red';
                window._cameraReady = false;
                console.error(err);
            }
        })();
    '''))

# ── Capture using the persistent window._webcamVideo reference ─────────────────
CAPTURE_JS = '''
(function() {
    const video = window._webcamVideo;
    if (!video)                  return "ERR:no_video_element";
    if (!window._cameraReady)    return "ERR:not_ready";
    if (video.readyState < 2)    return "ERR:readyState=" + video.readyState;
    if (video.videoWidth === 0)  return "ERR:zero_width";

    const canvas = document.createElement('canvas');
    canvas.width  = video.videoWidth;
    canvas.height = video.videoHeight;
    const ctx     = canvas.getContext('2d');
    ctx.drawImage(video, 0, 0);

    const data = canvas.toDataURL('image/jpeg', 0.85);
    return data.length > 1000 ? data : "ERR:blank_frame";
})()
'''

def capture_frame():
    result = eval_js(CAPTURE_JS)
    if result.startswith("ERR:"):
        raise ValueError(result)
    b64   = result.split(',')[1]
    arr   = np.frombuffer(b64decode(b64), dtype=np.uint8)
    frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if frame is None:
        raise ValueError("cv2 decode failed")
    return frame

# ── Wait for camera ready flag ─────────────────────────────────────────────────
def wait_for_camera(timeout=25):
    print("Waiting for camera", end="")
    for _ in range(timeout * 2):
        try:
            ready = eval_js("window._cameraReady === true ? '1' : '0'")
            if ready == '1':
                print(" ✅\n")
                return True
        except:
            pass
        print(".", end="", flush=True)
        time.sleep(0.5)
    print("\n❌ Timed out. Check browser camera permission.")
    return False

# ── YOLO detection + drawing ───────────────────────────────────────────────────
COLORS = [(255,56,56),(56,255,56),(56,56,255),(255,200,56),(255,56,200),(56,255,200)]

def run_detection(frame, conf=0.4):
    results   = model.predict(frame, conf=conf, verbose=False)[0]
    annotated = frame.copy()
    for box in results.boxes:
        cls_id       = int(box.cls[0])
        confidence   = float(box.conf[0])
        label        = f"{model.names[cls_id]}  {confidence:.0%}"
        x1,y1,x2,y2 = map(int, box.xyxy[0])
        color        = COLORS[cls_id % len(COLORS)]
        cv2.rectangle(annotated, (x1,y1), (x2,y2), color, 2)
        (tw,th),_    = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.65, 2)
        cv2.rectangle(annotated, (x1, y1-th-10), (x1+tw+8, y1), color, -1)
        cv2.putText(annotated, label, (x1+4, y1-5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255,255,255), 2)
    cv2.putText(annotated, f"Faces: {len(results.boxes)}", (10,30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,255), 2)
    return annotated

def to_display(frame):
    buf = io.BytesIO()
    PIL.Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)).save(buf, 'JPEG', quality=85)
    return Image(data=buf.getvalue())

# ── Main detection loop ────────────────────────────────────────────────────────
from IPython.display import clear_output

def run_live_detection(num_frames=300, conf=0.4, fps_limit=8):
    setup_camera_js()
    if not wait_for_camera(timeout=25):
        return

    delay        = 1.0 / fps_limit
    error_streak = 0

    for i in range(num_frames):
        t0 = time.time()
        try:
            frame     = capture_frame()
            annotated = run_detection(frame, conf=conf)
            clear_output(wait=True)
            display(to_display(annotated))
            print(f"Frame {i+1}/{num_frames}  |  conf={conf}  |  Interrupt kernel to stop")
            error_streak = 0

        except Exception as e:
            error_streak += 1
            print(f"⚠️  Frame error ({error_streak}): {e}")
            if error_streak >= 15:
                print("❌ Too many errors — stopping.")
                break
            time.sleep(0.3)
            continue

        elapsed = time.time() - t0
        if elapsed < delay:
            time.sleep(delay - elapsed)

    # Stop the camera stream when done
    eval_js('''
        if (window._webcamStream)
            window._webcamStream.getTracks().forEach(t => t.stop());
        const d = document.getElementById('yolo_cam_div');
        if (d) d.remove();
    ''')
    print("\n✅ Detection complete — camera stopped.")

# ── RUN ───────────────────────────────────────────────────────────────────────
run_live_detection(num_frames=300, conf=0.4, fps_limit=8)