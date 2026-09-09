import os
import base64
from flask import (Flask, render_template, Response,
                   jsonify, request, session)
from flask_wtf import FlaskForm
from wtforms import FileField, SubmitField
from wtforms.validators import InputRequired
from werkzeug.utils import secure_filename

import cv2
from YOLO_Video import video_detection, image_detection, process_single_frame, get_model

# ─── App Setup ────────────────────────────────────────────────────────────────

app = Flask(__name__)
app.config['SECRET_KEY'] = 'kanhaiya'
app.config['UPLOAD_FOLDER'] = 'static/files'
app.config['MAX_CONTENT_LENGTH'] = 64 * 1024 * 1024  # 64MB max upload limit

# Create upload folder if it doesn't exist
upload_dir = os.path.join(os.path.abspath(os.path.dirname(__file__)), app.config['UPLOAD_FOLDER'])
os.makedirs(upload_dir, exist_ok=True)

ALLOWED_VIDEO = {'mp4', 'avi', 'mov', 'mkv', 'webm'}
ALLOWED_IMAGE = {'jpg', 'jpeg', 'png', 'bmp', 'gif', 'webp'}

def allowed_extension(filename):
    ext = filename.rsplit('.', 1)[-1].lower() if '.' in filename else ''
    return ext, ext in ALLOWED_VIDEO or ext in ALLOWED_IMAGE

def is_video_ext(ext):
    return ext in ALLOWED_VIDEO

# Pre-load model at application startup to avoid first-request timeout/503
try:
    print("🚀 Pre-warming YOLO model on app startup...")
    get_model()
except Exception as e:
    print(f"⚠️ Model pre-load info: {e}")

# ─── Form ─────────────────────────────────────────────────────────────────────

class UploadFileForm(FlaskForm):
    file = FileField("File", validators=[InputRequired()])
    submit = SubmitField("Run")

def generate_frames(path_x=''):
    """Generator for MJPEG video streaming."""
    try:
        for frame in video_detection(path_x):
            _, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')
    except Exception as e:
        print(f"⚠️ Stream exception: {e}")

# ─── Routes ───────────────────────────────────────────────────────────────────

@app.route('/', methods=['GET', 'POST'])
@app.route('/home', methods=['GET', 'POST'])
def home():
    session.clear()
    return render_template('indexproject.html')


@app.route('/FrontPage', methods=['GET', 'POST'])
def front():
    form = UploadFileForm()
    image_result = None
    file_type = session.get('file_type', 'video')
    error = None

    if request.method == 'POST':
        if form.validate_on_submit():
            file = form.file.data
            filename = secure_filename(file.filename)
            if not filename:
                return render_template('videoprojectnew.html', form=form,
                                       error="❌ Invalid file selected.", file_type=file_type)

            ext, allowed = allowed_extension(filename)
            if not allowed:
                return render_template('videoprojectnew.html', form=form,
                                       error="❌ Unsupported file type. Please upload MP4, AVI, JPG, or PNG.",
                                       file_type=file_type)

            os.makedirs(upload_dir, exist_ok=True)
            save_path = os.path.join(upload_dir, filename)
            file.save(save_path)

            if is_video_ext(ext):
                session['video_path'] = save_path
                session['file_type'] = 'video'
                file_type = 'video'
            else:
                session['file_type'] = 'image'
                file_type = 'image'
                try:
                    annotated_img, alert = image_detection(save_path)
                    _, buffer = cv2.imencode('.jpg', annotated_img, [cv2.IMWRITE_JPEG_QUALITY, 85])
                    image_result = base64.b64encode(buffer).decode('utf-8')
                    session['last_alert'] = alert
                except Exception as e:
                    print(f"Detection error: {e}")
                    error = f"❌ Detection error: {str(e)}"
        else:
            error = "❌ Please select a file before clicking Run."

    return render_template('videoprojectnew.html', form=form,
                           image_result=image_result, file_type=file_type, error=error)


@app.route('/video')
def video():
    """MJPEG stream for uploaded video."""
    path = session.get('video_path', None)
    if not path or not os.path.exists(path):
        return "Video not found or session expired. Please upload again.", 404
    return Response(generate_frames(path_x=path),
                    mimetype='multipart/x-mixed-replace; boundary=frame')


# ─── Live Webcam (Browser-based WebRTC) ───────────────────────────────────────

@app.route('/webcam', methods=['GET', 'POST'])
def webcam():
    session.clear()
    return render_template('ui.html')


@app.route('/process_frame', methods=['POST'])
def process_frame():
    """
    Receives a single JPEG frame from the browser webcam (via fetch/AJAX),
    runs YOLO detection, and returns the annotated JPEG + alert status.
    """
    if 'frame' not in request.files:
        return jsonify({'error': 'No frame provided'}), 400

    frame_file = request.files['frame']
    frame_bytes = frame_file.read()

    try:
        annotated_bytes, alert = process_single_frame(frame_bytes)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

    b64 = base64.b64encode(annotated_bytes).decode('utf-8')
    return jsonify({'frame': b64, 'alert': alert})


# ─── Alert Status ─────────────────────────────────────────────────────────────

@app.route('/alert_status')
def get_alert_status():
    """Returns current alert status."""
    return jsonify({'alert': session.get('last_alert', False)})


# ─── Entry Point ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    app.run(debug=True)
