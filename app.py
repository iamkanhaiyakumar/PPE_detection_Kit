import os
import base64
from flask import (Flask, render_template, Response,
                   jsonify, request, session)
from flask_wtf import FlaskForm
from wtforms import FileField, SubmitField
from wtforms.validators import InputRequired
from werkzeug.utils import secure_filename

import cv2
from YOLO_Video import video_detection, image_detection, process_single_frame

# ─── App Setup ────────────────────────────────────────────────────────────────

app = Flask(__name__)
app.config['SECRET_KEY'] = 'kanhaiya'
app.config['UPLOAD_FOLDER'] = 'static/files'

# Create upload folder if it doesn't exist (important for Render)
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

ALLOWED_VIDEO = {'mp4', 'avi', 'mov', 'mkv', 'webm'}
ALLOWED_IMAGE = {'jpg', 'jpeg', 'png', 'bmp', 'gif', 'webp'}

def allowed_extension(filename):
    ext = filename.rsplit('.', 1)[-1].lower() if '.' in filename else ''
    return ext, ext in ALLOWED_VIDEO or ext in ALLOWED_IMAGE

def is_video_ext(ext):
    return ext in ALLOWED_VIDEO

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
        print(f"⚠️ Stream error: {e}")

# ─── Routes ───────────────────────────────────────────────────────────────────

@app.route('/', methods=['GET', 'POST'])
@app.route('/home', methods=['GET', 'POST'])
def home():
    session.clear()
    return render_template('indexproject.html')


@app.route('/FrontPage', methods=['GET', 'POST'])
def front():
    form = UploadFileForm()
    image_result = None  # base64 annotated image for image uploads
    file_type = session.get('file_type', 'video')

    if form.validate_on_submit():
        file = form.file.data
        filename = secure_filename(file.filename)
        ext, allowed = allowed_extension(filename)

        if not allowed:
            return render_template('videoprojectnew.html', form=form,
                                   error="❌ Unsupported file type.", file_type='video')

        save_path = os.path.join(
            os.path.abspath(os.path.dirname(__file__)),
            app.config['UPLOAD_FOLDER'], filename
        )
        file.save(save_path)

        if is_video_ext(ext):
            # Video → store path in session → stream via /video
            session['video_path'] = save_path
            session['file_type'] = 'video'
            file_type = 'video'
        else:
            # Image → run detection immediately → return base64 result
            session['file_type'] = 'image'
            file_type = 'image'
            try:
                annotated_img, alert = image_detection(save_path)
                _, buffer = cv2.imencode('.jpg', annotated_img)
                image_result = base64.b64encode(buffer).decode('utf-8')
                session['last_alert'] = alert
            except Exception as e:
                return render_template('videoprojectnew.html', form=form,
                                       error=f"❌ Detection error: {str(e)}", file_type='image')

    return render_template('videoprojectnew.html', form=form,
                           image_result=image_result, file_type=file_type)


@app.route('/video')
def video():
    """MJPEG stream for uploaded video."""
    path = session.get('video_path', None)
    if not path or not os.path.exists(path):
        return "Video not found or expired. Please upload again.", 404
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
    Works on Render because the browser provides the camera, not the server.
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
    """Returns current alert status (used by video page polling)."""
    return jsonify({'alert': session.get('last_alert', False)})


# ─── Entry Point ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    app.run(debug=True)
