from flask import Flask, render_template, Response, jsonify, request, session
from flask_wtf import FlaskForm
from wtforms import FileField, SubmitField
from werkzeug.utils import secure_filename
from wtforms.validators import InputRequired
import os
import cv2
import smtplib
import time
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart

# YOLO detection function
from YOLO_Video import video_detection

app = Flask(__name__)

app.config['SECRET_KEY'] = 'kanhaiya'
app.config['UPLOAD_FOLDER'] = 'static/files'

# -------------------- ALERT GLOBALS --------------------
alert_status = False
last_email_time = 0
EMAIL_INTERVAL = 60  # seconds between emails

# -------------------- EMAIL CONFIG --------------------
EMAIL_ADDRESS = "kanhaiyak0104@gmail.com"        # 🔴 CHANGE THIS
EMAIL_PASSWORD = "avpxagfbbbskoasi"         # 🔴 CHANGE THIS
RECEIVER_EMAIL = "receiver_email@gmail.com"  # 🔴 CHANGE THIS


# -------------------- EMAIL FUNCTION --------------------
def send_alert_email():
    global last_email_time

    current_time = time.time()

    if current_time - last_email_time < EMAIL_INTERVAL:
        return  # Prevent spam

    try:
        msg = MIMEMultipart()
        msg['From'] = EMAIL_ADDRESS
        msg['To'] = RECEIVER_EMAIL
        msg['Subject'] = "🚨 PPE VIOLATION ALERT"

        body = "Warning!\n\nPPE Violation Detected.\nPlease check the monitoring system immediately."
        msg.attach(MIMEText(body, 'plain'))

        server = smtplib.SMTP('smtp.gmail.com', 587)
        server.starttls()
        server.login(EMAIL_ADDRESS, EMAIL_PASSWORD)
        server.send_message(msg)
        server.quit()

        print("✅ Alert Email Sent!")
        last_email_time = current_time

    except Exception as e:
        print("❌ Email Error:", e)


# -------------------- FORM --------------------
class UploadFileForm(FlaskForm):
    file = FileField("File", validators=[InputRequired()])
    submit = SubmitField("Run")


# -------------------- FRAME GENERATORS --------------------
def generate_frames(path_x=''):
    global alert_status
    yolo_output = video_detection(path_x)

    for detection_ in yolo_output:

        # 🔔 Trigger Alert
        alert_status = True
        send_alert_email()

        ref, buffer = cv2.imencode('.jpg', detection_)
        frame = buffer.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')


def generate_frames_web(path_x):
    global alert_status
    yolo_output = video_detection(path_x)

    for detection_ in yolo_output:

        alert_status = True
        send_alert_email()

        ref, buffer = cv2.imencode('.jpg', detection_)
        frame = buffer.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')


# -------------------- ROUTES --------------------
@app.route('/', methods=['GET', 'POST'])
@app.route('/home', methods=['GET', 'POST'])
def home():
    session.clear()
    return render_template('indexproject.html')


@app.route("/webcam", methods=['GET', 'POST'])
def webcam():
    session.clear()
    return render_template('ui.html')


@app.route('/FrontPage', methods=['GET', 'POST'])
def front():
    form = UploadFileForm()

    if form.validate_on_submit():
        file = form.file.data

        save_path = os.path.join(
            os.path.abspath(os.path.dirname(__file__)),
            app.config['UPLOAD_FOLDER'],
            secure_filename(file.filename)
        )

        file.save(save_path)
        session['video_path'] = save_path

    return render_template('videoprojectnew.html', form=form)


@app.route('/video')
def video():
    return Response(
        generate_frames(path_x=session.get('video_path', None)),
        mimetype='multipart/x-mixed-replace; boundary=frame'
    )


@app.route('/webapp')
def webapp():
    return Response(
        generate_frames_web(path_x=0),
        mimetype='multipart/x-mixed-replace; boundary=frame'
    )


# -------------------- ALERT STATUS API --------------------
@app.route('/alert_status')
def get_alert_status():
    global alert_status
    return jsonify({"alert": alert_status})


# -------------------- RUN APP --------------------
if __name__ == "__main__":
    app.run(debug=True)
