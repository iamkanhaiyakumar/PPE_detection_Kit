import streamlit as st
import cv2
import os
from werkzeug.utils import secure_filename
from YOLO_Video import video_detection

def save_uploaded_file(uploaded_file, save_folder):
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)

    file_path = os.path.join(save_folder, secure_filename(uploaded_file.name))
    with open(file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    return file_path

def main():
    st.title("YOLO Object Detection with Streamlit")

    # Upload video file
    uploaded_file = st.file_uploader("Upload a video", type=["mp4", "avi", "mov", "mkv"])

    if uploaded_file is not None:
        st.success("File uploaded successfully!")

        # Save the uploaded video
        save_folder = "static/files"
        video_path = save_uploaded_file(uploaded_file, save_folder)

        st.video(video_path)

        if st.button("Run Object Detection"):
            st.info("Running YOLO object detection...")

            # Perform object detection
            for frame in video_detection(video_path):
                _, buffer = cv2.imencode(".jpg", frame)
                st.image(buffer.tobytes(), channels="BGR")

    # Real-time webcam feed
    if st.button("Start Webcam Detection"):
        st.info("Starting webcam feed...")
        cap = cv2.VideoCapture(0)

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                st.error("Failed to capture webcam feed")
                break

            # Apply YOLO on webcam frames
            yolo_frames = video_detection(0)

            for detection_ in yolo_frames:
                _, buffer = cv2.imencode(".jpg", detection_)
                st.image(buffer.tobytes(), channels="BGR")

        cap.release()

if __name__ == "__main__":
    main()