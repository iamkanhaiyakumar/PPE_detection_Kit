# from ultralytics import YOLO
# import cv2
# import math
# import os

# # -------------------- FACE + ATTENDANCE --------------------
# from deepface import DeepFace
# import csv
# from datetime import datetime

# known_faces_path = "known_faces"
# attendance_file = "attendance.csv"
# marked_today = set()

# # -------------------- AUTO CREATE ATTENDANCE FILE --------------------
# if not os.path.exists(attendance_file):
#     with open(attendance_file, "w", newline="") as f:
#         writer = csv.writer(f)
#         writer.writerow(["Name", "Date", "Time", "Attendance", "PPE_Status"])
# # ---------------------------------------------------------------------


# def mark_attendance(name, ppe_ok):

#     today = datetime.now().strftime("%Y-%m-%d")

#     if (name, today) in marked_today:
#         return

#     now = datetime.now()
#     attendance_status = "Present"
#     ppe_status = "Full PPE" if ppe_ok else "NO PPE"

#     with open(attendance_file, "a", newline="") as f:
#         writer = csv.writer(f)
#         writer.writerow([
#             name,
#             now.strftime("%Y-%m-%d"),
#             now.strftime("%H:%M:%S"),
#             attendance_status,
#             ppe_status
#         ])

#     marked_today.add((name, today))
#     print(f"✅ Attendance marked for {name} | PPE: {ppe_status}")


# def video_detection(path_x):

#     print(f"🎥 Input Value: {path_x}, Type: {type(path_x)}")

#     # -------------------- INPUT VALIDATION --------------------
#     if isinstance(path_x, int):
#         print("📸 Using webcam input.")
#     elif isinstance(path_x, str):
#         if not os.path.isfile(path_x):
#             raise FileNotFoundError(f"❌ File not found: {path_x}")
#     else:
#         raise TypeError("⚠️ Invalid input type. Expected int (webcam) or str (video path).")

#     cap = cv2.VideoCapture(path_x)

#     if not cap.isOpened():
#         raise ValueError(f"❌ Unable to open video source: {path_x}")

#     print("✅ Video source opened successfully.")

#     # -------------------- LOAD YOLO MODEL --------------------
#     model = YOLO("YOLO-Weights/ppe.pt")

#     classNames = [
#         'Hardhat', 'Mask', 'NO-Hardhat', 'NO-Mask',
#         'NO-Safety Vest', 'Person', 'Safety Cone',
#         'Safety Vest', 'machinery', 'vehicle'
#     ]

#     # -------------------- MAIN LOOP --------------------
#     while True:

#         success, img = cap.read()

#         if not success:
#             print("🚫 No frame captured. Exiting.")
#             break

#         results = model(img, stream=True)

#         # PPE tracking flags
#         hardhat = False
#         mask = False
#         vest = False

#         for r in results:
#             boxes = r.boxes

#             for box in boxes:

#                 x1, y1, x2, y2 = map(int, box.xyxy[0])
#                 conf = round(float(box.conf[0]), 2)
#                 class_id = int(box.cls[0])
#                 class_name = classNames[class_id]
#                 label = f"{class_name} {conf}"

#                 # Track PPE presence
#                 if class_name == 'Hardhat':
#                     hardhat = True
#                 if class_name == 'Mask':
#                     mask = True
#                 if class_name == 'Safety Vest':
#                     vest = True

#                 # Color logic
#                 if class_name in ['Mask', 'Hardhat', 'Safety Vest']:
#                     color = (0, 255, 0)
#                 elif class_name in ['NO-Hardhat', 'NO-Mask', 'NO-Safety Vest']:
#                     color = (0, 0, 255)
#                 elif class_name in ['machinery', 'vehicle']:
#                     color = (0, 149, 255)
#                 else:
#                     color = (85, 45, 255)

#                 if conf > 0.5:
#                     cv2.rectangle(img, (x1, y1), (x2, y2), color, 3)
#                     cv2.putText(
#                         img,
#                         label,
#                         (x1, y1 - 10),
#                         cv2.FONT_HERSHEY_SIMPLEX,
#                         0.7,
#                         (255, 255, 255),
#                         2
#                     )

#         # Final PPE status
#         ppe_ok = hardhat and mask and vest

#         # -------------------- FACE + ATTENDANCE --------------------
#         try:
#             for filename in os.listdir(known_faces_path):

#                 person_name = os.path.splitext(filename)[0]
#                 person_image_path = os.path.join(known_faces_path, filename)

#                 result = DeepFace.verify(
#                     img1_path=img,
#                     img2_path=person_image_path,
#                     enforce_detection=False
#                 )

#                 if result["verified"]:

#                     mark_attendance(person_name, ppe_ok)

#                     status_text = "Full PPE" if ppe_ok else "NO PPE"

#                     cv2.putText(
#                         img,
#                         f"{person_name} - {status_text}",
#                         (20, 40),
#                         cv2.FONT_HERSHEY_SIMPLEX,
#                         1,
#                         (0, 255, 0) if ppe_ok else (0, 0, 255),
#                         3
#                     )

#                     break

#         except Exception as e:
#             print("Face recognition error:", e)
#         # ----------------------------------------------------------

#         yield img

#     cap.release()
#     cv2.destroyAllWindows()
















from ultralytics import YOLO
import cv2
import math
import os

def video_detection(path_x):
    # Debug: Check input value and type
    print(f"🎥 Input Value: {path_x}, Type: {type(path_x)}")

    # Validate input (supports webcam or video file path)
    if isinstance(path_x, int):
        print("📸 Using webcam input.")
    elif isinstance(path_x, str):
        if not os.path.isfile(path_x):
            raise FileNotFoundError(f"❌ File not found: {path_x}")
    else:
        raise TypeError("⚠️ Invalid input type. Expected int (webcam) or str (video path).")

    # Capture video
    cap = cv2.VideoCapture(path_x)
    if not cap.isOpened():
        raise ValueError(f"❌ Unable to open video source: {path_x}")

    print("✅ Video source opened successfully.")

    # Load YOLO model
    model = YOLO("YOLO-Weights/ppe.pt")
    classNames = ['Hardhat', 'Mask', 'NO-Hardhat', 'NO-Mask', 'NO-Safety Vest', 'Person',
                  'Safety Cone', 'Safety Vest', 'machinery', 'vehicle']

    while True:
        success, img = cap.read()
        if not success:
            print("🚫 No frame captured. Exiting.")
            break

        results = model(img, stream=True)

        for r in results:
            boxes = r.boxes
            for box in boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                conf = round(float(box.conf[0]), 2)
                class_id = int(box.cls[0])
                class_name = classNames[class_id]
                label = f"{class_name} {conf}"

                # Set color based on class
                if class_name in ['Mask', 'Hardhat', 'Safety Vest']:
                    color = (0, 255, 0)  # Green for safety gear
                elif class_name in ['NO-Hardhat', 'NO-Mask', 'NO-Safety Vest']:
                    color = (0, 0, 255)  # Red for missing gear
                elif class_name in ['machinery', 'vehicle']:
                    color = (0, 149, 255)  # Orange for machinery
                else:
                    color = (85, 45, 255)  # Purple for others

                if conf > 0.5:
                    cv2.rectangle(img, (x1, y1), (x2, y2), color, 3)
                    cv2.putText(img, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        yield img

    cap.release()
    cv2.destroyAllWindows()









# from ultralytics import YOLO
# import cv2
# import math

# def video_detection(path_x):
#     video_capture = path_x
#     #Create a Webcam Object
#     cap=cv2.VideoCapture(video_capture)
#     frame_width=int(cap.get(3))
#     frame_height=int(cap.get(4))
#     #out=cv2.VideoWriter('output.avi', cv2.VideoWriter_fourcc('M', 'J', 'P','G'), 10, (frame_width, frame_height))

#     model=YOLO("YOLO-Weights/ppe.pt")
#     classNames = ['Hardhat', 'Mask', 'NO-Hardhat', 'NO-Mask', 'NO-Safety Vest', 'Person', 'Safety Cone',
#                 'Safety Vest', 'machinery', 'vehicle']
#     while True:
#         success, img = cap.read()
#         results=model(img,stream=True)
#         for r in results:
#             boxes=r.boxes
#             for box in boxes:
#                 x1,y1,x2,y2=box.xyxy[0]
#                 x1,y1,x2,y2=int(x1), int(y1), int(x2), int(y2)
#                 print(x1,y1,x2,y2)
#                 conf=math.ceil((box.conf[0]*100))/100
#                 cls=int(box.cls[0])
#                 class_name=classNames[cls]
#                 label=f'{class_name}{conf}'
#                 t_size = cv2.getTextSize(label, 0, fontScale=1, thickness=2)[0]
#                 print(t_size)
#                 c2 = x1 + t_size[0], y1 - t_size[1] - 3
#                 if class_name == 'Mask' or class_name == 'Hardhat' or class_name == 'Safety Vest':
#                     color=(0, 255,0)

#                 elif class_name == 'NO-Hardhat' or class_name == 'NO-Mask' or class_name == 'NO-Safety Vest':
#                     color = (0,0,255)

#                 elif class_name == 'machinery' or class_name == 'vehicle':
#                     color = (0, 149, 255)
#                 else:
#                     color = (85,45,255)
#                 if conf>0.5:
#                     cv2.rectangle(img, (x1,y1), (x2,y2), color,3)
#                     cv2.rectangle(img, (x1,y1), c2, color, -1, cv2.LINE_AA)  # filled
#                     cv2.putText(img, label, (x1,y1-2),0, 1,[255,255,255], thickness=1,lineType=cv2.LINE_AA)

#         yield img
#         #out.write(img)
#         #cv2.imshow("image", img)
#         #if cv2.waitKey(1) & 0xFF==ord('1'):
#             #break
#     #out.release()
# cv2.destroyAllWindows()








# from ultralytics import YOLO
# import cv2
# import math
# import threading
# from playsound import playsound

# # Flag to prevent overlapping alerts
# alert_playing = False

# def play_alert():
#     """Play alert sound."""
#     global alert_playing
#     alert_playing = True
#     playsound("static/alert.mp3", block=False)  # Path to the alert sound file
#     alert_playing = False

# def video_detection(path_x):
#     video_capture = path_x
#     cap = cv2.VideoCapture(video_capture)
#     frame_width = int(cap.get(3))
#     frame_height = int(cap.get(4))
    
#     # Load the YOLO model
#     model = YOLO("YOLO-Weights/ppe.pt")
#     classNames = ['Hardhat', 'Mask', 'NO-Hardhat', 'NO-Mask', 'NO-Safety Vest', 'Person', 'Safety Cone',
#                   'Safety Vest', 'machinery', 'vehicle']
    
#     global alert_playing

#     while True:
#         success, img = cap.read()
#         if not success:
#             break

#         results = model(img, stream=True)
#         alert_triggered = False

#         for r in results:
#             boxes = r.boxes
#             for box in boxes:
#                 x1, y1, x2, y2 = box.xyxy[0]
#                 x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

#                 conf = math.ceil((box.conf[0] * 100)) / 100
#                 cls = int(box.cls[0])
#                 class_name = classNames[cls]
#                 label = f'{class_name}{conf}'

#                 # Set bounding box colors
#                 if class_name == 'Mask' or class_name == 'Hardhat' or class_name == 'Safety Vest':
#                     color = (0, 255, 0)
#                 elif class_name == 'NO-Hardhat' or class_name == 'NO-Mask' or class_name == 'NO-Safety Vest':
#                     color = (0, 0, 255)
#                     alert_triggered = True  # Alert condition
#                 elif class_name == 'machinery' or class_name == 'vehicle':
#                     color = (0, 149, 255)
#                 else:
#                     color = (85, 45, 255)

#                 # Draw bounding boxes and labels
#                 if conf > 0.5:
#                     cv2.rectangle(img, (x1, y1), (x2, y2), color, 3)
#                     t_size = cv2.getTextSize(label, 0, fontScale=1, thickness=2)[0]
#                     c2 = x1 + t_size[0], y1 - t_size[1] - 3
#                     cv2.rectangle(img, (x1, y1), c2, color, -1, cv2.LINE_AA)  # Filled box for text
#                     cv2.putText(img, label, (x1, y1 - 2), 0, 1, [255, 255, 255], thickness=1, lineType=cv2.LINE_AA)

#         # Trigger alert sound
#         if alert_triggered and not alert_playing:
#             threading.Thread(target=play_alert).start()

#         # Yield the processed frame
#         yield img

#     cap.release()
#     cv2.destroyAllWindows()
