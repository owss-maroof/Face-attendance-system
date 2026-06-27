"""
04_attendance.py
-----------------
Real-time face recognition attendance system.

Loads the fine-tuned student model, opens the webcam,
recognises students in real time, and logs attendance
with timestamps to a CSV file.

Usage:
    python 04_attendance.py
    python 04_attendance.py --source 0 --threshold 0.75

Controls:
    Q → quit
    S → save attendance manually
"""

import os
import cv2
import csv
import json
import pickle
import argparse
import numpy as np
from datetime import datetime
import tensorflow as tf


# ── Config ─────────────────────────────────────────────────────────────────
IMG_SIZE       = 160
THRESHOLD      = 0.75   # confidence below this → Unknown
COOLDOWN_SECS  = 30     # seconds before same student is marked again

FACE_CASCADE = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)


# ── Load Model & Labels ────────────────────────────────────────────────────
def load_resources():
    model_path    = "models/student_model.keras"
    labels_path   = "models/student_labels.json"
    encodings_path = "models/student_encodings.pkl"

    if not os.path.exists(model_path):
        raise FileNotFoundError(
            "student_model.keras not found!\n"
            "Run 03_finetune_students.py first."
        )

    print("[INFO] Loading model...")
    model = tf.keras.models.load_model(model_path)

    with open(labels_path, "r") as f:
        labels = json.load(f)

    with open(encodings_path, "rb") as f:
        encodings = pickle.load(f)

    print(f"[INFO] Loaded model for {len(labels)} students")
    return model, labels, encodings


# ── Attendance Log ─────────────────────────────────────────────────────────
def get_log_path():
    today = datetime.now().strftime("%Y-%m-%d")
    os.makedirs("attendance", exist_ok=True)
    return f"attendance/{today}_attendance.csv"


def init_log(log_path, labels):
    if not os.path.exists(log_path):
        with open(log_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["Name", "Time", "Date"])
        print(f"[INFO] Created attendance log: {log_path}")


def mark_attendance(log_path, name, marked):
    now  = datetime.now()
    time = now.strftime("%H:%M:%S")
    date = now.strftime("%Y-%m-%d")

    last_marked = marked.get(name, 0)
    if (now.timestamp() - last_marked) < COOLDOWN_SECS:
        return False

    with open(log_path, "a", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([name, time, date])

    marked[name] = now.timestamp()
    print(f"[✓] Marked attendance: {name} at {time}")
    return True


# ── Face Recognition ───────────────────────────────────────────────────────
def recognise_face(face_img, model, labels):
    img = cv2.resize(face_img, (IMG_SIZE, IMG_SIZE))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img.astype(np.float32) / 255.0
    img = np.expand_dims(img, axis=0)

    predictions = model.predict(img, verbose=0)[0]
    confidence  = np.max(predictions)
    class_idx   = np.argmax(predictions)

    if confidence < THRESHOLD:
        return "Unknown", confidence

    return labels[class_idx], confidence


# ── Main Loop ──────────────────────────────────────────────────────────────
def run(args):
    model, labels, encodings = load_resources()

    log_path = get_log_path()
    init_log(log_path, labels)

    marked = {}  # tracks last marked time per student

    try:
        source = int(args.source)
    except ValueError:
        source = args.source

    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open camera: {source}")

    print(f"\n[INFO] Attendance system running")
    print(f"[INFO] Log: {log_path}")
    print("[INFO] Press Q to quit\n")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        gray  = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = FACE_CASCADE.detectMultiScale(gray, 1.1, 5, minSize=(80, 80))

        for (x, y, w, h) in faces:
            face_crop = frame[y:y+h, x:x+w]
            name, confidence = recognise_face(face_crop, model, labels)

            # Mark attendance
            if name != "Unknown":
                just_marked = mark_attendance(log_path, name, marked)
                color = (0, 255, 0) if just_marked else (255, 165, 0)
            else:
                color = (0, 0, 255)

            # Draw box and label
            cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
            label_text = f"{name} ({confidence*100:.0f}%)"
            cv2.putText(frame, label_text, (x, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.65, color, 2)

        # Show attendance count
        total    = len(marked)
        students = len(labels)
        cv2.putText(frame, f"Present: {total}/{students}",
                    (10, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)

        # Show date and time
        now_str = datetime.now().strftime("%Y-%m-%d  %H:%M:%S")
        cv2.putText(frame, now_str, (10, frame.shape[0] - 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 200, 200), 1)

        cv2.imshow("Attendance System — Q to quit", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()

    print(f"\n[INFO] Session ended")
    print(f"[INFO] Total present: {len(marked)} students")
    print(f"[INFO] Attendance saved to: {log_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--source",    default="0", help="Camera index or video path")
    parser.add_argument("--threshold", type=float, default=THRESHOLD,
                        help="Confidence threshold (default 0.75)")
    args = parser.parse_args()

    THRESHOLD = args.threshold

    gpus = tf.config.list_physical_devices("GPU")
    if gpus:
        tf.config.experimental.set_memory_growth(gpus[0], True)
        print(f"[INFO] GPU detected: {gpus[0].name}")

    run(args)
