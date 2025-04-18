import warnings
warnings.filterwarnings('ignore', category=UserWarning)

import cv2
import numpy as np
import json
import os
import logging
import time
from settings.settings import CAMERA, FACE_DETECTION, PATHS

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def initialize_camera(camera_index: int = 0, retries: int = 3, delay: float = 1.0) -> cv2.VideoCapture:
    for attempt in range(1, retries + 1):
        try:
            logger.info(f"Attempt {attempt} to initialize camera index {camera_index}")
            cam = cv2.VideoCapture(camera_index)
            if cam.isOpened():
                cam.set(cv2.CAP_PROP_FRAME_WIDTH, CAMERA['width'])
                cam.set(cv2.CAP_PROP_FRAME_HEIGHT, CAMERA['height'])
                logger.info("Camera initialized successfully")
                return cam
            else:
                logger.error("Could not open webcam on attempt %d", attempt)
        except Exception as e:
            logger.error(f"Error initializing camera on attempt {attempt}: {e}")
        time.sleep(delay)
    return None

def load_names(filename: str) -> dict:
    try:
        names_json = {}
        if os.path.exists(filename):
            with open(filename, 'r') as fs:
                content = fs.read().strip()
                if content:
                    names_json = json.loads(content)
        return names_json
    except Exception as e:
        logger.error(f"Error loading names: {e}")
        return {}

def run_face_recognition():
    try:
        logger.info("Starting face recognition system...")
        recognizer = cv2.face.LBPHFaceRecognizer_create()
        if not os.path.exists(PATHS['trainer_file']):
            raise ValueError("Trainer file not found. Please train the model first.")
        recognizer.read(PATHS['trainer_file'])

        face_cascade = cv2.CascadeClassifier(PATHS['cascade_file'])
        if face_cascade.empty():
            raise ValueError("Error loading cascade classifier")

        cam = initialize_camera(CAMERA['index'])
        if cam is None:
            raise ValueError("Failed to initialize camera after multiple attempts")

        names = load_names(PATHS['names_file'])
        if not names:
            logger.warning("No names loaded, recognition will be limited")

        logger.info("Press 'ESC' to exit.")

        while True:
            ret, img = cam.read()
            if not ret:
                logger.warning("Failed to grab frame, trying again...")
                continue

            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(
                gray,
                scaleFactor=FACE_DETECTION['scale_factor'],
                minNeighbors=FACE_DETECTION['min_neighbors'],
                minSize=FACE_DETECTION['min_size']
            )

            for (x, y, w, h) in faces:
                cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)
                face_roi = gray[y:y+h, x:x+w]
                face_id, confidence = recognizer.predict(face_roi)

                if confidence <= 100:
                    name = names.get(str(face_id), "Unknown")
                    confidence_text = f"{confidence:.1f}%"
                    cv2.putText(img, name, (x+5, y-5), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
                    cv2.putText(img, confidence_text, (x+5, y+h-5), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 1)
                else:
                    cv2.putText(img, "Uncertain", (x+5, y-5), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

            cv2.imshow('Face Recognition', img)

            if cv2.waitKey(1) & 0xFF == 27:
                break

        logger.info("Face recognition stopped")

    except Exception as e:
        logger.error(f"An error occurred: {e}")
    finally:
        if 'cam' in locals() and cam is not None:
            cam.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    run_face_recognition()