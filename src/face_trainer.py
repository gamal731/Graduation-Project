import cv2
import numpy as np
from PIL import Image
import os
import logging
from settings.settings import PATHS

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def get_images_and_labels(path: str):
    try:
        imagePaths = [os.path.join(path, f) for f in os.listdir(path)]
        faceSamples = []
        ids = []
        detector = cv2.CascadeClassifier(PATHS['cascade_file'])
        if detector.empty():
            raise ValueError("Error loading cascade classifier")

        for imagePath in imagePaths:
            PIL_img = Image.open(imagePath).convert('L')
            img_numpy = np.array(PIL_img, 'uint8')
            id = int(os.path.split(imagePath)[-1].split("-")[1])
            faces = detector.detectMultiScale(img_numpy)

            for (x, y, w, h) in faces:
                faceSamples.append(img_numpy[y:y+h, x:x+w])
                ids.append(id)

        return faceSamples, ids
    except Exception as e:
        logger.error(f"Error processing images: {e}")
        raise

if __name__ == "__main__":
    try:
        logger.info("Starting face recognition training...")
        recognizer = cv2.face.LBPHFaceRecognizer_create()
        faces, ids = get_images_and_labels(PATHS['image_dir'])
        if not faces or not ids:
            raise ValueError("No training data found")
        logger.info("Training model...")
        recognizer.train(faces, np.array(ids))
        recognizer.write(PATHS['trainer_file'])
        logger.info(f"Model trained with {len(np.unique(ids))} faces")
    except Exception as e:
        logger.error(f"An error occurred: {e}")
