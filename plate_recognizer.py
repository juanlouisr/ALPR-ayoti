import yolov5
import base64
import io
from PIL import Image
import hashlib
import os
import cv2
import re
import numpy as np
from paddleocr import PaddleOCR
import easyocr
import time

reader = easyocr.Reader(['en'], verbose=False, recog_network='english_g2', gpu=True)
paddle = PaddleOCR(lang="ch", use_angle_cls=False, show_log=False)


def preprocess_image(src):
    normalize = cv2.normalize(
        src, np.zeros((src.shape[0], src.shape[1])), 0, 255, cv2.NORM_MINMAX
    )
    denoise = cv2.fastNlMeansDenoisingColored(
        normalize, h=10, hColor=10, templateWindowSize=7, searchWindowSize=15
    )
    grayscale = cv2.cvtColor(denoise, cv2.COLOR_BGR2GRAY)
    threshold = cv2.threshold(grayscale, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
    return threshold

def easy_ocr(src):

    image_array = np.array(src)

    # preprocessed = preprocess_image(image_array)
    image_array = preprocess_image(image_array)
    return reader.readtext(image_array)


def ocr_plate(src):
    # Preprocess the image for better OCR results
    image_array = np.array(src)
    # preprocessed = preprocess_image(image_array)

    # OCR the preprocessed image
    results = paddle.ocr(image_array, det=False, cls=False)

    # Get the best OCR result
    plate_text, ocr_confidence = max(
        results,
        key=lambda ocr_prediction: max(
            ocr_prediction,
            key=lambda ocr_prediction_result: ocr_prediction_result[1],  # type: ignore
        ),
    )[0]

    # Filter out anything but uppercase letters, digits, hypens and whitespace.
    # Also, remove hypens and whitespaces at the first and last positions
    plate_text_filtered = re.sub(r"[^A-Z0-9- ]", "", plate_text).strip("- ")

    return {"plate": plate_text_filtered, "ocr_conf": ocr_confidence}


def load_model():
    model = yolov5.load('./model/yolov5-plate-object-recognition.pt', device='cuda:8')
    model.conf = 0.25  # NMS confidence threshold
    model.iou = 0.45  # NMS IoU threshold
    model.agnostic = False  # NMS class-agnostic
    model.multi_label = False  # NMS multiple labels per box
    model.max_det = 1  # maximum number of detections per image
    
    return model

def process_image(img_base64):
    image_bytes = base64.b64decode(img_base64)
    image_io = io.BytesIO(image_bytes)
    image = Image.open(image_io)
    image_hash = hashlib.md5(image.tobytes()).hexdigest()
    output_filename = f"{image_hash}.jpg"
    temp_dir = "temp"
    os.makedirs(temp_dir, exist_ok=True)
    file_path = os.path.join(temp_dir, output_filename)
    image.save(file_path, "JPEG")
    
    return file_path, image, image_hash

def detect_objects(model, file_path):
    results = model(file_path, augment=True)
    return results.pred[0]

def save_cropped_image(image, boxes, image_hash):
    pred_dir = "prediction"
    prefix = "pred_"
    filename = f"{prefix}{image_hash}.jpg"
    os.makedirs(pred_dir, exist_ok=True)
    x1, y1, x2, y2 = boxes[0].tolist()
    cropped_image = image.crop((x1, y1, x2, y2))
    file_path = os.path.join(pred_dir, filename)
    cropped_image.save(file_path, "JPEG")
    
    return file_path, cropped_image


if __name__ == '__main__':
    img_base64 = ''
    start = time.time()
    model = load_model()
    file_path, image, image_hash = process_image(img_base64)
    predictions = detect_objects(model, file_path)
    cropped_path, cropped_image = save_cropped_image(image, predictions[:, :4], image_hash)
    print(easy_ocr(cropped_image))
    print(time.time() - start)