import cv2
import numpy as np
from ultralytics import YOLO
import math
from collections import deque

# Cargar el modelo YOLO
model = YOLO("yolov8n-seg.pt")

def detect_mouse(frame, results):
    phone_label = 'mouse'
    phone_bboxes = []
    phone_masks = []
    
    for result in results:
        for c in result:
            label = c.names[int(c.boxes.cls.tolist().pop())]
            bbox = c.boxes.xyxy[0].cpu().numpy().astype(int)
            
            if label == phone_label:
                phone_bboxes.append(bbox)
                phone_masks.append(c.masks.xy.pop().astype(np.int32).reshape(-1, 1, 2))
    
    if phone_bboxes:
        print("Cell phone detected")
        return phone_bboxes[0], phone_masks[0]  # Retorna solo la primera detección de teléfono
    else:
        return None, None

def detect_qr_and_color(frame, qr_detector):
    data, points, _ = qr_detector.detectAndDecode(frame)
    
    if data in ["RED", "GREEN", "BLUE"]:
        return data, points
    return "", []

def process_frame(input_frame, qr_detector):
    results_yolo = model.predict(source=input_frame)
    person_bboxes = []
    person_masks = []
    for result in results_yolo:
        for r in result:
            print(r)
            label = r.names[int(r.boxes.cls.tolist().pop())]
            bbox = r.boxes.xyxy[0].cpu().numpy().astype(int)
            
            if label == 'person':
                person_bboxes.append(bbox)
                person_masks.append(r.masks.xy.pop().astype(np.int32).reshape(-1, 1, 2))

    input_frame_gray = cv2.cvtColor(input_frame, cv2.COLOR_BGR2GRAY)
    qr_data, qr_points = detect_qr_and_color(input_frame_gray, qr_detector)
    
    background = ""
    if qr_data != "":
        background = "GRAYSCALE"
    else:
        background = "COLOR"

    
    processed_frame = None
    if background == "COLOR": ## fondo de color
        processed_frame = input_frame.copy()
    else: ## fondo blanco y negro
        processed_frame = cv2.cvtColor(input_frame, cv2.COLOR_BGR2GRAY)
        processed_frame = cv2.cvtColor(processed_frame, cv2.COLOR_GRAY2BGR)

    for (bbox, mask) in zip(person_bboxes, person_masks):
        x1, y1, x2, y2 = bbox
        
        ## Dibujar rectangulo y label
        cv2.rectangle(processed_frame, (x1, y1), (x2, y2), (128, 0, 128), 2)
        cv2.putText(processed_frame, "Futuro Cachimbo UTEC", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (128, 0, 128), 2, cv2.LINE_AA)
        ## dibujar la silueta de la persona
        b_mask = np.zeros(input_frame.shape[:2], np.uint8)
        cv2.drawContours(b_mask, [mask], -1, (255, 255, 255), cv2.FILLED)
        color_mask = np.zeros_like(input_frame)
        color_mask[b_mask == 255] = (255, 0, 255)

        processed_frame = cv2.bitwise_and(processed_frame, processed_frame, mask=cv2.bitwise_not(b_mask))
        processed_frame = cv2.add(processed_frame, color_mask)

    return processed_frame


def main():
    cap = cv2.VideoCapture(0)
    qr_detector = cv2.QRCodeDetector()

    while cap.isOpened():
        ret, input_frame = cap.read()
        if not ret:
            break
    
        processed_frame = process_frame(input_frame, qr_detector)

        cv2.imshow('Real-time Detection', processed_frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

main()
