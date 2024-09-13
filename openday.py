import cv2
import numpy as np
from ultralytics import YOLO
import time

# Cargar el modelo YOLO
model = YOLO("yolov8n-seg.pt")

# Diccionario para mantener el tiempo activo del QR
active_qr = {"color": "", "timestamp": 0, "person_index": -1}

def detect_qr_and_color(frame, qr_detector):
    data, points, _ = qr_detector.detectAndDecode(frame)
    
    if data in ["RED", "GREEN", "BLUE"]:
        return data, points
    return "", []

def find_closest_person_to_qr(person_bboxes, qr_points):
    if len(person_bboxes) == 0 or len(qr_points) == 0:
        return -1
    
    qr_center = np.mean(qr_points, axis=0)  # Centro aproximado del QR
    min_dist = float('inf')
    closest_index = -1

    for i, (x1, y1, x2, y2) in enumerate(person_bboxes):
        person_center = [(x1 + x2) / 2, (y1 + y2) / 2]
        dist = np.linalg.norm(np.array(person_center) - np.array(qr_center))
        if dist < min_dist:
            min_dist = dist
            closest_index = i

    return closest_index

def process_frame(input_frame, qr_detector):
    global active_qr
    results_yolo = model.predict(source=input_frame)
    person_bboxes = []
    person_masks = []
    
    for result in results_yolo:
        for r in result:
            label = r.names[int(r.boxes.cls.tolist().pop())]
            bbox = r.boxes.xyxy[0].cpu().numpy().astype(int)
            
            if label == 'person':
                person_bboxes.append(bbox)
                person_masks.append(r.masks.xy.pop().astype(np.int32).reshape(-1, 1, 2))

    input_frame_gray = cv2.cvtColor(input_frame, cv2.COLOR_BGR2GRAY)
    qr_data, qr_points = detect_qr_and_color(input_frame_gray, qr_detector)
    
    # Actualizar la detección del QR
    current_time = time.time()
    if qr_data != "":
        closest_person_index = find_closest_person_to_qr(person_bboxes, qr_points)
        active_qr = {"color": qr_data, "timestamp": current_time, "person_index": closest_person_index}

    # Verificar si el tiempo del QR ha expirado (5 segundos)
    time_left = 5 - (current_time - active_qr["timestamp"])
    if time_left <= 0:
        active_qr = {"color": "", "timestamp": 0, "person_index": -1}
    
    # Procesar el fondo
    processed_frame = input_frame.copy()
    if active_qr["color"] != "":  # Sin QR detectado o tiempo expirado
        processed_frame = cv2.cvtColor(input_frame, cv2.COLOR_BGR2GRAY)
        processed_frame = cv2.cvtColor(processed_frame, cv2.COLOR_GRAY2BGR)

    # Dibujar las personas y aplicar el color solo a la persona con el QR
    for i, (bbox, mask) in enumerate(zip(person_bboxes, person_masks)):
        x1, y1, x2, y2 = bbox
        
        # Solo colorear a la persona más cercana al QR
        if active_qr["color"] != "" and i == active_qr["person_index"]:
            # Dibujar la silueta de la persona con el color del QR detectado
            color_map = {"RED": (0, 0, 255), "GREEN": (0, 255, 0), "BLUE": (255, 0, 0)}
            qr_color = color_map.get(active_qr["color"], (255, 0, 255))

            b_mask = np.zeros(input_frame.shape[:2], np.uint8)
            cv2.drawContours(b_mask, [mask], -1, (255, 255, 255), cv2.FILLED)
            color_mask = np.zeros_like(input_frame)
            color_mask[b_mask == 255] = qr_color

            processed_frame = cv2.bitwise_and(processed_frame, processed_frame, mask=cv2.bitwise_not(b_mask))
            processed_frame = cv2.add(processed_frame, color_mask)
        
        # Dibujar rectángulo y etiqueta de la persona
        cv2.rectangle(processed_frame, (x1, y1), (x2, y2), (128, 0, 128), 2)
        cv2.putText(processed_frame, "Futuro Cachimbo UTEC", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (128, 0, 128), 2, cv2.LINE_AA)

    # Mostrar el tiempo restante
    if active_qr["color"] != "":
        cv2.putText(processed_frame, f"Tiempo restante: {int(time_left)}", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 3, cv2.LINE_AA)

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

