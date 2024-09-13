import cv2
import numpy as np
from ultralytics import YOLO
import time

# Cargar el modelo YOLO
model = YOLO("yolov8n-seg.pt")

# Diccionario para mantener el tiempo activo del QR y el ID de la persona
active_qr = {"color": "", "timestamp": 0, "person_id": None}

def detect_qr_and_color(frame, qr_detector):
    data, points, _ = qr_detector.detectAndDecode(frame)
    
    if data in ["RED", "GREEN", "BLUE"]:
        return data, points
    return "", []

def find_closest_person_to_qr(person_data, qr_points):
    if len(person_data) == 0 or len(qr_points) == 0:
        return None, None, None
    
    qr_center = np.mean(qr_points, axis=0)
    min_dist = float('inf')
    closest_bbox = None
    closest_id = None
    closest_mask = None

    for bbox, person_id, mask in person_data:
        x1, y1, x2, y2 = bbox
        person_center = [(x1 + x2) / 2, (y1 + y2) / 2]
        dist = np.linalg.norm(np.array(person_center) - np.array(qr_center))
        if dist < min_dist:
            min_dist = dist
            closest_bbox = bbox
            closest_id = person_id
            closest_mask = mask

    return closest_bbox, closest_id, closest_mask

def process_frame(input_frame, qr_detector):
    global active_qr

    results = model.track(source=input_frame, persist=True, verbose=False)
    
    person_data = []
    for i, r in enumerate(results[0].boxes):
        if int(r.cls.item()) == 0:  # 0 is the class ID for person in COCO dataset
            bbox = r.xyxy[0].cpu().numpy().astype(int)
            person_id = int(r.id.item()) if r.id is not None else i  # Use index as fallback ID
            mask = results[0].masks.data[i].cpu().numpy() if results[0].masks is not None else None
            person_data.append((bbox, person_id, mask))

    input_frame_gray = cv2.cvtColor(input_frame, cv2.COLOR_BGR2GRAY)
    qr_data, qr_points = detect_qr_and_color(input_frame_gray, qr_detector)
    
    current_time = time.time()
    processed_frame = input_frame.copy()

    if qr_data != "":
        closest_person_bbox, closest_person_id, closest_person_mask = find_closest_person_to_qr(person_data, qr_points)
        if closest_person_bbox is not None:
            active_qr = {
                "color": qr_data, 
                "timestamp": current_time, 
                "person_id": closest_person_id
            }

    # Verificar si el tiempo del QR ha expirado (5 segundos)
    time_left = 5 - (current_time - active_qr["timestamp"])
    if time_left <= 0:
        active_qr = {"color": "", "timestamp": 0, "person_id": None}

    if active_qr["color"] != "":
        # Convertir toda la imagen a escala de grises
        gray_frame = cv2.cvtColor(input_frame, cv2.COLOR_BGR2GRAY)
        processed_frame = cv2.cvtColor(gray_frame, cv2.COLOR_GRAY2BGR)
        
        # Aplicar el color solo a la silueta de la persona detectada con QR activo
        for bbox, person_id, mask in person_data:
            if person_id == active_qr["person_id"]:
                color_map = {"RED": (0, 0, 255), "GREEN": (0, 255, 0), "BLUE": (255, 0, 0)}
                qr_color = color_map.get(active_qr["color"], (255, 0, 255))
                
                if mask is not None:
                    # Redimensionar la máscara al tamaño del frame
                    mask = cv2.resize(mask, (input_frame.shape[1], input_frame.shape[0]))
                    mask = (mask > 0.5).astype(np.uint8) * 255
                    
                    # Crear una imagen coloreada para la silueta
                    colored_silhouette = np.zeros_like(processed_frame)
                    colored_silhouette[mask > 0] = qr_color
                    
                    # Combinar la silueta coloreada con el frame procesado
                    processed_frame = cv2.addWeighted(processed_frame, 1, colored_silhouette, 0.5, 0)
                    
                    # Dibujar el contorno de la silueta
                    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                    cv2.drawContours(processed_frame, contours, -1, qr_color, 2)
                
                x1, y1, x2, y2 = bbox
                cv2.putText(processed_frame, f"Futuro Cachimbo UTEC (ID: {active_qr['person_id']})", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, qr_color, 2)
    else:
        # Dibujar bounding boxes y etiquetas para todas las personas detectadas
        for bbox, person_id, _ in person_data:
            x1, y1, x2, y2 = bbox
            cv2.rectangle(processed_frame, (x1, y1), (x2, y2), (128, 0, 128), 2)
            cv2.putText(processed_frame, f"Persona (ID: {person_id})", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (128, 0, 128), 2)

    if active_qr["color"] != "":
        cv2.putText(processed_frame, f"Tiempo restante: {int(time_left)}", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 3)

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

if __name__ == "__main__":
    main()
