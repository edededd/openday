import cv2
import numpy as np
from ultralytics import YOLO
import time
import random

# Cargar el modelo YOLO
model = YOLO("yolov8n-seg.pt")

class ActiveUser:    
    lastColor = 0
    def __init__(self, id):
        global lastColor
        self.id = id
        self.start_time = time.time()
        self.color = ActiveUser.lastColor
        ActiveUser.lastColor = (ActiveUser.lastColor + 1) % 6
        self.is_active = True

    def get_remaining_time(self, current_time, timeout=10):
        elapsed_time = current_time - self.start_time
        remaining_time = max(0, timeout - elapsed_time)
        if remaining_time == 0:
            self.is_active = False
        return remaining_time

# Diccionario para mantener los usuarios activos
active_users = {}

def detect_phones(results):
    phones = []
    for r in results[0].boxes:
        if int(r.cls.item()) == 67:  # 67 es el ID de clase para 'cell phone' en COCO dataset
            phones.append(r.xyxy[0].cpu().numpy().astype(int))
    return phones

def find_closest_person_to_phone(person_data, phone_bbox):
    if len(person_data) == 0:
        return None, None, None
    
    phone_center = [(phone_bbox[0] + phone_bbox[2]) / 2, (phone_bbox[1] + phone_bbox[3]) / 2]
    min_dist = float('inf')
    closest_bbox = None
    closest_id = None
    closest_mask = None

    for bbox, person_id, mask in person_data:
        x1, y1, x2, y2 = bbox
        person_center = [(x1 + x2) / 2, (y1 + y2) / 2]
        dist = np.linalg.norm(np.array(person_center) - np.array(phone_center))
        if dist < min_dist:
            min_dist = dist
            closest_bbox = bbox
            closest_id = person_id
            closest_mask = mask

    return closest_bbox, closest_id, closest_mask

def process_frame(input_frame):
    global active_users
    silhouette_gray = None
    input_frame = cv2.cvtColor(input_frame, cv2.COLOR_BGR2GRAY)
    input_frame = cv2.cvtColor(input_frame, cv2.COLOR_GRAY2BGR)
    results = model.track(source=input_frame, persist=True, verbose=False,conf=0.55)
    
    person_data = []
    for i, r in enumerate(results[0].boxes):
        if int(r.cls.item()) == 0:  # 0 es el ID de clase para 'person' en COCO dataset
            bbox = r.xyxy[0].cpu().numpy().astype(int)
            person_id = int(r.id.item()) if r.id is not None else i  # Usar índice como ID de respaldo
            mask = results[0].masks.data[i].cpu().numpy() if results[0].masks is not None else None
            person_data.append((bbox, person_id, mask))


    phone_bboxes = detect_phones(results)
    
    current_time = time.time()
    processed_frame = input_frame.copy()

    # Actualizar usuarios activos
    for phone_bbox in phone_bboxes:
        closest_person_bbox, closest_person_id, closest_person_mask = find_closest_person_to_phone(person_data, phone_bbox)
        if closest_person_id is not None and closest_person_id not in active_users:
            active_users[closest_person_id] = ActiveUser(closest_person_id)

    # Procesar cada persona detectada
    for bbox, person_id, mask in person_data:
        if person_id in active_users and active_users[person_id].is_active:
            user = active_users[person_id]
            remaining_time = user.get_remaining_time(current_time)
            
            if mask is not None:
                # Redimensionar la máscara al tamaño del frame
                mask = cv2.resize(mask, (input_frame.shape[1], input_frame.shape[0]))
                mask = (mask > 0.5).astype(np.uint8) * 255
                
                # Aplicar la máscara de color
                silhouette_gray_bgr = cv2.bitwise_and(input_frame, input_frame, mask=mask)
                A, _, _ = cv2.split(silhouette_gray_bgr);
                Z = np.ones(A.shape, dtype=np.uint8)
                
                colored_silhouette = silhouette_gray_bgr
                if user.color % 6 == 0:
                    colored_silhouette = cv2.merge([Z, Z, A]) 
                elif user.color % 6 == 1:
                    colored_silhouette = cv2.merge([Z, A, Z]) 
                elif user.color % 6 == 2:
                    colored_silhouette = cv2.merge([A, Z, Z]) 
                elif user.color % 6 == 3:
                    colored_silhouette = cv2.merge([A, A, Z]) 
                elif user.color % 6 == 4:
                    colored_silhouette = cv2.merge([A, Z, A]) 
                elif user.color % 6 == 5:
                    colored_silhouette = cv2.merge([Z, A, A]) 
              
                # Combinar la silueta coloreada con el frame procesado
                mask_inv = cv2.bitwise_not(mask)
                processed_frame = cv2.add(processed_frame, colored_silhouette)
                
                # Dibujar el contorno de la silueta
                contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                cv2.drawContours(processed_frame, contours, -1, (255, 255, 255), 2)
            
            x1, y1, x2, y2 = bbox
            cv2.putText(processed_frame, f"Futuro Cachimbo UTEC", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
            cv2.putText(processed_frame, f"Tiempo: {remaining_time:.1f}s", (x1, y2 + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
        else:
            x1, y1, x2, y2 = bbox
            cv2.rectangle(processed_frame, (x1, y1), (x2, y2), (128, 0, 128), 2)
            cv2.putText(processed_frame, f"Futuro Cachimbo UTEC", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (128, 0, 128), 2)

    # Limpiar usuarios inactivos
    active_users = {id: user for id, user in active_users.items() if user.is_active}

    return processed_frame, silhouette_gray

def main():
    cap = cv2.VideoCapture(0)

    while cap.isOpened():
        ret, input_frame = cap.read()
        if not ret:
            break
    
        processed_frame, temp = process_frame(input_frame)
        if temp is not None:
            cv2.imshow('temp', temp)
        cv2.imshow('Real-time Detection', processed_frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
