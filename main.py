import os
import cv2
import easyocr
import numpy as np
import json
import util
from datetime import datetime, timedelta

# --- UPDATED Configuration ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Note the change to 'yolov3.cfg' and 'coco.names'
MODEL_CFG = os.path.join(BASE_DIR, 'model', 'cfg', 'yolov3.cfg')
MODEL_WEIGHTS = os.path.join(BASE_DIR, 'model', 'weights', 'yolov3.weights')
CLASS_NAMES = os.path.join(BASE_DIR, 'model', 'coco.names')
DB_FILE = os.path.join(BASE_DIR, "parking_data.json")
BACKUP_FOLDER = os.path.join(BASE_DIR, 'temp')

os.makedirs(BACKUP_FOLDER, exist_ok=True)

class LicensePlateDetector:
    def __init__(self):
        print("Loading Standard YOLOv3 Model...")
        # 1. Load Class Names
        with open(CLASS_NAMES, 'r') as f:
            self.class_names = [j.strip() for j in f.readlines()]

        # 2. Load YOLO Model
        self.net = cv2.dnn.readNetFromDarknet(MODEL_CFG, MODEL_WEIGHTS)
        self.net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
        self.net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)

        # 3. Load OCR Reader
        self.reader = easyocr.Reader(['en'], gpu=False)

    def detect_and_read(self, image_path):
        if not os.path.exists(image_path):
            print(f"Error: Image {image_path} not found.")
            return None

        img = cv2.imread(image_path)
        H, W, _ = img.shape

        blob = cv2.dnn.blobFromImage(img, 1 / 255, (416, 416), (0, 0, 0), True, crop=False)
        self.net.setInput(blob)
        detections = util.get_outputs(self.net)

        bboxes, class_ids, scores = [], [], []

        for detection in detections:
            bbox = detection[:4]
            xc, yc, w, h = bbox
            bbox = [int(xc * W), int(yc * H), int(w * W), int(h * H)]

            # GET THE CLASS ID
            class_id = np.argmax(detection[5:])
            score = np.amax(detection[5:])

            # CHECK IF IT IS A CAR (Class ID 2 in COCO dataset)
            # We also accept 'truck' (7) or 'bus' (5) just in case
            if class_id in [2, 5, 7]:
                bboxes.append(bbox)
                class_ids.append(class_id)
                scores.append(score)

        bboxes, class_ids, scores = util.NMS(bboxes, class_ids, scores)

        detected_plate_text = None

        for bbox in bboxes:
            xc, yc, w, h = bbox
            x1 = max(0, int(xc - (w / 2)))
            y1 = max(0, int(yc - (h / 2)))
            x2 = min(W, int(xc + (w / 2)))
            y2 = min(H, int(yc + (h / 2)))

            # Crop the WHOLE CAR
            car_roi = img[y1:y2, x1:x2, :].copy()

            # Attempt OCR on the whole car (might be messy, but proves concept)
            gray = cv2.cvtColor(car_roi, cv2.COLOR_BGR2GRAY)
            output = self.reader.readtext(gray)

            for result in output:
                _, text, confidence = result
                # Clean text to alphanumeric only
                text_clean = ''.join(e for e in text if e.isalnum())

                # Loose filter: look for something that looks like a plate (4-8 chars)
                if confidence > 0.3 and 4 < len(text_clean) < 9:
                    detected_plate_text = text_clean.upper()

                    # Save debug image
                    cv2.imwrite(os.path.join(BACKUP_FOLDER, f"{detected_plate_text}.jpg"), car_roi)
                    break

            if detected_plate_text:
                break

        return detected_plate_text

class ParkingSystem:
    def __init__(self):
        self.detector = LicensePlateDetector()
        self.db = self.load_db()

    def load_db(self):
        """Loads parking data from JSON file."""
        if os.path.exists(DB_FILE):
            try:
                with open(DB_FILE, 'r') as f:
                    data = json.load(f)
                    # Convert string timestamps back to datetime objects
                    for plate, info in data.items():
                        if isinstance(info, str):  # Handle legacy format
                            data[plate] = datetime.fromisoformat(info)
                        elif 'entry_time' in info:
                            data[plate]['entry_time'] = datetime.fromisoformat(info['entry_time'])
                    return data
            except Exception as e:
                print(f"Error loading DB: {e}")
                return {}
        return {}

    def save_db(self):
        """Saves parking data to JSON file."""
        # Convert datetime objects to strings for JSON serialization
        serializable_data = {}
        for plate, info in self.db.items():
            if isinstance(info, datetime):
                serializable_data[plate] = info.isoformat()
            elif isinstance(info, dict) and 'entry_time' in info:
                # If you store more data later, handle it here
                temp_info = info.copy()
                temp_info['entry_time'] = info['entry_time'].isoformat()
                serializable_data[plate] = temp_info

        with open(DB_FILE, 'w') as f:
            json.dump(serializable_data, f, indent=4)

    def entry_vehicle(self, image_path):
        plate_num = self.detector.detect_and_read(image_path)

        if not plate_num:
            print(f"FAILED: Could not read license plate from {image_path}")
            return

        if plate_num in self.db:
            print(f"ALERT: Vehicle {plate_num} is already in the parking lot.")
        else:
            entry_time = datetime.now()
            # Store just the time, or a dict if you want more info later
            self.db[plate_num] = entry_time
            self.save_db()
            print(f"SUCCESS: Vehicle {plate_num} entered at {entry_time.strftime('%H:%M:%S')}")

    def exit_vehicle(self, image_path, base_rate=10.0, additional_rate=5.0):
        plate_num = self.detector.detect_and_read(image_path)

        if not plate_num:
            print(f"FAILED: Could not read license plate from {image_path}")
            return

        if plate_num in self.db:
            entry_time = self.db[plate_num]

            # SIMULATION: Add 75 minutes to current time to simulate parking duration
            exit_time = datetime.now() + timedelta(minutes=75)

            # Calculate Duration
            duration = exit_time - entry_time
            minutes_spent = duration.total_seconds() / 60

            # Calculate Fee
            parking_fee = base_rate
            if minutes_spent > 30:
                additional_periods = (minutes_spent - 30) // 30
                parking_fee += additional_periods * additional_rate

            print(f"--- EXIT RECEIPT ---")
            print(f"Vehicle: {plate_num}")
            print(f"Entry: {entry_time.strftime('%Y-%m-%d %H:%M:%S')}")
            print(f"Exit:  {exit_time.strftime('%Y-%m-%d %H:%M:%S')}")
            print(f"Duration: {minutes_spent:.2f} min")
            print(f"Total Fee: ${parking_fee:.2f}")
            print(f"--------------------")

            # Checkout complete: Remove from DB
            del self.db[plate_num]
            self.save_db()

            # Optional: Clean up temp image
            # temp_img = os.path.join(BACKUP_FOLDER, f"{plate_num}.jpg")
            # if os.path.exists(temp_img): os.remove(temp_img)

        else:
            print(f"ERROR: Vehicle {plate_num} not found in system. Gate will not open.")

# --- Main Execution Block ---
if __name__ == '__main__':
    parking_lot = ParkingSystem()
    img_folder = os.path.join(BASE_DIR, 'data')

    print("\n====== PARKING SIMULATION STARTED ======")

    # 1. Simulate Entries (Cars 1 to 10 entering)
    print("\n--- 🚗 VEHICLES ENTERING ---")
    for i in range(1, 11):  # Loop from 1 to 10
        img_name = f"{i}.png"  # <--- CHANGED TO .png
        img_path = os.path.join(img_folder, img_name)

        # Check if file exists before trying
        if os.path.exists(img_path):
            parking_lot.entry_vehicle(img_path)
        else:
            print(f"Warning: {img_name} missing at {img_path}")

    # 2. Simulate Exits (Cars 1, 3, and 5 leaving)
    print("\n--- 🏁 VEHICLES EXITING ---")

    # Let's say cars 1, 3, and 5 decide to leave
    cars_leaving = [1, 3, 5]

    for i in cars_leaving:
        img_name = f"{i}.png"  # <--- CHANGED TO .png
        img_path = os.path.join(img_folder, img_name)

        if os.path.exists(img_path):
            parking_lot.exit_vehicle(img_path)
        else:
            print(f"Warning: {img_name} missing for exit.")

    print("\n====== SIMULATION COMPLETE ======")