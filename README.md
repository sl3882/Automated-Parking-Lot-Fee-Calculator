```markdown
# 🚗 Automated Parking Lot Fee Calculator (ALPR)

This project is an automated Parking Management System that uses Computer Vision to detect vehicles, read their license plates (OCR), and calculate parking fees based on the duration of their stay.

It uses **YOLOv3** for vehicle detection and **EasyOCR** for reading license plate text.

## 📂 Project Structure

```text
Parking_Lot_Project/
│
├── main.py              # The main script (Entry/Exit logic + GUI)
├── util.py              # Helper functions for YOLO processing
├── parking_data.json    # Database (Stores entry times automatically)
├── README.md            # This file
│
├── data/                # Test images (cars entering/exiting)
│   ├── 1.png
│   └── ...
│
└── model/               # AI Model configuration
    ├── coco.names       # List of objects YOLO detects
    ├── cfg/
    │   └── yolov3.cfg
    └── weights/
        └── (Place yolov3.weights here!) <--- IMPORTANT

```

## ⚠️ Prerequisite: Download the AI Model

Because the AI model file is too large for GitHub (230MB+), it is not included in this repository. **The code will crash if you skip this step.**

1. **Download the `yolov3.weights` file** from my Google Drive:
* 👉 https://drive.google.com/file/d/1cRPcyNOrlr6BkoFfIerEg7Undi_Dg83j/view?usp=sharing


2. Go to the folder `model/weights/` inside this project.
3. Paste the `yolov3.weights` file there.




## 🧠 How it Works

1. **Detection:** The YOLOv3 model scans the image to locate a vehicle (Car, Truck, or Bus).
2. **Cropping:** The system crops the vehicle from the main image.
3. **OCR (Optical Character Recognition):** EasyOCR analyzes the cropped vehicle to find alphanumeric text (the license plate).
4. **Logic:**
* **Entry:** Checks if the car is new. If yes, saves the timestamp.
* **Exit:** specific cars are triggered to leave. The system calculates `(Exit Time - Entry Time)` and applies the fee rate ($10 base + $5 per extra 30 mins).



---



```

```
