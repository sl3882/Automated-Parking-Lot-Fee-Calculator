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
