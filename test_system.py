import cv2
import torch
from ultralytics import YOLO
import torchvision.models as models
from pymongo import MongoClient
import os
import sys
import numpy as np
from PIL import Image
import tkinter as tk
from tkinter import messagebox
from torchvision.models.detection import fasterrcnn_resnet50_fpn, FasterRCNN_ResNet50_FPN_Weights

def check_models():
    print("\n=== Checking Model Loading ===")
    models_status = True
    
    try:
        # Check YOLOv8
        print("\n1. Checking YOLOv8...")
        if not os.path.exists('yolov8n.pt'):
            print("YOLOv8 model not found. Downloading...")
            YOLO('yolov8n.pt').download()
        yolo_model = YOLO('yolov8n.pt')
        print("✓ YOLOv8 loaded successfully")
    except Exception as e:
        print(f"✗ YOLOv8 failed: {str(e)}")
        models_status = False

    try:
        # Check SSD MobileNet
        print("\n2. Checking SSD MobileNet...")
        pb_file = os.path.join('models', 'frozen_inference_graph.pb')
        pbtxt_file = os.path.join('models', 'ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt')
        
        if not os.path.exists(pb_file) or not os.path.exists(pbtxt_file):
            print("✗ SSD MobileNet model files not found")
            models_status = False
        else:
            ssd_model = cv2.dnn_DetectionModel(pb_file, pbtxt_file)
            ssd_model.setInputSize(320, 320)
            ssd_model.setInputScale(1.0 / 127.5)
            ssd_model.setInputMean((127.5, 127.5, 127.5))
            ssd_model.setInputSwapRB(True)
            print("✓ SSD MobileNet loaded successfully")
    except Exception as e:
        print(f"✗ SSD MobileNet failed: {str(e)}")
        models_status = False

    try:
        # Check Faster R-CNN
        print("\n3. Checking Faster R-CNN...")
        weights = FasterRCNN_ResNet50_FPN_Weights.COCO_V1
        faster_rcnn = fasterrcnn_resnet50_fpn(weights=weights)
        faster_rcnn.eval()

        print("✓ Faster R-CNN loaded successfully")
    except Exception as e:
        print(f"✗ Faster R-CNN failed: {str(e)}")
        models_status = False

    try:
        # Check class labels
        print("\n4. Checking class labels...")
        labels_file = os.path.join('models', 'labels.txt')
        if not os.path.exists(labels_file):
            print("✗ Class labels file not found")
            models_status = False
        else:
            with open(labels_file, 'rt') as f:
                labels = f.read().rstrip('\n').split('\n')
            print(f"✓ Class labels loaded successfully ({len(labels)} classes)")
    except Exception as e:
        print(f"✗ Class labels failed: {str(e)}")
        models_status = False

    return models_status

def check_mongodb():
    print("\n=== Checking MongoDB Connection ===")
    try:
        client = MongoClient('mongodb://localhost:27017/')
        server_info = client.server_info()
        print(f"✓ MongoDB Server Version: {server_info['version']}")
        
        # Test database operations
        db = client['object_detection_db']
        test_collection = db['test_collection']
        
        # Test insert
        test_collection.insert_one({'test': 'connection'})
        print("✓ MongoDB write test successful")
        
        # Test read
        result = test_collection.find_one({'test': 'connection'})
        print("✓ MongoDB read test successful")
        
        # Clean up
        test_collection.drop()
        client.close()
        return True
    except Exception as e:
        print(f"✗ MongoDB connection failed: {str(e)}")
        return False

def check_camera():
    print("\n=== Checking Camera Access ===")
    try:
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("✗ No camera found")
            return False
        
        ret, frame = cap.read()
        if not ret:
            print("✗ Could not read from camera")
            return False
        
        print("✓ Camera access successful")
        cap.release()
        return True
    except Exception as e:
        print(f"✗ Camera check failed: {str(e)}")
        return False

def check_directories():
    print("\n=== Checking Required Directories ===")
    directories = ['models', 'recordings', 'exports']
    all_ok = True
    
    for directory in directories:
        if not os.path.exists(directory):
            print(f"✗ Directory '{directory}' not found")
            all_ok = False
        else:
            print(f"✓ Directory '{directory}' exists")
    
    return all_ok

def main():
    print("Starting comprehensive system check...")
    
    # Check all components
    directories_ok = check_directories()
    models_ok = check_models()
    mongodb_ok = check_mongodb()
    camera_ok = check_camera()
    
    print("\n=== Summary ===")
    print(f"Directories Status: {'✓ OK' if directories_ok else '✗ Failed'}")
    print(f"Models Status: {'✓ OK' if models_ok else '✗ Failed'}")
    print(f"MongoDB Status: {'✓ OK' if mongodb_ok else '✗ Failed'}")
    print(f"Camera Status: {'✓ OK' if camera_ok else '✗ Failed'}")
    
    if all([directories_ok, models_ok, mongodb_ok, camera_ok]):
        print("\nAll components are working correctly!")
        return True
    else:
        print("\nSome components failed to initialize. Please check the error messages above.")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 