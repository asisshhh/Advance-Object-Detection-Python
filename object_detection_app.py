import cv2
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
from PIL import Image, ImageTk
import threading
import time
import os
import numpy as np
from pymongo import MongoClient
from datetime import datetime
import json
import torch
from ultralytics import YOLO
import torchvision.models as models
import torchvision.transforms as transforms
from collections import defaultdict
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from torchvision.models.detection import fasterrcnn_resnet50_fpn, FasterRCNN_ResNet50_FPN_Weights

class ObjectDetectionApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Advanced Object Detection System")
        self.root.geometry("1400x900")
        self.root.configure(bg="#1e1e2e")

        # Initialize variables
        self.cap = None
        self.video_running = False
        self.recording = False
        self.out = None
        self.frame_size = (640, 480)
        self.current_model = None
        self.models = {}
        self.object_counts = defaultdict(int)
        self.tracked_objects = {}
        self.alerts = []
        self.camera_list = []
        self.selected_camera = 0
        self.user_settings = self.load_settings()

        # Create necessary directories
        os.makedirs("recordings", exist_ok=True)
        os.makedirs("exports", exist_ok=True)
        os.makedirs("models", exist_ok=True)

        # Initialize database
        self.init_database()

        self.setup_ui()
        self.load_models()
        self.scan_cameras()

    def init_database(self):
        try:
            # Connect to MongoDB
            self.client = MongoClient('mongodb://localhost:27017/')
            self.db = self.client['object_detection_db']

            # Create collections if they don't exist
            self.detections = self.db['detections']
            self.alerts_collection = self.db['alerts']
            self.logs_collection = self.db['logs']

            # Create indexes for better query performance
            self.detections.create_index([('timestamp', 1)])
            self.alerts_collection.create_index([('timestamp', 1)])
            self.logs_collection.create_index([('timestamp', 1)])

        except Exception as e:
            messagebox.showerror("Database Error", f"Failed to connect to MongoDB: {str(e)}")
            raise

    def load_settings(self):
        try:
            with open('settings.json', 'r') as f:
                return json.load(f)
        except:
            default_settings = {
                'confidence_threshold': 0.5,
                'tracking_enabled': True,
                'alert_threshold': 3,
                'export_format': 'csv',
                'auto_save': True
            }
            with open('settings.json', 'w') as f:
                json.dump(default_settings, f)
            return default_settings

    def setup_ui(self):
        # Main frame
        main_frame = tk.Frame(self.root, bg="#1e1e2e")
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        # Left panel for video display
        self.video_frame = tk.Frame(main_frame, bg="#1e1e2e")
        self.video_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        self.label = tk.Label(self.video_frame, bg="#1e1e2e")
        self.label.pack(pady=10)

        # Right panel for controls and statistics
        control_frame = tk.Frame(main_frame, bg="#1e1e2e", width=300)
        control_frame.pack(side=tk.RIGHT, fill=tk.Y, padx=10)

        # Model selection
        model_frame = tk.LabelFrame(control_frame, text="Model Selection", bg="#1e1e2e", fg="white")
        model_frame.pack(fill=tk.X, pady=5)
        self.model_var = tk.StringVar()
        model_combo = ttk.Combobox(model_frame, textvariable=self.model_var)
        model_combo['values'] = ('YOLOv8', 'SSD MobileNet', 'Faster R-CNN')
        model_combo.pack(fill=tk.X, padx=5, pady=5)
        model_combo.bind('<<ComboboxSelected>>', self.on_model_change)

        # Camera selection
        camera_frame = tk.LabelFrame(control_frame, text="Camera Selection", bg="#1e1e2e", fg="white")
        camera_frame.pack(fill=tk.X, pady=5)
        self.camera_var = tk.StringVar()
        self.camera_combo = ttk.Combobox(camera_frame, textvariable=self.camera_var)
        self.camera_combo.pack(fill=tk.X, padx=5, pady=5)
        self.camera_combo.bind('<<ComboboxSelected>>', self.on_camera_change)

        # Control buttons
        buttons = [
            ("Detect Image", self.load_image, "#6c5ce7"),
            ("Start Webcam", self.start_webcam, "#00b894"),
            ("Start Recording", self.start_recording, "#e17055"),
            ("Upload Video", self.load_video, "#0984e3"),
            ("Stop", self.stop_video, "#d63031"),
            ("Export Report", self.export_report, "#00cec9"),
            ("Settings", self.show_settings, "#6c5ce7")
        ]

        for text, command, color in buttons:
            btn = tk.Button(control_frame, text=text, command=command,
                          bg=color, fg="white", font=("Helvetica", 12), width=20)
            btn.pack(pady=5)

        # Statistics display
        stats_frame = tk.LabelFrame(control_frame, text="Statistics", bg="#1e1e2e", fg="white")
        stats_frame.pack(fill=tk.X, pady=5)
        self.stats_text = tk.Text(stats_frame, height=10, width=30, bg="#2d3436", fg="white")
        self.stats_text.pack(padx=5, pady=5)

        # Alert display
        alert_frame = tk.LabelFrame(control_frame, text="Alerts", bg="#1e1e2e", fg="white")
        alert_frame.pack(fill=tk.X, pady=5)
        self.alert_text = tk.Text(alert_frame, height=5, width=30, bg="#2d3436", fg="white")
        self.alert_text.pack(padx=5, pady=5)

    def load_models(self):
        try:
            # Load YOLOv8
            if not os.path.exists('yolov8n.pt'):
                messagebox.showinfo("Info", "Downloading YOLOv8 model...")
                YOLO('yolov8n.pt').download()

            self.models['YOLOv8'] = YOLO('yolov8n.pt')

            # Load SSD MobileNet
            pb_file = os.path.join('models', 'frozen_inference_graph.pb')
            pbtxt_file = os.path.join('models', 'ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt')

            if not os.path.exists(pb_file) or not os.path.exists(pbtxt_file):
                messagebox.showerror("Error", "SSD MobileNet model files not found. Please download them manually.")
            else:
                self.models['SSD MobileNet'] = cv2.dnn_DetectionModel(pb_file, pbtxt_file)
                self.models['SSD MobileNet'].setInputSize(320, 320)
                self.models['SSD MobileNet'].setInputScale(1.0 / 127.5)
                self.models['SSD MobileNet'].setInputMean((127.5, 127.5, 127.5))
                self.models['SSD MobileNet'].setInputSwapRB(True)

            # Load Faster R-CNN
            try:
                weights = FasterRCNN_ResNet50_FPN_Weights.COCO_V1
                self.models['Faster R-CNN'] = fasterrcnn_resnet50_fpn(weights=weights)
                self.models['Faster R-CNN'].eval()
            except Exception as e:
                messagebox.showwarning("Warning", f"Could not load Faster R-CNN: {str(e)}")

            # Load class labels
            labels_file = os.path.join('models', 'labels.txt')
            if not os.path.exists(labels_file):
                messagebox.showerror("Error", "Class labels file not found. Please create models/labels.txt")
            else:
                with open(labels_file, 'rt') as f:
                    self.classLabels = f.read().rstrip('\n').split('\n')

            # Add COCO class labels for Faster R-CNN
            self.coco_labels = [
                '__background__', 'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus',
                'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'N/A', 'stop sign',
                'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
                'elephant', 'bear', 'zebra', 'giraffe', 'N/A', 'backpack', 'umbrella', 'N/A', 'N/A',
                'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
                'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket',
                'bottle', 'N/A', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl',
                'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza',
                'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed', 'N/A', 'dining table',
                'N/A', 'N/A', 'toilet', 'N/A', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone',
                'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'N/A', 'book',
                'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush'
            ]

            self.model_var.set('YOLOv8')
            self.current_model = self.models['YOLOv8']
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load models: {str(e)}")
            raise

    def scan_cameras(self):
        self.camera_list = []
        for i in range(10):  # Check first 10 camera indices
            cap = cv2.VideoCapture(i)
            if cap.isOpened():
                self.camera_list.append(f"Camera {i}")
                cap.release()
        
        if self.camera_list:
            self.camera_combo['values'] = self.camera_list
            self.camera_combo.set(self.camera_list[0])

    def on_model_change(self, event):
        model_name = self.model_var.get()
        if model_name in self.models:
            self.current_model = self.models[model_name]
            messagebox.showinfo("Info", f"Switched to {model_name} model")

    def on_camera_change(self, event):
        camera_index = self.camera_list.index(self.camera_var.get())
        if self.video_running:
            self.stop_video()
            self.selected_camera = camera_index
            self.start_webcam()
        else:
            self.selected_camera = camera_index

    def detect_objects(self, frame):
        try:
            if isinstance(self.current_model, YOLO):
                results = self.current_model(frame)
                for result in results:
                    boxes = result.boxes
                    for box in boxes:
                        x1, y1, x2, y2 = box.xyxy[0]
                        conf = box.conf[0]
                        cls = int(box.cls[0])
                        if cls < len(self.classLabels):
                            label = f"{self.classLabels[cls]}: {conf:.2f}"
                            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (255, 0, 0), 2)
                            cv2.putText(frame, label, (int(x1), int(y1)-10),
                                      cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
                            self.update_statistics(self.classLabels[cls], float(conf))
                            
                            # Save detection to MongoDB
                            self.logs_collection.insert_one({
                                'timestamp': datetime.now(),
                                'object_type': self.classLabels[cls],
                                'confidence': float(conf),
                                'location': f"({int(x1)}, {int(y1)})",
                                'image_path': None  
                            })
            elif isinstance(self.current_model, cv2.dnn_DetectionModel):
                classIndex, confidence, bbox = self.current_model.detect(frame, confThreshold=0.5)
                if len(classIndex) != 0:
                    for ClassInd, conf, boxes in zip(classIndex.flatten(), confidence.flatten(), bbox):
                        if ClassInd < len(self.classLabels):
                            label = f"{self.classLabels[ClassInd-1]}: {conf:.2f}"
                            cv2.rectangle(frame, boxes, (255, 0, 0), 2)
                            cv2.putText(frame, label, (boxes[0], boxes[1]-10),
                                      cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
                            self.update_statistics(self.classLabels[ClassInd-1], float(conf))
                            
                            # Save detection to MongoDB
                            self.logs_collection.insert_one({
                                'timestamp': datetime.now(),
                                'object_type': self.classLabels[ClassInd-1],
                                'confidence': float(conf),
                                'location': f"({boxes[0]}, {boxes[1]})",
                                'image_path': None
                            })
            else:  # Faster R-CNN
                # Convert frame to tensor
                transform = transforms.Compose([transforms.ToTensor()])
                frame_tensor = transform(frame)
                frame_tensor = frame_tensor.unsqueeze(0)
                
                # Run detection
                with torch.no_grad():
                    predictions = self.current_model(frame_tensor)
                
                # Process predictions
                boxes = predictions[0]['boxes'].cpu().numpy()
                scores = predictions[0]['scores'].cpu().numpy()
                labels = predictions[0]['labels'].cpu().numpy()
                
                # Filter predictions with confidence > 0.5
                threshold = 0.5
                valid_indices = scores > threshold
                boxes = boxes[valid_indices]
                scores = scores[valid_indices]
                labels = labels[valid_indices]
                
                # Draw detections
                for box, score, label in zip(boxes, scores, labels):
                    x1, y1, x2, y2 = box.astype(int)
                    label_name = self.coco_labels[label]
                    confidence = score

                    # Draw rectangle
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)

                    # Add label and confidence
                    text = f"{label_name}: {confidence:.2f}"
                    cv2.putText(frame, text, (x1, y1 - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

                    # Update statistics and save to MongoDB
                    self.update_statistics(label_name, float(confidence))
                    self.logs_collection.insert_one({
                        'timestamp': datetime.now(),
                        'object_type': label_name,
                        'confidence': float(confidence),
                        'location': f"({x1}, {y1})",
                        'image_path': None
                    })

            return frame
        except Exception as e:
            print(f"Detection error: {str(e)}")
            return frame

    def update_statistics(self, object_type, confidence):
        self.object_counts[object_type] += 1
        self.stats_text.delete(1.0, tk.END)
        for obj, count in self.object_counts.items():
            self.stats_text.insert(tk.END, f"{obj}: {count}\n")

        # Check for alerts
        if self.object_counts[object_type] >= self.user_settings['alert_threshold']:
            self.add_alert(object_type, f"Multiple {object_type} detected!")

    def add_alert(self, object_type, message):
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        self.alerts.append((timestamp, object_type, message))
        self.alert_text.delete(1.0, tk.END)
        for alert in self.alerts[-5:]:  # Show last 5 alerts
            self.alert_text.insert(tk.END, f"{alert[0]} - {alert[1]}: {alert[2]}\n")

        # Save to database
        self.alerts_collection.insert_one({
            'timestamp': timestamp,
            'object_type': object_type,
            'message': message,
            'severity': 'high'
        })

    def export_report(self):
        try:
            # Generate statistics report
            df = pd.DataFrame(list(self.object_counts.items()), columns=['Object', 'Count'])

            # Create visualization
            fig, ax = plt.subplots(figsize=(8, 6))
            df.plot(kind='bar', x='Object', y='Count', ax=ax)
            plt.title('Object Detection Statistics')
            plt.xticks(rotation=45)
            plt.tight_layout()

            # Save report
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            report_path = f"exports/report_{timestamp}.csv"
            fig_path = f"exports/stats_{timestamp}.png"

            df.to_csv(report_path, index=False)
            plt.savefig(fig_path)
            plt.close()

            messagebox.showinfo("Success", f"Report exported to {report_path}")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to export report: {str(e)}")

    def show_settings(self):
        settings_window = tk.Toplevel(self.root)
        settings_window.title("Settings")
        settings_window.geometry("400x300")

        # Confidence threshold
        tk.Label(settings_window, text="Confidence Threshold:").pack()
        conf_scale = tk.Scale(settings_window, from_=0.1, to=1.0, resolution=0.1,
                              orient=tk.HORIZONTAL)
        conf_scale.set(self.user_settings['confidence_threshold'])
        conf_scale.pack()

        # Alert threshold
        tk.Label(settings_window, text="Alert Threshold:").pack()
        alert_scale = tk.Scale(settings_window, from_=1, to=10, orient=tk.HORIZONTAL)
        alert_scale.set(self.user_settings['alert_threshold'])
        alert_scale.pack()

        # Tracking enabled
        tracking_var = tk.BooleanVar(value=self.user_settings['tracking_enabled'])
        tk.Checkbutton(settings_window, text="Enable Object Tracking",
                      variable=tracking_var).pack()

        def save_settings():
            self.user_settings.update({
                'confidence_threshold': conf_scale.get(),
                'alert_threshold': alert_scale.get(),
                'tracking_enabled': tracking_var.get()
            })
            with open('settings.json', 'w') as f:
                json.dump(self.user_settings, f)
            settings_window.destroy()

        tk.Button(settings_window, text="Save", command=save_settings).pack(pady=10)

    def show_frame(self, frame):
        try:
            frame = cv2.resize(frame, self.frame_size)
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(rgb)
            imgtk = ImageTk.PhotoImage(image=img)
            self.label.configure(image=imgtk)
            self.label.image = imgtk
        except Exception as e:
            print(f"Display error: {str(e)}")

    def load_image(self):
        try:
            path = filedialog.askopenfilename(filetypes=[("Image Files", "*.jpg *.png *.jpeg")])
            if path:
                img = cv2.imread(path)
                if img is not None:
                    img = self.detect_objects(img)
                    self.show_frame(img)
                else:
                    messagebox.showerror("Error", "Failed to load image")
        except Exception as e:
            messagebox.showerror("Error", str(e))

    def start_webcam(self):
        if not self.video_running:
            try:
                if self.cap is not None:
                    self.cap.release()
                self.cap = cv2.VideoCapture(0)
                if not self.cap.isOpened():
                    messagebox.showerror("Error", "Could not access webcam")
                    return
                self.video_running = True
                self.recording = False
                threading.Thread(target=self.webcam_loop, daemon=True).start()
            except Exception as e:
                messagebox.showerror("Error", str(e))

    def webcam_loop(self):
        try:
            while self.video_running and self.cap is not None:
                ret, frame = self.cap.read()
                if not ret:
                    break
                frame = self.detect_objects(frame)
                if self.recording and self.out is not None:
                    self.out.write(frame)
                self.show_frame(frame)
                time.sleep(0.016)  # ~60 FPS
        except Exception as e:
            print(f"Webcam error: {str(e)}")
        finally:
            self.stop_video()

    def start_recording(self):
        if self.cap is not None and self.video_running and not self.recording:
            try:
                filename = os.path.join("recordings", f"recording_{int(time.time())}.avi")
                fourcc = cv2.VideoWriter_fourcc(*'XVID')
                fps = self.cap.get(cv2.CAP_PROP_FPS)
                fps = 30.0 if fps <= 0 else fps
                self.out = cv2.VideoWriter(filename, fourcc, fps, self.frame_size)
                if not self.out.isOpened():
                    raise Exception("Failed to create video writer")
                self.recording = True
                messagebox.showinfo("Info", f"Recording started: {filename}")
            except Exception as e:
                self.recording = False
                if self.out is not None:
                    self.out.release()
                    self.out = None
                messagebox.showerror("Error", f"Failed to start recording: {str(e)}")

    def load_video(self):
        try:
            path = filedialog.askopenfilename(filetypes=[("Video Files", "*.mp4 *.avi *.mov")])
            if path:
                self.stop_video()
                self.cap = cv2.VideoCapture(path)
                if not self.cap.isOpened():
                    messagebox.showerror("Error", "Could not open video file")
                    return
                self.video_running = True
                threading.Thread(target=self.video_loop, daemon=True).start()
        except Exception as e:
            messagebox.showerror("Error", f"Error loading video: {str(e)}")

    def video_loop(self):
        try:
            while self.video_running:
                ret, frame = self.cap.read()
                if not ret:
                    self.video_running = False
                    break
                frame = self.detect_objects(frame)
                self.show_frame(frame)
                time.sleep(0.033)  # ~30 FPS for smoother playback
        except Exception as e:
            print(f"Video playback error: {str(e)}")
        finally:
            self.stop_video()

    def stop_video(self):
        self.video_running = False
        self.recording = False
        
        if self.out is not None:
            self.out.release()
            self.out = None
            
        if self.cap is not None:
            self.cap.release()
            self.cap = None
            
        # Clear the display
        blank = np.zeros((self.frame_size[1], self.frame_size[0], 3), dtype=np.uint8)
        self.show_frame(blank)

    def cleanup_and_exit(self):
        try:
            self.stop_video()
            if hasattr(self, 'client'):
                self.client.close()
            self.root.quit()
        except Exception as e:
            print(f"Error during cleanup: {str(e)}")
            self.root.quit()

if __name__ == "__main__":
    root = tk.Tk()
    app = ObjectDetectionApp(root)
    root.protocol("WM_DELETE_WINDOW", app.cleanup_and_exit)
    root.mainloop()
