#!/usr/bin/env python3
import os
import time
import platform
import json
import cv2
import numpy as np
import tensorflow as tf
from PIL import Image, ImageTk
import customtkinter as ctk
from tkinter import filedialog, messagebox
import paho.mqtt.client as mqtt
import firebase_admin
from firebase_admin import credentials, storage

# ==========================================================
# CONFIGURATION
# ==========================================================
MQTT_BROKER = "10.222.99.189"
MQTT_PORT = 1883
MQTT_TOPIC = "arc/Crop-monitoring/group5/Crop-sensor"

FIREBASE_BUCKET = "cropvisionsystem.firebasestorage.app"
FIREBASE_KEYFILE = "firebase-key.json"

# ==========================================================
# FIREBASE INITIALIZATION
# ==========================================================
bucket = None
try:
    if not firebase_admin._apps:
        cred = credentials.Certificate(FIREBASE_KEYFILE)
        firebase_admin.initialize_app(cred, {"storageBucket": FIREBASE_BUCKET})
        print("[LOG] ✅ Firebase initialized successfully.")
    bucket = storage.bucket()
except Exception as e:
    print(f"[ERROR] Firebase initialization failed: {e}")
    bucket = None


def upload_image_to_firebase(local_path, remote_name):
    if bucket is None:
        print("[ERROR] Firebase bucket not initialized.")
        return None
    try:
        blob = bucket.blob(f"crops/{remote_name}")
        blob.upload_from_filename(local_path)
        blob.make_public()
        print(f"[LOG] ✅ Uploaded to Firebase: {blob.public_url}")
        return blob.public_url
    except Exception as e:
        print(f"[ERROR] Firebase upload failed: {e}")
        return None


# ==========================================================
# MODEL CONFIGURATION
# ==========================================================
MODEL_PATHS = {
    "Apple Diseases": r"/home/ifs325-group5/trained models/Apples_disease_20251013_204650.keras",
    "Grape Diseases": r"/home/ifs325-group5/trained models/Grapes_disease_20251014_112936.keras",
    "Tomato Fruit Diseases": r"/home/ifs325-group5/trained models/TomatoFruitDiseases_disease_model.keras",
    "Tomato Leaf Diseases": r"/home/ifs325-group5/trained models/TomatoLeafDiseases_disease_model.keras",
    "Pest Detection": r"/home/ifs325-group5/trained models/pest_mobilenetv2.keras",
    "Lettuce Growth Stage": r"/home/ifs325-group5/trained models/growth_stage_disease_20251020_192429.h5",
    "Basil Growth Stage": r"/home/ifs325-group5/trained models/basil_growth_stage_cnn.h5",
}

MODEL_INPUT_SIZES = {
    "Apple Diseases": (224, 224),
    "Grape Diseases": (224, 224),
    "Tomato Fruit Diseases": (224, 224),
    "Tomato Leaf Diseases": (224, 224),
    "Pest Detection": (224, 224),
    "Lettuce Growth Stage": (112, 112),
    "Basil Growth Stage": (112, 112),
}

MODEL_COLOR_MODES = {
    "Apple Diseases": "rgb",
    "Grape Diseases": "rgb",
    "Tomato Fruit Diseases": "rgb",
    "Tomato Leaf Diseases": "rgb",
    "Pest Detection": "rgb",
    "Lettuce Growth Stage": "rgb",
    "Basil Growth Stage": "grayscale", 
}

CATEGORY_MODELS = {
    "Diseases": [
        "Apple Diseases", "Grape Diseases",
        "Tomato Fruit Diseases", "Tomato Leaf Diseases"
    ],
    "Pests": ["Pest Detection"],
    "Growth Stages": [
        "Lettuce Growth Stage",
        "Basil Growth Stage"
    ]
}

CLASS_NAMES = {
    "Apple Diseases": [
        "Black Rot Leaves", "Blotch Apple", "Blotch Leaves",
        "Cedar Apple Rust Leaves", "Healthy Apple", "Healthy Leaves",
        "Rotten Apple", "Scab Apple", "Scab Leaves"
    ],
    "Grape Diseases": [
        "Black Rot Leaf", "Blight Leaf", "Esca Leaf",
        "Healthy Fruit", "Healthy Leaf", "Insect Hole Leaf", "Unhealthy Fruit"
    ],
    "Tomato Fruit Diseases": ["Healthy", "Viral", "Wilt", "Graymold"],
    "Tomato Leaf Diseases": [
        "Bacterial Spot", "Early Blight", "Late Blight", "Healthy",
        "Leaf Mold", "Septoria Leaf Spot", "Spider Mites",
        "Target Spot", "Yellow Leaf Curl Virus", "Mosaic Virus"
    ],
    "Pest Detection": [
        "stem-borer", "rice-bug", "leaf-folder", "green-leafhopper", "whorl-maggot", "brown-planthopper"
    ],
    "Lettuce Growth Stage": ["Lettuce-young seedling", "Ready to harvest lettuce"],
    "Basil Growth Stage": ["baby_basil","mature_basil"],
}

loaded_models = {}


def load_model(name):
    if name in loaded_models:
        return loaded_models[name]
    path = MODEL_PATHS.get(name)
    if not path or not os.path.exists(path):
        raise FileNotFoundError(f"[ERROR] Model not found: {path}")
    print(f"[LOG] 🧠 Loading model: {name} from {path}")
    try:
        model = tf.keras.models.load_model(path)
        loaded_models[name] = model
        print(f"[LOG] ✅ Model loaded successfully: {name}")
        print(f"[LOG] 📊 Model input shape: {model.input_shape}")
        return model
    except Exception as e:
        print(f"[ERROR] Failed to load model '{name}': {e}")
        raise


def _get_class_list_for(model_name):
    if model_name in CLASS_NAMES:
        return CLASS_NAMES[model_name]
    low = model_name.lower()
    for key in CLASS_NAMES:
        if key.lower() == low:
            return CLASS_NAMES[key]
    print(f"[WARN] No class names found for model '{model_name}', returning ['Unknown']")
    return ["Unknown"]


def preprocess_image(frame, target_size=(224, 224), color_mode="rgb", expected_channels=None):
    if frame is None:
        raise ValueError("[ERROR] No frame to preprocess.")

    if len(frame.shape) == 3 and frame.shape[2] == 4:
        frame = frame[:, :, :3]

    if color_mode == "grayscale":
        if len(frame.shape) == 3:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        img = cv2.resize(frame, target_size)
        img = img.astype("float32") / 255.0
        img = np.expand_dims(img, axis=-1)
    else:
        if len(frame.shape) == 2:
            frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
        img = cv2.resize(frame, target_size)
        img = img.astype("float32") / 255.0

    if expected_channels is not None:
        if img.shape[-1] != expected_channels:
            if expected_channels == 3 and img.shape[-1] == 1:
                img = np.repeat(img, 3, axis=-1)
            elif expected_channels == 1 and img.shape[-1] == 3:
                img = cv2.cvtColor((img * 255).astype("uint8"), cv2.COLOR_RGB2GRAY)
                img = img.astype("float32") / 255.0
                img = np.expand_dims(img, axis=-1)

    return np.expand_dims(img, axis=0)


def predict_disease(model, frame, model_name):
    try:
        target_size = MODEL_INPUT_SIZES.get(model_name, (224, 224))
        color_mode = MODEL_COLOR_MODES.get(model_name, "rgb")
        model_input_shape = model.input_shape
        expected_channels = model_input_shape[-1] if len(model_input_shape) == 4 else 3

        print(f"[LOG] Using input size {target_size}, color mode '{color_mode}', expected_channels={expected_channels}")

        processed = preprocess_image(frame, target_size, color_mode, expected_channels)
        print(f"[LOG] Preprocessed shape: {processed.shape}")

        preds = model.predict(processed, verbose=0)
        if preds.ndim == 1:
            probs = preds
        else:
            probs = preds[0]

        idx = int(np.argmax(probs))
        conf = float(np.max(probs))

        labels = _get_class_list_for(model_name)
        if idx >= len(labels):
            print(f"[WARN] Prediction index {idx} out of bounds for {model_name}, defaulting to first class.")
            idx = 0
        label = labels[idx]

        print(f"[DEBUG] Model: {model_name}, Predicted index: {idx}, Label: {label}, Confidence: {conf:.3f}")

        return label, conf

    except Exception as e:
        print(f"[ERROR] Prediction failed: {e}")
        import traceback
        traceback.print_exc()
        return "Error", 0.0


def get_recommendation(label, model_name=None):
    label_lower = label.lower()
    if model_name and model_name in CATEGORY_MODELS["Growth Stages"]:
        if "seedling" in label_lower:
            return "Young seedling detected. Maintain soil moisture and protect from pests."
        if "harvest" in label_lower:
            return "Ready to harvest. Harvest early morning for best quality."
        if "vegetative" in label_lower:
            return "Vegetative stage. Ensure adequate nutrients and water."
        if "flowering" in label_lower:
            return "Flowering stage. Avoid disturbing plants and maintain consistent watering."
        return "Monitor crop growth and adjust nutrients as needed."
    if model_name and model_name in CATEGORY_MODELS["Pests"]:
        if "aphid" in label_lower:
            return "Aphids detected. Spray neem oil or insecticidal soap."
        if "whitefly" in label_lower:
            return "Whiteflies detected. Use yellow sticky traps."
        if "spider mite" in label_lower:
            return "Spider mites detected. Increase humidity and apply miticide."
        if "thrip" in label_lower:
            return "Thrips detected. Remove infested leaves and use blue traps."
        return "No pests detected. Maintain good field hygiene."
    if "healthy" in label_lower:
        return "Healthy crop. Maintain routine monitoring."
    if "blight" in label_lower:
        return "Blight detected. Remove infected leaves and apply copper-based spray."
    if "scab" in label_lower:
        return "Scab detected. Apply captan-based fungicide."
    if "rot" in label_lower:
        return "Rot detected. Improve ventilation and prune affected parts."
    if "rust" in label_lower:
        return "Rust detected. Apply sulfur-based fungicide."
    if "viral" in label_lower or "mosaic" in label_lower:
        return "Viral symptoms. Isolate infected plants and disinfect tools."
    return "Monitor crop closely for further symptoms."


def send_to_mqtt(payload):
    try:
        print(f"[LOG] 🔌 Connecting to MQTT broker at {MQTT_BROKER}:{MQTT_PORT}...")
        client = mqtt.Client(protocol=mqtt.MQTTv311)
        client.connect(MQTT_BROKER, MQTT_PORT, 60)
        print(f"[LOG] ✅ Connected to MQTT broker successfully!")
        client.loop_start()
        
        print(f"[LOG] 📤 Publishing to topic: {MQTT_TOPIC}")
        result = client.publish(MQTT_TOPIC, json.dumps(payload), retain=True)
        print(f"[LOG] ✅ Message published with result code: {result.rc}")
        
        time.sleep(1)
        client.loop_stop()
        client.disconnect()
        print(f"[LOG] ✅ Published to MQTT successfully!")
        print(f"[LOG] 📊 Payload: {json.dumps(payload, indent=2)}")
        return True
    except Exception as e:
        print(f"[ERROR] ❌ MQTT publish failed: {e}")
        import traceback
        traceback.print_exc()
        return False


# ==========================================================
# MAIN APPLICATION - REDESIGNED UI
# ==========================================================
class CropDetectionApp(ctk.CTk):
    def __init__(self):
        super().__init__()
        self.title("🌿 CropVision Detection System")
        self.geometry("1000x800")
        ctk.set_appearance_mode("dark")
        ctk.set_default_color_theme("green")

        # Vars
        self.category_var = ctk.StringVar(value="Diseases")
        self.model_name = ctk.StringVar(value=CATEGORY_MODELS["Diseases"][0])
        self.model = None
        self.cap = None
        self.picam2 = None
        self.running = False
        self.frame_count = 0
        self.label_text = ""
        self.conf = 0.0
        self.current_frame = None
        self.camera_index = None

        self.captures_folder = "captures"
        os.makedirs(self.captures_folder, exist_ok=True)

        # UI
        self.create_widgets()
        self.protocol("WM_DELETE_WINDOW", self.exit_app)

    def create_widgets(self):
        # Header
        header = ctk.CTkFrame(self, fg_color=("#2d5016", "#1a3010"), height=80)
        header.pack(fill="x", padx=0, pady=0)
        header.pack_propagate(False)
        
        ctk.CTkLabel(
            header, 
            text="🌿 CropVision Detection System", 
            font=("Segoe UI", 26, "bold"),
            text_color="white"
        ).pack(pady=20)

        # Main container
        main_container = ctk.CTkFrame(self, fg_color="transparent")
        main_container.pack(fill="both", expand=True, padx=20, pady=10)

        # Left panel - Controls
        left_panel = ctk.CTkFrame(main_container, width=280)
        left_panel.pack(side="left", fill="y", padx=(0, 10))
        left_panel.pack_propagate(False)

        # Model Selection Section
        selection_frame = ctk.CTkFrame(left_panel)
        selection_frame.pack(fill="x", padx=15, pady=(15, 10))

        ctk.CTkLabel(
            selection_frame, 
            text="Model Configuration", 
            font=("Segoe UI", 15, "bold")
        ).pack(pady=(10, 5))

        ctk.CTkLabel(
            selection_frame, 
            text="Category:", 
            font=("Segoe UI", 12),
            anchor="w"
        ).pack(fill="x", padx=10, pady=(10, 2))

        self.category_menu = ctk.CTkOptionMenu(
            selection_frame,
            variable=self.category_var,
            values=list(CATEGORY_MODELS.keys()),
            command=self.on_category_change,
            height=35,
            font=("Segoe UI", 12)
        )
        self.category_menu.pack(fill="x", padx=10, pady=(0, 10))

        ctk.CTkLabel(
            selection_frame, 
            text="Detection Model:", 
            font=("Segoe UI", 12),
            anchor="w"
        ).pack(fill="x", padx=10, pady=(5, 2))

        self.model_menu = ctk.CTkOptionMenu(
            selection_frame,
            variable=self.model_name,
            values=CATEGORY_MODELS[self.category_var.get()],
            height=35,
            font=("Segoe UI", 12)
        )
        self.model_menu.pack(fill="x", padx=10, pady=(0, 15))

        # Control Buttons
        controls_frame = ctk.CTkFrame(left_panel)
        controls_frame.pack(fill="both", expand=True, padx=15, pady=10)

        ctk.CTkLabel(
            controls_frame, 
            text="Controls", 
            font=("Segoe UI", 15, "bold")
        ).pack(pady=(10, 15))

        btn_style = {
            "height": 45,
            "font": ("Segoe UI", 13),
            "corner_radius": 8
        }

        self.start_btn = ctk.CTkButton(
            controls_frame,
            text="▶ Start Live Detection",
            command=self.start_live_detection,
            fg_color="#28a745",
            hover_color="#218838",
            **btn_style
        )
        self.start_btn.pack(fill="x", padx=10, pady=5)

        self.stop_btn = ctk.CTkButton(
            controls_frame,
            text="⏹ Stop Detection",
            command=self.stop_live_detection,
            fg_color="#dc3545",
            hover_color="#c82333",
            **btn_style
        )
        self.stop_btn.pack(fill="x", padx=10, pady=5)

        ctk.CTkButton(
            controls_frame,
            text="📁 Upload Image",
            command=self.upload_image,
            fg_color="#007bff",
            hover_color="#0056b3",
            **btn_style
        ).pack(fill="x", padx=10, pady=5)

        # Spacer
        ctk.CTkFrame(controls_frame, height=20, fg_color="transparent").pack()

        ctk.CTkButton(
            controls_frame,
            text="🚪 Exit Application",
            command=self.exit_app,
            fg_color="#6c757d",
            hover_color="#5a6268",
            **btn_style
        ).pack(fill="x", padx=10, pady=5, side="bottom")

        # Right panel - Video Feed
        right_panel = ctk.CTkFrame(main_container)
        right_panel.pack(side="right", fill="both", expand=True)

        # Video section
        video_section = ctk.CTkFrame(right_panel)
        video_section.pack(fill="both", expand=True, padx=15, pady=15)

        ctk.CTkLabel(
            video_section, 
            text="Live Detection Feed", 
            font=("Segoe UI", 16, "bold")
        ).pack(pady=(10, 5))

        # Video frame container
        self.video_container = ctk.CTkFrame(video_section, fg_color="#000000")
        self.video_container.pack(fill="both", expand=True, padx=10, pady=10)

        self.video_label = ctk.CTkLabel(
            self.video_container, 
            text="No video feed\n\nClick 'Start Live Detection' to begin",
            font=("Segoe UI", 14),
            text_color="#888888"
        )
        self.video_label.pack(fill="both", expand=True, padx=20, pady=20)

        # Capture button directly below video frame
        self.capture_btn = ctk.CTkButton(
            video_section,
            text="📷 Capture Screenshot",
            command=self.capture_screenshot,
            fg_color="#ff9800",
            hover_color="#e68900",
            height=50,
            font=("Segoe UI", 14, "bold"),
            corner_radius=10
        )
        self.capture_btn.pack(fill="x", padx=10, pady=(5, 10))

        # Status bar at bottom
        status_frame = ctk.CTkFrame(right_panel, height=60)
        status_frame.pack(fill="x", padx=15, pady=(0, 15))
        status_frame.pack_propagate(False)

        ctk.CTkLabel(
            status_frame, 
            text="Status:", 
            font=("Segoe UI", 12, "bold")
        ).pack(side="left", padx=15, pady=15)

        self.status = ctk.CTkLabel(
            status_frame,
            text="Ready to start detection",
            font=("Segoe UI", 13),
            text_color="#28a745"
        )
        self.status.pack(side="left", padx=5, pady=15)

    def on_category_change(self, _value=None):
        cat = self.category_var.get()
        models = CATEGORY_MODELS.get(cat, [])
        self.model_menu.configure(values=models)
        if models:
            self.model_name.set(models[0])

    def find_working_camera(self):
        if self.camera_index is not None:
            print(f"[LOG] Trying previously working camera index {self.camera_index}...")
            cap = cv2.VideoCapture(self.camera_index)
            if cap.isOpened():
                ret, _ = cap.read()
                if ret:
                    print(f"[LOG] ✅ Reusing camera at index {self.camera_index}")
                    return cap
                cap.release()
        
        for index in range(5):
            print(f"[LOG] Trying camera index {index}...")
            cap = cv2.VideoCapture(index)
            if cap.isOpened():
                ret, _ = cap.read()
                if ret:
                    print(f"[LOG] ✅ Found working camera at index {index}")
                    self.camera_index = index
                    return cap
                cap.release()
        
        return None

    def start_live_detection(self):
        if self.running:
            print("[WARN] Detection already running.")
            return
        
        self.stop_live_detection()
        time.sleep(0.3)
        
        try:
            self.model = load_model(self.model_name.get())
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load model: {e}")
            return

        self.running = True
        print("[LOG] Starting live detection...")
        self.status.configure(text="Starting camera...", text_color="#ffc107")

        if platform.system() == "Linux":
            try:
                from picamera2 import Picamera2
                self.picam2 = Picamera2()
                config = self.picam2.create_preview_configuration(main={"size": (640, 480), "format": "RGB888"})
                self.picam2.configure(config)
                self.picam2.start()
                print("[LOG] ✅ Picamera2 started with RGB888 format.")
                self.status.configure(text="Live detection active", text_color="#28a745")
                self.update_frame_pi()
                return
            except Exception as e:
                print(f"[WARN] Picamera2 start failed: {e} – falling back to OpenCV")

        self.cap = self.find_working_camera()
        
        if self.cap is None:
            error_msg = (
                "No camera found!\n\n"
                "Troubleshooting:\n"
                "1. Check if camera is connected\n"
                "2. Check camera permissions\n"
                "3. Close other apps using the camera\n"
                "4. Try unplugging and replugging camera"
            )
            messagebox.showerror("Camera Error", error_msg)
            print("[ERROR] No working camera found.")
            self.running = False
            self.status.configure(text="Camera not found", text_color="#dc3545")
            return

        print("[LOG] ✅ Camera started (OpenCV).")
        self.status.configure(text="Live detection active", text_color="#28a745")
        self.update_frame()

    def update_frame(self):
        if not self.running or not self.cap:
            return
        ret, frame = self.cap.read()
        if not ret:
            print("[ERROR] Frame read failed.")
            self.after(30, self.update_frame)
            return
        
        self._process_and_display_frame(frame)
        self.after(30, self.update_frame)

    def update_frame_pi(self):
        if not self.running or not self.picam2:
            return
        
        try:
            frame = self.picam2.capture_array()
            self._process_and_display_frame(frame)
        except Exception as e:
            print(f"[ERROR] Picamera2 capture failed: {e}")
        
        self.after(30, self.update_frame_pi)

    def _process_and_display_frame(self, frame):
        if frame is None:
            return
        
        self.current_frame = frame.copy()

        if self.frame_count % 5 == 0:
            self.label_text, self.conf = predict_disease(self.model, frame, self.model_name.get())
            status_color = "#28a745" if "healthy" in str(self.label_text).lower() else "#dc3545"
            self.status.configure(
                text=f"{self.label_text} ({self.conf*100:.1f}%)",
                text_color=status_color
            )

        self.frame_count += 1
        
        display_frame = frame.copy()
        overlay = f"{self.label_text} ({self.conf*100:.1f}%)"
        color = (0, 255, 0) if "healthy" in str(self.label_text).lower() else (0, 0, 255)
        cv2.putText(display_frame, overlay, (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)

        try:
            display_rgb = cv2.cvtColor(display_frame, cv2.COLOR_BGR2RGB)
            img = ImageTk.PhotoImage(Image.fromarray(display_rgb))
            self.video_label.img = img
            self.video_label.configure(image=img, text="")
        except Exception as e:
            print(f"[ERROR] Frame display failed: {e}")

    def capture_screenshot(self):
        if not self.running or self.current_frame is None:
            messagebox.showwarning("Warning", "No live feed to capture! Start live detection first.")
            return
        
        print("[LOG] 📷 Capturing screenshot from live feed...")
        
        try:
            timestamp = int(time.time())
            filename = f"capture_{timestamp}.jpg"
            filepath = os.path.join(self.captures_folder, filename)
            
            cv2.imwrite(filepath, self.current_frame)
            print(f"[LOG] ✅ Screenshot saved to: {filepath}")
            
            label_text, conf = predict_disease(self.model, self.current_frame, self.model_name.get())
            recommendation = get_recommendation(label_text, self.model_name.get())
            
            print(f"[LOG] 📊 Prediction: {label_text} ({conf*100:.1f}%)")
            print(f"[LOG] 💡 Recommendation: {recommendation}")
            
            firebase_filename = f"capture_{timestamp}.jpg"
            print(f"[LOG] ☁️ Uploading to Firebase as: {firebase_filename}")
            image_url = upload_image_to_firebase(filepath, firebase_filename)
            
            payload = {
                "Group_ID":"5",
                "Crop_ID": f"capture_{timestamp}",
                "Model_Name": self.model_name.get(),
                "Label": label_text,
                "Confidence": f"{conf*100:.2f}%",
                "Recommendation": recommendation,
                "Image_URL": image_url,
            }
            
            print(f"[LOG] 📡 Sending to MQTT broker...")
            mqtt_success = send_to_mqtt(payload)
            
            if mqtt_success:
                messagebox.showinfo(
                    "Capture Success", 
                    f"Screenshot captured and processed!\n\n"
                    f"Prediction: {label_text}\n"
                    f"Confidence: {conf*100:.1f}%\n\n"
                    f"Recommendation:\n{recommendation}\n\n"
                    f"Saved to: {filepath}\n"
                    f"Uploaded to Firebase ✅\n"
                    f"Sent to MQTT ✅"
                )
            else:
                messagebox.showwarning(
                    "Partial Success",
                    f"Screenshot captured and processed!\n\n"
                    f"Prediction: {label_text}\n"
                    f"Confidence: {conf*100:.1f}%\n\n"
                    f"Recommendation:\n{recommendation}\n\n"
                    f"⚠️ MQTT publish failed - check broker connection"
                )
            
        except Exception as e:
            print(f"[ERROR] Screenshot capture failed: {e}")
            import traceback
            traceback.print_exc()
            messagebox.showerror("Error", f"Failed to capture screenshot: {str(e)}")

    def upload_image(self):
        file_path = filedialog.askopenfilename(
            title="Select Image",
            filetypes=[("Image files", "*.jpg *.jpeg *.png *.bmp"), ("All files", "*.*")]
        )
        if not file_path:
            return
        
        if not os.path.exists(file_path):
            messagebox.showerror("Error", f"File not found: {file_path}")
            return
        
        print(f"[LOG] Loading image from: {file_path}")
        
        try:
            self.model = load_model(self.model_name.get())
            
            frame = cv2.imread(file_path)
            if frame is None:
                raise ValueError("Failed to read image. File may be corrupted or unsupported format.")
            
            print(f"[LOG] Image loaded: {frame.shape}")
            
            label_text, conf = predict_disease(self.model, frame, self.model_name.get())
            recommendation = get_recommendation(label_text, self.model_name.get())

            firebase_filename = f"{os.path.splitext(os.path.basename(file_path))[0]}_{int(time.time())}.jpg"
            image_url = upload_image_to_firebase(file_path, firebase_filename)

            payload = {
                "Group_ID":"5",
                "Crop_ID":os.path.splitext(os.path.basename(file_path))[0],
                "Model_Name": self.model_name.get(),
                "Label": label_text,
                "Confidence": f"{conf*100:.2f}%",
                "Recommendation": recommendation,
                "Image_URL": image_url,
            }
            
            print(f"[LOG] 📡 Sending to MQTT broker...")
            mqtt_success = send_to_mqtt(payload)

            height, width = frame.shape[:2]
            max_display_size = 600
            if width > max_display_size or height > max_display_size:
                scale = max_display_size / max(width, height)
                new_width = int(width * scale)
                new_height = int(height * scale)
                frame = cv2.resize(frame, (new_width, new_height))
            
            overlay = f"{label_text} ({conf*100:.1f}%)"
            color = (0, 255, 0) if "healthy" in label_text.lower() else (0, 0, 255)
            cv2.putText(frame, overlay, (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)

            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img = ImageTk.PhotoImage(Image.fromarray(frame_rgb))
            self.video_label.img = img
            self.video_label.configure(image=img, text="")
            
            status_color = "#28a745" if "healthy" in label_text.lower() else "#dc3545"
            self.status.configure(text=f"{label_text} ({conf*100:.1f}%)", text_color=status_color)

            print(f"[LOG] ✅ Upload complete: Label={label_text} Conf={conf:.3f}")
            
            if mqtt_success:
                messagebox.showinfo(
                    "Success", 
                    f"Prediction: {label_text}\nConfidence: {conf*100:.1f}%\n\n"
                    f"{recommendation}\n\n"
                    f"Uploaded to Firebase ✅\nSent to MQTT ✅"
                )
            else:
                messagebox.showwarning(
                    "Partial Success", 
                    f"Prediction: {label_text}\nConfidence: {conf*100:.1f}%\n\n"
                    f"{recommendation}\n\n"
                    f"Uploaded to Firebase ✅\n⚠️ MQTT publish failed"
                )

        except Exception as e:
            print(f"[ERROR] Upload image failed: {e}")
            import traceback
            traceback.print_exc()
            messagebox.showerror("Error", f"Upload failed: {str(e)}")

    def stop_live_detection(self):
        print("[LOG] Stopping detection...")
        self.running = False
        
        if self.cap:
            try:
                self.cap.release()
                print("[LOG] ✅ OpenCV camera released.")
            except Exception as e:
                print(f"[WARN] Error releasing camera: {e}")
            finally:
                self.cap = None
        
        if self.picam2:
            try:
                self.picam2.stop()
                self.picam2.close()
                print("[LOG] ✅ Picamera2 stopped and closed.")
            except Exception as e:
                print(f"[WARN] Error stopping Picamera2: {e}")
            finally:
                self.picam2 = None
        
        time.sleep(0.2)
        
        self.video_label.configure(
            image="",
            text="Detection stopped\n\nClick 'Start Live Detection' to begin again"
        )
        self.status.configure(text="Detection stopped", text_color="#6c757d")
        print("[LOG] 🛑 Detection stopped.")

    def exit_app(self):
        self.stop_live_detection()
        print("[LOG] Exiting app...")
        self.destroy()


# ==========================================================
# ENTRY POINT
# ==========================================================
if __name__ == "__main__":
    print("[LOG] 🚀 Starting CropVision (Redesigned UI)...")
    app = CropDetectionApp()
    app.mainloop()
