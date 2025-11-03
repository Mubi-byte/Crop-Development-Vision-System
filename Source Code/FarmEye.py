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
    
    # ==================== GROWTH STAGES ====================
    if model_name and model_name in CATEGORY_MODELS["Growth Stages"]:
        if "baby" in label_lower or "young" in label_lower or "seedling" in label_lower:
            return "Young seedling stage. Maintain consistent moisture, provide 14-16 hours of light daily, and apply diluted fertilizer (1/4 strength) weekly. Monitor for aphids and prevent overwatering to avoid damping off."
        
        if "mature" in label_lower or "harvest" in label_lower or "ready" in label_lower:
            return "Ready to harvest! Harvest in early morning when plants are crisp. Cut 2-3cm above soil for potential regrowth. Wash immediately and store in refrigerator at 4-7°C. Use within 5-7 days for best quality."
        
        if "vegetative" in label_lower:
            return "Vegetative growth stage. Increase watering frequency, apply balanced NPK fertilizer (10-10-10) bi-weekly, ensure 12-14 hours light daily. Thin overcrowded plants for better air circulation."
        
        if "flowering" in label_lower:
            return "Flowering stage detected. Switch to bloom fertilizer (lower nitrogen, higher phosphorus). Maintain consistent watering, stable temperature, and avoid disturbing plants. Hand pollinate if growing indoors."
        
        return "Monitor crop growth and adjust nutrients as needed."
    
    # ==================== PEST DETECTION ====================
    if model_name and model_name in CATEGORY_MODELS["Pests"]:
        if "stem" in label_lower and "borer" in label_lower:
            return "Stem borer detected - urgent action needed. Remove and destroy affected stems. Apply Chlorantraniliprole or use Bacillus thuringiensis (Bt) spray. Release Trichogramma wasps for biological control. Avoid excessive nitrogen fertilizer."
        
        if "leaf" in label_lower and "folder" in label_lower:
            return "Leaf folder detected. Hand-pick folded leaves with larvae. Spray with Fipronil 5% SC (2ml/liter) or neem oil (5ml/liter) every 5 days. Use light traps to catch adult moths. Maintain balanced fertilization."
        
        if "planthopper" in label_lower or "hopper" in label_lower:
            return "Planthopper infestation - serious threat! Apply Imidacloprid or Thiamethoxam immediately. Spray both leaf surfaces. ⚠️ These pests transmit viral diseases. Use sticky traps for monitoring and avoid close plant spacing. Re-inspect every 2-3 days."
        
        if "whorl" in label_lower and "maggot" in label_lower:
            return "Whorl maggot detected. Apply Carbofuran 3G granules directly in plant whorl or spray Chlorpyrifos 20 EC (2.5ml/liter). Remove heavily infested plants. Use seed treatment with Imidacloprid for prevention."
        
        if "bug" in label_lower and "rice" in label_lower:
            return "Rice bug detected. Apply Malathion or Lambda-cyhalothrin. Hand-pick bugs in small infestations. Drain fields periodically to reduce populations. Remove weeds and grasses that harbor bugs."
        
        if "aphid" in label_lower:
            return "Aphid infestation detected. Spray neem oil (5ml/liter) or insecticidal soap. Introduce ladybugs for biological control. ⚠️ Aphids transmit viruses - act quickly! Use yellow sticky traps and remove weeds that harbor them."
        
        if "whitefly" in label_lower or "white fly" in label_lower:
            return "Whitefly detected. Install yellow sticky traps immediately. Spray neem oil (10ml/liter) on leaf undersides. Apply Imidacloprid if severe. Remove heavily infested leaves. ⚠️ Whiteflies spread viral diseases."
        
        if "spider" in label_lower and "mite" in label_lower:
            return "Spider mites detected. Spray plants with strong water jet daily, especially leaf undersides. Apply miticide (Abamectin) or neem oil every 3 days. Increase humidity to 60-70%. Release predatory mites for biological control."
        
        if "thrip" in label_lower:
            return "Thrips detected. Install blue sticky traps. Spray Spinosad or insecticidal soap. Remove infested plant parts. Release predatory mites (Amblyseius cucumeris). Use reflective mulch and remove surrounding weeds."
        
        return "Pest detected. Remove affected plant parts, apply neem oil or appropriate insecticide. Maintain field hygiene and monitor daily. Consult local agricultural expert for specific identification."
    
    # ==================== APPLE DISEASES ====================
    if "black rot" in label_lower and ("apple" in label_lower or "Apple" in model_name):
        return "Black rot detected. Remove and destroy infected fruit/leaves immediately. Prune infected branches 6-8 inches below damage. Apply Captan or Mancozeb fungicide every 7-10 days. Clean up all fallen debris."
    
    if "blotch" in label_lower and ("apple" in label_lower or "Apple" in model_name):
        return "Apple blotch detected. Apply copper-based fungicide or Mancozeb every 10-14 days. Remove infected leaves and fruit. Prune for better air circulation. Maintain good orchard sanitation."
    
    if "cedar" in label_lower and "rust" in label_lower:
        return "Cedar apple rust detected. Remove nearby cedar/juniper trees if possible (alternate host). Apply Myclobutanil or sulfur fungicide at bud break and repeat every 7-10 days. Plant resistant varieties."
    
    if "scab" in label_lower and ("apple" in label_lower or "Apple" in model_name):
        return "Apple scab detected. Apply Captan fungicide immediately and repeat every 7-10 days during wet weather. Remove infected leaves. Rake and destroy fallen leaves. Apply dormant spray before bud break next season."
    
    if "rotten" in label_lower and ("apple" in label_lower or "Apple" in model_name):
        return "Rotten apple detected. Remove immediately to prevent spread. Improve air circulation, avoid wounding fruit, and control insects that create entry points. Harvest at proper maturity and handle carefully."
    
    # ==================== GRAPE DISEASES ====================
    if "black rot" in label_lower and ("grape" in label_lower or "Grape" in model_name):
        return "Grape black rot detected. Remove all infected berries, leaves, and mummified fruit. Apply Mancozeb or Myclobutanil fungicide every 10-14 days. Prune for air circulation. Clean up all vineyard debris."
    
    if "blight" in label_lower and ("grape" in label_lower or "Grape" in model_name):
        return "Grape blight detected. Remove infected tissue immediately. Spray copper fungicide or Mancozeb. Improve air circulation through proper pruning. Avoid overhead irrigation. Apply fungicide preventatively during wet periods."
    
    if "esca" in label_lower:
        return "Esca disease detected (serious vascular disease). Remove severely affected vines. Prune out infected wood during dormancy. Avoid pruning wounds when possible. No effective chemical treatment - focus on prevention and sanitation."
    
    if "insect hole" in label_lower and "leaf" in label_lower:
        return "Insect damage on grape leaves. Identify specific pest (likely grape flea beetle or leafhopper). Apply appropriate insecticide or neem oil. Remove heavily damaged leaves. Monitor regularly for pest activity."
    
    # ==================== TOMATO DISEASES ====================
    if "bacterial spot" in label_lower:
        return "Bacterial spot detected. Remove infected leaves immediately. Apply copper-based bactericide every 5-7 days. Avoid overhead watering. Disinfect tools between plants. Use disease-free seeds and resistant varieties."
    
    if "early blight" in label_lower:
        return "Early blight detected. Remove lower infected leaves. Apply Chlorothalonil or copper fungicide every 7-10 days. Mulch to prevent soil splash. Stake plants for better air circulation. Rotate crops yearly."
    
    if "late blight" in label_lower:
        return "Late blight detected - URGENT! Remove and destroy infected plants immediately. Apply Chlorothalonil or Mancozeb every 5-7 days. Avoid overhead watering. This disease spreads rapidly in cool, wet conditions."
    
    if "leaf mold" in label_lower:
        return "Leaf mold detected. Reduce humidity below 85%. Improve ventilation. Remove infected lower leaves. Apply sulfur or copper fungicide. Space plants wider. Water at soil level only, not foliage."
    
    if "septoria" in label_lower:
        return "Septoria leaf spot detected. Remove infected leaves from bottom of plant upward. Apply Chlorothalonil or copper fungicide every 7-10 days. Mulch soil to prevent splash. Rotate crops and stake plants."
    
    if "target spot" in label_lower:
        return "Target spot detected. Remove infected leaves. Apply Chlorothalonil or Azoxystrobin fungicide every 7-14 days. Improve air circulation. Avoid overhead irrigation. Use mulch to reduce soil splash."
    
    if "yellow leaf curl" in label_lower or "curl virus" in label_lower:
        return "Yellow leaf curl virus detected. Remove infected plants immediately to prevent spread. Control whiteflies aggressively (primary vector). Use yellow sticky traps. Plant virus-resistant varieties. No cure available."
    
    if "mosaic virus" in label_lower:
        return "Mosaic virus detected. Remove and destroy infected plants. Control aphids (virus vector). Disinfect tools with 10% bleach. Wash hands between handling plants. Plant resistant varieties. No chemical cure available."
    
    if "viral" in label_lower and "tomato" in label_lower:
        return "Viral infection detected. Isolate or remove infected plants immediately. Control insect vectors (aphids, whiteflies). Disinfect tools between plants. Plant virus-resistant varieties. Viruses cannot be cured chemically."
    
    if "wilt" in label_lower and "tomato" in label_lower:
        return "Wilt disease detected (likely Fusarium or Verticillium). Remove infected plants immediately. Do not replant tomatoes in same soil for 3-4 years. Use wilt-resistant varieties. Maintain soil pH 6.5-7.0. Improve drainage."
    
    if "graymold" in label_lower or "gray mold" in label_lower:
        return "Gray mold (Botrytis) detected. Remove infected fruit and tissue immediately. Improve air circulation and reduce humidity. Avoid overhead watering. Apply Chlorothalonil or sulfur fungicide. Don't over-fertilize with nitrogen."
    
    # ==================== LETTUCE DISEASES ====================
    if "lettuce" in label_lower and ("rot" in label_lower or "disease" in label_lower):
        return "Lettuce disease detected. Improve air circulation and reduce leaf wetness. Avoid overhead watering. Remove infected plants. Apply copper fungicide if fungal. Ensure proper spacing (20-30cm apart)."
    
    # ==================== BASIL DISEASES ====================
    if "basil" in label_lower and ("disease" in label_lower or "rot" in label_lower):
        return "Basil disease detected. Remove infected leaves immediately. Reduce humidity and improve air flow. Water at soil level only. Apply copper fungicide if fungal. Avoid overcrowding plants."
    
    # ==================== HEALTHY PLANTS ====================
    if "healthy" in label_lower:
        return "Healthy crop detected! Continue routine monitoring and maintain current care practices. Ensure consistent watering, proper nutrition, and good air circulation. Check regularly for early signs of pests or diseases."
    
    # ==================== GENERAL DISEASE PATTERNS ====================
    if "blight" in label_lower:
        return "Blight detected. Remove infected tissue immediately. Apply copper-based fungicide or Mancozeb. Improve air circulation through pruning. Avoid overhead watering. Apply preventative fungicide during wet weather."
    
    if "scab" in label_lower:
        return "Scab disease detected. Apply Captan or Mancozeb fungicide every 7-10 days. Remove infected plant parts. Rake up and destroy fallen leaves. Improve air circulation and reduce leaf wetness."
    
    if "rot" in label_lower:
        return "Rot detected. Remove affected parts immediately. Improve ventilation and reduce humidity. Avoid overwatering. Apply copper fungicide. Ensure proper drainage and spacing between plants."
    
    if "rust" in label_lower:
        return "Rust disease detected. Remove infected leaves. Apply sulfur-based fungicide or Myclobutanil every 7-14 days. Improve air circulation. Water at soil level. Avoid working with plants when wet."
    
    if "spot" in label_lower:
        return "Leaf spot disease detected. Remove infected leaves. Apply copper or Chlorothalonil fungicide every 7-10 days. Improve air circulation. Mulch to prevent soil splash. Avoid overhead watering."
    
    if "mold" in label_lower or "mildew" in label_lower:
        return "Mold/mildew detected. Reduce humidity and improve air flow. Remove infected tissue. Apply sulfur or potassium bicarbonate fungicide. Space plants properly. Water early in day so foliage dries quickly."
    
    if "viral" in label_lower or "virus" in label_lower:
        return "Viral infection detected. Remove and destroy infected plants immediately. Control insect vectors (aphids, whiteflies, thrips). Disinfect tools with 10% bleach solution. Plant resistant varieties. No chemical cure exists."
    
    # ==================== DEFAULT ====================
    return "Monitor crop closely for symptoms. Maintain good cultural practices: proper watering, adequate spacing, and regular inspection. Remove any diseased tissue promptly. Consult local agricultural extension if symptoms persist."

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
# MAIN APPLICATION
# ==========================================================
class CropDetectionApp(ctk.CTk):
    def __init__(self):
        super().__init__()
        self.title("🌿 CropVision Detection System")
        self.geometry("1400x850")
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
        header = ctk.CTkFrame(self, fg_color=("#2d5016", "#1a3010"), height=80)
        header.pack(fill="x", padx=0, pady=0)
        header.pack_propagate(False)
        
        ctk.CTkLabel(
            header, 
            text="🌿 CropVision Detection System", 
            font=("Segoe UI", 30, "bold"),
            text_color="white"
        ).pack(pady=20)

        # Main container
        main_container = ctk.CTkFrame(self, fg_color="transparent")
        main_container.pack(fill="both", expand=True, padx=20, pady=10)

        # Left panel - Controls
        left_panel = ctk.CTkFrame(main_container, width=300)
        left_panel.pack(side="left", fill="y", padx=(0, 10))
        left_panel.pack_propagate(False)

        # Model Selection Section
        selection_frame = ctk.CTkFrame(left_panel)
        selection_frame.pack(fill="x", padx=15, pady=(15, 10))

        ctk.CTkLabel(
            selection_frame, 
            text="Model Configuration", 
            font=("Segoe UI", 18, "bold")
        ).pack(pady=(10, 8))

        ctk.CTkLabel(
            selection_frame, 
            text="Category:", 
            font=("Segoe UI", 14),
            anchor="w"
        ).pack(fill="x", padx=10, pady=(10, 2))

        self.category_menu = ctk.CTkOptionMenu(
            selection_frame,
            variable=self.category_var,
            values=list(CATEGORY_MODELS.keys()),
            command=self.on_category_change,
            height=38, 
            font=("Segoe UI", 14) 
        )
        self.category_menu.pack(fill="x", padx=10, pady=(0, 10))

        ctk.CTkLabel(
            selection_frame, 
            text="Detection Model:", 
            font=("Segoe UI", 14), 
            anchor="w"
        ).pack(fill="x", padx=10, pady=(5, 2))

        self.model_menu = ctk.CTkOptionMenu(
            selection_frame,
            variable=self.model_name,
            values=CATEGORY_MODELS[self.category_var.get()],
            height=38, 
            font=("Segoe UI", 14)
        )
        self.model_menu.pack(fill="x", padx=10, pady=(0, 15))

        # Control Buttons
        controls_frame = ctk.CTkFrame(left_panel)
        controls_frame.pack(fill="both", expand=True, padx=15, pady=10)

        ctk.CTkLabel(
            controls_frame, 
            text="Controls", 
            font=("Segoe UI", 18, "bold")
        ).pack(pady=(10, 15))

        btn_style = {
            "height": 50,
            "font": ("Segoe UI", 15),
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

        # Center panel
        center_panel = ctk.CTkFrame(main_container, width=600)
        center_panel.pack(side="left", fill="both", expand=True, padx=(0, 10))

        # Video section
        video_section = ctk.CTkFrame(center_panel)
        video_section.pack(fill="both", expand=True, padx=10, pady=10)

        ctk.CTkLabel(
            video_section, 
            text="Live Detection Feed", 
            font=("Segoe UI", 19, "bold")
        ).pack(pady=(10, 5))

        # Video frame container
        self.video_container = ctk.CTkFrame(video_section, fg_color="#000000")
        self.video_container.pack(fill="both", expand=True, padx=10, pady=10)

        self.video_label = ctk.CTkLabel(
            self.video_container, 
            text="No video feed\n\nClick 'Start Live Detection' to begin",
            font=("Segoe UI", 16),
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
            height=52,
            font=("Segoe UI", 16, "bold"),
            corner_radius=8
        )
        self.capture_btn.pack(fill="x", padx=10, pady=(5, 10))

        # Status bar at bottom of center panel
        status_frame = ctk.CTkFrame(center_panel, height=55)
        status_frame.pack(fill="x", padx=10, pady=(0, 10))
        status_frame.pack_propagate(False)

        ctk.CTkLabel(
            status_frame, 
            text="Status:", 
            font=("Segoe UI", 14, "bold")
        ).pack(side="left", padx=15, pady=12)

        self.status = ctk.CTkLabel(
            status_frame,
            text="Ready to start detection",
            font=("Segoe UI", 15),
            text_color="#28a745"
        )
        self.status.pack(side="left", padx=5, pady=12)

        # Right panel - Results and Recommendations
        right_panel = ctk.CTkFrame(main_container, width=420)
        right_panel.pack(side="right", fill="both", expand=False)
        right_panel.pack_propagate(False)

        # Results header
        results_header = ctk.CTkFrame(right_panel, fg_color=("#2d5016", "#1a3010"), height=55) 
        results_header.pack(fill="x", padx=10, pady=(10, 0))
        results_header.pack_propagate(False)

        ctk.CTkLabel(
            results_header,
            text="📊 Detection Results",
            font=("Segoe UI", 20, "bold"),
            text_color="white"
        ).pack(pady=12)

        # Scrollable results frame
        results_scroll = ctk.CTkScrollableFrame(right_panel, fg_color="transparent")
        results_scroll.pack(fill="both", expand=True, padx=10, pady=10)

        # Prediction section
        prediction_frame = ctk.CTkFrame(results_scroll)
        prediction_frame.pack(fill="x", pady=(0, 10))

        ctk.CTkLabel(
            prediction_frame,
            text="Prediction",
            font=("Segoe UI", 16, "bold"),
            anchor="w"
        ).pack(fill="x", padx=15, pady=(12, 5))

        self.prediction_label = ctk.CTkLabel(
            prediction_frame,
            text="No prediction yet",
            font=("Segoe UI", 17),
            text_color="#888888",
            anchor="w",
            wraplength=370, 
            justify="left"
        )
        self.prediction_label.pack(fill="x", padx=15, pady=(0, 5))

        self.confidence_label = ctk.CTkLabel(
            prediction_frame,
            text="Confidence: --",
            font=("Segoe UI", 15),  
            text_color="#888888",
            anchor="w"
        )
        self.confidence_label.pack(fill="x", padx=15, pady=(0, 12))

        # Recommendation section 
        recommendation_frame = ctk.CTkFrame(results_scroll)
        recommendation_frame.pack(fill="x", pady=(0, 10))

        ctk.CTkLabel(
            recommendation_frame,
            text="💡 AI Recommendation",
            font=("Segoe UI", 16, "bold"),  
            anchor="w"
        ).pack(fill="x", padx=15, pady=(12, 8))

        self.recommendation_text = ctk.CTkTextbox(
            recommendation_frame,
            height=400,
            font=("Segoe UI", 13),  
            wrap="word",
            fg_color=("#f0f0f0", "#2b2b2b"),
            text_color=("#000000", "#ffffff")
        )
        self.recommendation_text.pack(fill="both", expand=True, padx=15, pady=(0, 12))
        self.recommendation_text.insert("1.0", "Waiting for detection results...\n\nStart live detection or upload an image to receive AI-powered recommendations.")
        self.recommendation_text.configure(state="disabled")

        # Upload status section
        upload_frame = ctk.CTkFrame(results_scroll)
        upload_frame.pack(fill="x", pady=(0, 10))

        ctk.CTkLabel(
            upload_frame,
            text="☁️ Upload Status",
            font=("Segoe UI", 15, "bold"),  
            anchor="w"
        ).pack(fill="x", padx=15, pady=(10, 5))

        self.firebase_status = ctk.CTkLabel(
            upload_frame,
            text="Firebase: Pending",
            font=("Segoe UI", 13), 
            text_color="#888888",
            anchor="w"
        )
        self.firebase_status.pack(fill="x", padx=15, pady=2)

        self.mqtt_status = ctk.CTkLabel(
            upload_frame,
            text="MQTT: Pending",
            font=("Segoe UI", 13), 
            text_color="#888888",
            anchor="w"
        )
        self.mqtt_status.pack(fill="x", padx=15, pady=(2, 10))

    def update_results_panel(self, label, confidence, recommendation, firebase_success=None, mqtt_success=None):
        """Update the results panel with prediction and recommendation"""
        # Update prediction
        is_healthy = "healthy" in label.lower()
        color = "#28a745" if is_healthy else "#dc3545"
        
        self.prediction_label.configure(text=label, text_color=color)
        self.confidence_label.configure(
            text=f"Confidence: {confidence*100:.1f}%",
            text_color=color
        )

        # Update recommendation
        self.recommendation_text.configure(state="normal")
        self.recommendation_text.delete("1.0", "end")
        self.recommendation_text.insert("1.0", recommendation)
        self.recommendation_text.configure(state="disabled")

        # Update upload status if provided
        if firebase_success is not None:
            if firebase_success:
                self.firebase_status.configure(
                    text="Firebase: ✅ Uploaded",
                    text_color="#28a745"
                )
            else:
                self.firebase_status.configure(
                    text="Firebase: ❌ Failed",
                    text_color="#dc3545"
                )
        
        if mqtt_success is not None:
            if mqtt_success:
                self.mqtt_status.configure(
                    text="MQTT: ✅ Published",
                    text_color="#28a745"
                )
            else:
                self.mqtt_status.configure(
                    text="MQTT: ❌ Failed",
                    text_color="#dc3545"
                )

    def clear_results_panel(self):
        """Clear the results panel"""
        self.prediction_label.configure(text="No prediction yet", text_color="#888888")
        self.confidence_label.configure(text="Confidence: --", text_color="#888888")
        
        self.recommendation_text.configure(state="normal")
        self.recommendation_text.delete("1.0", "end")
        self.recommendation_text.insert("1.0", "Waiting for detection results...\n\nStart live detection or upload an image to receive AI-powered recommendations.")
        self.recommendation_text.configure(state="disabled")
        
        self.firebase_status.configure(text="Firebase: Pending", text_color="#888888")
        self.mqtt_status.configure(text="MQTT: Pending", text_color="#888888")

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
        self.clear_results_panel()

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
            recommendation = get_recommendation(self.label_text, self.model_name.get())
            
            # Update results panel
            self.update_results_panel(self.label_text, self.conf, recommendation)
            
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
            firebase_success = image_url is not None
            
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
            
            # Update results panel with upload status
            self.update_results_panel(label_text, conf, recommendation, firebase_success, mqtt_success)
            
            # Show simple success message
            if firebase_success and mqtt_success:
                messagebox.showinfo("Capture Success", f"Screenshot captured and processed successfully!\n\nSaved to: {filepath}")
            else:
                messagebox.showwarning("Partial Success", f"Screenshot captured but some uploads failed.\n\nSaved to: {filepath}")
            
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
            firebase_success = image_url is not None

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

            # Update results panel
            self.update_results_panel(label_text, conf, recommendation, firebase_success, mqtt_success)

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
            
            messagebox.showinfo("Success", f"Image uploaded and processed successfully!\n\nCheck the results panel for details.")

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
    print("[LOG] 🚀 Starting CropVision...")
    app = CropDetectionApp()
    app.mainloop()
