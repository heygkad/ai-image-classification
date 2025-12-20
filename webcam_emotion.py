import cv2
import torch
import time
from PIL import Image
from transformers import AutoImageProcessor, AutoModelForImageClassification

# Select device
if torch.backends.mps.is_available():
    device = "mps"
elif torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"

print(f"Using device: {device}")
torch.set_float32_matmul_precision("high")

# Our trained models
MODEL_OPTIONS = [
    "models/ConvNext",
    "models/DeIT-Small",
    "models/DeIT-Small-Undersampled",
    "models/ViT-Base",
    "models/ViT-Base-Undersampled",
]

# Color mapping for boundary boxes
EMOTION_COLORS = {
    "angry":    (75, 85, 240),
    "disgust":  (80, 175, 180),
    "fear":     (180, 130, 200),
    "happy":    (100, 220, 160),
    "sad":      (220, 160, 100),
    "surprise": (130, 200, 255),
    "neutral":  (180, 180, 180),
}

# Initial model state
current_model_idx = 0
processor = None
model = None
dropdown_open = False
should_stop = False

# Face detection cascade (Haar Cascades)
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')


def load_model(model_path):
    global processor, model
    processor = AutoImageProcessor.from_pretrained(model_path)
    model = AutoModelForImageClassification.from_pretrained(model_path).to(device).eval()
    print("Model loaded")
    # Warm-up for MPS/GPU to improve performance, like we did in tuned_model_inference.py
    with torch.no_grad():
        _ = model(torch.ones((1, 3, 224, 224)).to(device))


def mouse_callback(event, x, y, flags, param):
    global dropdown_open, current_model_idx, should_stop
    
    if event == cv2.EVENT_LBUTTONDOWN:
        # Stop button
        if 260 <= x <= 310 and 5 <= y <= 30:
            should_stop = True
        # Dropdown button area
        elif 10 <= x <= 250 and 5 <= y <= 30:
            dropdown_open = not dropdown_open
        elif dropdown_open and 10 <= x <= 250:
            for i in range(len(MODEL_OPTIONS)):
                item_y = 30 + i * 25
                if item_y <= y <= item_y + 25:
                    if i != current_model_idx:
                        current_model_idx = i
                        load_model(MODEL_OPTIONS[i])
                    dropdown_open = False
                    break
            else:
                dropdown_open = False
        else:
            dropdown_open = False


def predict_emotion(face_img):
    face_rgb = cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB)
    pil_image = Image.fromarray(face_rgb)
    inputs = processor(images=pil_image, return_tensors="pt").to(device)
    
    with torch.no_grad():
        logits = model(**inputs).logits
    
    probs = logits.softmax(dim=1)
    pred_id = probs.argmax(dim=1).item()
    confidence = probs.max().item()
    label = model.config.id2label[pred_id]
    return label, confidence


def draw_face_box(frame, x, y, w, h, color, thickness=2):
    cv2.rectangle(frame, (x, y), (x + w, y + h), color, thickness)


def draw_label(frame, text, x, y, color):
    cv2.putText(frame, text, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)


def draw_dropdown(frame):
    model_name = MODEL_OPTIONS[current_model_idx].split("/")[-1]
    
    # Dropdown button
    cv2.rectangle(frame, (10, 5), (250, 30), (40, 40, 40), -1)
    cv2.putText(frame, model_name, (15, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    
    # Stop button
    cv2.rectangle(frame, (260, 5), (310, 30), (50, 50, 200), -1)
    cv2.putText(frame, "STOP", (265, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)
    
    if dropdown_open:
        for i, opt in enumerate(MODEL_OPTIONS):
            y_pos = 30 + i * 25
            color = (80, 80, 80) if i == current_model_idx else (40, 40, 40)
            cv2.rectangle(frame, (10, y_pos), (250, y_pos + 25), color, -1)
            cv2.putText(frame, opt.split("/")[-1], (15, y_pos + 18), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1)


def open_camera():
    for idx in [0, 1, 2]:
        cap = cv2.VideoCapture(idx)
        if cap.isOpened():
            time.sleep(0.5)
            ret, frame = cap.read()
            if ret and frame is not None:
                return cap
            cap.release()
    return None


def main():
    load_model(MODEL_OPTIONS[current_model_idx])
    
    cap = open_camera()
    
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    
    cv2.namedWindow("Emotion", cv2.WINDOW_AUTOSIZE)
    cv2.setMouseCallback("Emotion", mouse_callback)

    frame_count = 0
    skip_frames = 3
    last_emotion = "neutral"
    
    while True:
        ret, frame = cap.read()
        if not ret or frame is None:
            continue
        
        frame_count += 1
        frame = cv2.flip(frame, 1)
        
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.1, 5, minSize=(80, 80))
        
        for (x, y, w, h) in faces:
            pad = int(0.12 * w)
            x1, y1 = max(0, x - pad), max(0, y - pad)
            x2, y2 = min(frame.shape[1], x + w + pad), min(frame.shape[0], y + h + pad)
            face_roi = frame[y1:y2, x1:x2]
            
            if frame_count % skip_frames == 0:
                try:
                    emotion, _ = predict_emotion(face_roi)
                    last_emotion = emotion
                except:
                    pass
            
            color = EMOTION_COLORS.get(last_emotion, (180, 180, 180))
            
            draw_face_box(frame, x, y, w, h, color, 2)
            draw_label(frame, last_emotion.upper(), x, y, color)
        
        draw_dropdown(frame)
        
        cv2.imshow("Emotion", frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q') or should_stop:
            break
    
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
