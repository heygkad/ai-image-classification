from transformers import AutoImageProcessor, AutoModelForImageClassification
import torch
from PIL import Image

# -----------------------
# Device Selection
# -----------------------
if torch.backends.mps.is_available():
    device = "mps"
elif torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"

print("Using device:", device)

# Improve MPS / CPU transformer performance
torch.set_float32_matmul_precision("high")


# -----------------------
# Fine-Tuned Emotion Model Wrapper
# -----------------------
class EmotionModelTuned:
    def __init__(self, model_path="fine_tuned_emotion_model"):
        """
        Loads your fine-tuned HuggingFace model from the local folder.
        """

        # Load custom processor + model from local directory
        self.processor = AutoImageProcessor.from_pretrained(model_path)
        self.model = AutoModelForImageClassification.from_pretrained(
            model_path
        ).to(device).eval()

        print("Loaded fine-tuned model from:", model_path)

        # Optional: warm-up for MPS/GPU
        dummy = torch.ones((1, 3, 224, 224)).to(device)
        with torch.no_grad():
            _ = self.model(dummy)

    def predict(self, image):
        """
        Predict a single PIL image.
        Returns (predicted_label, confidence)
        """

        # Ensure RGB
        if not isinstance(image, Image.Image):
            raise ValueError("Input must be a PIL.Image")

        inputs = self.processor(images=image, return_tensors="pt").to(device)

        with torch.no_grad():
            outputs = self.model(**inputs)
            logits = outputs.logits

        probs = logits.softmax(dim=1)
        pred_id = probs.argmax(dim=1).item()
        confidence = probs.max().item()

        # Convert using id2label mapping
        label_str = self.model.config.id2label[pred_id]

        return label_str, confidence

    def evaluate(self, images, labels):
        """
        Evaluate accuracy over a list of (PIL.Image, int_label).
        """

        correct = 0
        total = len(images)
        img_num = 0
        for image, label in zip(images, labels):
            pred_label_str, _ = self.predict(image)
            img_num += 1
            print(f"Model 1 predicting image: {img_num}")

            # Convert label string back to class ID
            pred_id = int([k for k, v in self.model.config.id2label.items() if v == pred_label_str][0])

            if pred_id == label:
                correct += 1

        accuracy = correct / total
        return accuracy
