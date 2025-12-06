from transformers import AutoImageProcessor, AutoModelForImageClassification
import torch


if torch.backends.mps.is_available():
    # uses mps to run model if you are on an apple silicon device
    device = "mps"
elif torch.cuda.is_available():
    # uses cuda if you have an NVIDIA GPU, most likely a windows or linux
    device = "cuda"
else:
    # otherwise uses cpu to run model
    device = "cpu"
print("Using device:", device)
# setting speed of float32 matrix multiplications
torch.set_float32_matmul_precision("high")


# this is the backend model class
class EmotionModel:
    # initialize the pretrained model from HuggingFace
    def __init__(self, model_name="dima806/facial_emotions_image_detection"):
        self.processor = AutoImageProcessor.from_pretrained(model_name)
        self.model = AutoModelForImageClassification.from_pretrained(model_name).to(device).eval()
        print("MODEL LABEL ORDER:", self.model.config.id2label)

        # run forward pass with a dummy image to warm up gpu
        dummy = torch.ones((1, 3, 224, 224)).to(device)
        with torch.no_grad():
            warm = self.model(dummy)

    # function to classify the image 
    def predict(self, image):
        # converts image into pytorch tensors
        inputs = self.processor(images=image, return_tensors="pt").to(device)
        # gets logits and uses it in an argmax function to classify the label
        with torch.no_grad():
            logits = self.model(**inputs).logits
        return logits.argmax(dim=1).item()
    

    
    # evaluates the model accuracy
    def evaluate(self, images, labels):
        FER_TO_MODEL = {
            0: 2, # angry
            1: 1, # disgust
            2: 4, # fear
            3: 6, # happy
            4: 0, # sad
            5: 5, # surprise
            6: 3, # neutral
        }
        correct = 0
        total = len(images)
        img_num = 0
        for image, label in zip(images, labels):
            img_num += 1
            print(f"Model 1 predicting image: {img_num}")
            prediction = self.predict(image)
            mapped_label = FER_TO_MODEL[label]
            if prediction == mapped_label:
                correct += 1
        acc = correct/total
        return acc