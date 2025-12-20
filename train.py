import os
from dataclasses import dataclass

from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader, random_split

from transformers import (
    AutoImageProcessor,
    AutoModelForImageClassification,
    get_scheduler,
)

# Device setup

if torch.backends.mps.is_available():
    device = "mps"
elif torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"

print(f"Using device: {device}")

# MPS matmul precision to improve training performance
if device == "mps":
    torch.set_float32_matmul_precision("high")

# Emotion name -> Model label mapping
emotion_to_label = {
    "angry": 2,
    "disgust": 1,
    "fear": 4,
    "happy": 6,
    "neutral": 3,
    "sad": 0,
    "surprise": 5,
}

# Custom collate function

def collate_fn(batch):
    #Keeps PIL images as Python objects so that the HuggingFace processor can handle them
    images = [item[0] for item in batch]
    labels = torch.tensor([item[1] for item in batch])
    return images, labels


# Emotion Dataset

class EmotionDataset(Dataset):
    def __init__(self, root_dir):
        self.samples = []
        # Sort label folders
        for label_folder in sorted(os.listdir(root_dir)):
            folder_path = os.path.join(root_dir, label_folder)
            if not os.path.isdir(folder_path):
                continue

            # Map folder name to model label
            if label_folder.lower() not in emotion_to_label:
                print(f"Warning: Unknown emotion folder '{label_folder}', skipping...")
                continue
            
            true_label = emotion_to_label[label_folder.lower()]
            # Append image path and true label to samples
            for file in os.listdir(folder_path):
                img_path = os.path.join(folder_path, file)
                self.samples.append((img_path, true_label))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        print("Loading:", img_path)
        image = Image.open(img_path).convert("RGB")
        return image, label


# Training one epoch

def train_one_epoch(model, processor, dataloader, optimizer, epoch, num_epochs):
    model.train()
    total_loss = 0.0
    total_correct = 0
    total = 0

    for step, (images, labels) in enumerate(dataloader):

        inputs = processor(images=images, return_tensors="pt").to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        # Autocast
        with torch.autocast(device_type=device, dtype=torch.float16, enabled=(device == "mps")):
            outputs = model(**inputs, labels=labels)
            loss = outputs.loss
            logits = outputs.logits

        loss.backward()
        optimizer.step()

        # Get predictions
        preds = logits.argmax(dim=1)
        total_loss += loss.item() * labels.size(0)
        total_correct += (preds == labels).sum().item()
        total += labels.size(0)

    return total_loss / total, total_correct / total


# Evaluate function, prints loss and accuracy for each epoch
@torch.no_grad()
def evaluate(model, processor, dataloader, desc="Eval"):
    model.eval()
    total_loss = 0
    total_correct = 0
    total = 0
    for images, labels in dataloader:
        inputs = processor(images=images, return_tensors="pt").to(device)
        labels = labels.to(device)
        # Autocast
        with torch.autocast(device_type=device, dtype=torch.float16, enabled=(device == "mps")):
            outputs = model(**inputs, labels=labels)
            loss = outputs.loss
            logits = outputs.logits

        # Get predictions
        preds = logits.argmax(dim=1)
        total_loss += loss.item() * labels.size(0)
        total_correct += (preds == labels).sum().item()
        total += labels.size(0)

    print(f"[{desc}] Loss: {total_loss / total:.4f} | Acc: {total_correct / total:.4f}")
    return total_loss / total, total_correct / total


# Main training loop

def main():
    # Get current directory and dataset root
    current_dir = os.getcwd()
    dataset_root = os.path.join(current_dir, "DATASET")

    train_root = os.path.join(dataset_root, "train")
    test_root = os.path.join(dataset_root, "test")

    full_train_dataset = EmotionDataset(train_root)
    test_dataset = EmotionDataset(test_root)

    print(f"Train samples: {len(full_train_dataset)}")
    print(f"Test samples:  {len(test_dataset)}")

    # Train/val split
    val_size = int(len(full_train_dataset) * 0.2)
    train_size = len(full_train_dataset) - val_size

    train_dataset, val_dataset = random_split(
        full_train_dataset,
        [train_size, val_size],
        generator=torch.Generator().manual_seed(42),
    )

    batch_size = 32

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
                              num_workers=0, collate_fn=collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False,
                            num_workers=0, collate_fn=collate_fn)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False,
                             num_workers=0, collate_fn=collate_fn)

    # Load a model for Transfer Learning and get its processor and model
    model_name = "facebook/deit-small-patch16-224"
    processor = AutoImageProcessor.from_pretrained(model_name)
    model = AutoModelForImageClassification.from_pretrained(model_name).to(device)

    # AdamW optimizer with learning rate of 5e-5
    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5)

    num_epochs = 15
    total_steps = num_epochs * len(train_loader)

    # Linear learning rate scheduler
    scheduler = get_scheduler(
        "linear",
        optimizer=optimizer,
        num_warmup_steps=int(0.1 * total_steps),
        num_training_steps=total_steps,
    )

    best_val_acc = 0.0
    save_dir = os.path.join(current_dir, "models", "fine_tuned_emotion_model")

    # Training loop with validation, prints loss and accuracy for each epoch
    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch+1}/{num_epochs}")

        train_loss, train_acc = train_one_epoch(model, processor, train_loader, optimizer, epoch, num_epochs)
        print(f"Training Loss: {train_loss:.4f} | Training Accuracy: {train_acc:.4f}")
        val_loss, val_acc = evaluate(model, processor, val_loader, desc="Validation")
        print(f"Validation Loss: {val_loss:.4f} | Validation Accuracy: {val_acc:.4f}")
        scheduler.step()

if __name__ == "__main__":
    main()
