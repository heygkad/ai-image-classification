import os
from dataclasses import dataclass
import matplotlib.pyplot as plt

from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader, random_split

from transformers import (
    AutoImageProcessor,
    AutoModelForImageClassification,
    get_scheduler,
)

########################################
# Device setup (M4 optimized)
########################################

if torch.backends.mps.is_available():
    device = "mps"
elif torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"

print(f"Using device: {device}")

# MPS matmul precision improves ViT performance
if device == "mps":
    torch.set_float32_matmul_precision("high")


########################################
# Custom collate function (critical for HF)
########################################

def collate_fn(batch):
    """
    Keeps PIL images as Python objects (HF processor handles them)
    and batches labels into a tensor.
    """
    images = [item[0] for item in batch]  # do NOT convert to tensor
    labels = torch.tensor([item[1] for item in batch])
    return images, labels


########################################
# Emotion Dataset
########################################

FER_LABEL_MAP = {
    "angry": 0,
    "disgust": 1,
    "fear": 2,
    "happy": 3,
    "sad": 4,
    "surprise": 5,
    "neutral": 6
}


class EmotionDataset(Dataset):
    def __init__(self, root_dir):
        self.samples = []

        for class_name, fer_label in FER_LABEL_MAP.items():
            class_dir = os.path.join(root_dir, class_name)
            if not os.path.isdir(class_dir):
                continue

            for file in os.listdir(class_dir):
                img_path = os.path.join(class_dir, file)
                self.samples.append((img_path, fer_label))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        image = Image.open(img_path).convert("RGB")
        return image, label




########################################
# Training / Eval (M4-optimized with autocast)
########################################

from tqdm import tqdm

def train_one_epoch(model, processor, dataloader, optimizer, epoch, num_epochs):
    model.train()
    total_loss = 0.0
    total_correct = 0
    total = 0

    loop = tqdm(dataloader, desc=f"Train Epoch {epoch+1}", ncols=100)

    for step, (images, labels) in enumerate(loop):

        inputs = processor(images=images, return_tensors="pt").to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        with torch.autocast(device_type=device, dtype=torch.float16, enabled=(device == "mps")):
            outputs = model(**inputs, labels=labels)
            loss = outputs.loss
            logits = outputs.logits

        loss.backward()
        optimizer.step()

        preds = logits.argmax(dim=1)
        total_loss += loss.item() * labels.size(0)
        total_correct += (preds == labels).sum().item()
        total += labels.size(0)

        loop.set_postfix(loss=loss.item(), acc=(total_correct / total))

    return total_loss / total, total_correct / total

@torch.no_grad()
def evaluate(model, processor, dataloader, desc="Eval"):
    model.eval()
    total_loss = 0
    total_correct = 0
    total = 0

    for batch_i, (images, labels) in enumerate(
            tqdm(dataloader, desc=f"{desc} Progress", ncols=100)
        ):
        
        inputs = processor(images=images, return_tensors="pt").to(device)
        labels = labels.to(device)

        with torch.autocast(device_type=device, dtype=torch.float16, enabled=(device == "mps")):
            outputs = model(**inputs, labels=labels)
            loss = outputs.loss
            logits = outputs.logits

        preds = logits.argmax(dim=1)
        total_loss += loss.item() * labels.size(0)
        total_correct += (preds == labels).sum().item()
        total += labels.size(0)

    print(f"[{desc}] Loss: {total_loss / total:.4f} | Acc: {total_correct / total:.4f}")
    return total_loss / total, total_correct / total



########################################
# Main training loop
########################################

def main():
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

    batch_size = 16

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
                              num_workers=0, collate_fn=collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False,
                            num_workers=0, collate_fn=collate_fn)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False,
                             num_workers=0, collate_fn=collate_fn)

    # Load vision transformer
    model_name = "google/vit-base-patch16-224"

    id2label = {
        0: "angry",
        1: "disgust",
        2: "fear",
        3: "happy",
        4: "sad",
        5: "surprise",
        6: "neutral"
    }

    label2id = {v: k for k, v in id2label.items()}

    processor = AutoImageProcessor.from_pretrained(
        model_name,
        id2label=id2label,
        label2id=label2id
    )

    model = AutoModelForImageClassification.from_pretrained(
        model_name,
        num_labels=7,
        ignore_mismatched_sizes=True,
        id2label=id2label,
        label2id=label2id
    ).to(device)

    model.config.classifier_dropout = 0.1
    model.config.label_smoothing = 0.1

    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5)

    num_epochs = 15
    total_steps = num_epochs * len(train_loader)

    scheduler = get_scheduler(
        "linear",
        optimizer=optimizer,
        num_warmup_steps=int(0.1 * total_steps),
        num_training_steps=total_steps,
    )

    best_val_acc = 0.0
    def create_new_model_dir(base_name="fine_tuned_emotion_model"):
        current_dir = os.getcwd()

        # Find all existing model folders
        existing = [
            d for d in os.listdir(current_dir)
            if d.startswith(base_name) and os.path.isdir(os.path.join(current_dir, d))
        ]

        # Determine next model version
        if len(existing) == 0:
            version = 1
        else:
            # Extract numbers at the end of folder names
            nums = []
            for name in existing:
                try:
                    nums.append(int(name.split("_")[-1]))
                except:
                    pass
            version = max(nums) + 1 if nums else 1

        # Zero-padded folder name
        new_name = f"{base_name}_{version:03d}"
        new_path = os.path.join(current_dir, new_name)

        os.makedirs(new_path, exist_ok=True)
        return new_path

    save_dir = create_new_model_dir()
    print("Model will be saved to:", save_dir)

    import matplotlib.pyplot as plt

    def plot_and_save_metrics(train_losses, val_losses, train_accuracies, val_accuracies, save_dir):
        epochs = range(1, len(train_losses) + 1)

        # --- Loss Plot ---
        plt.figure(figsize=(10,5))
        plt.plot(epochs, train_losses, label="Train Loss", marker='o')
        plt.plot(epochs, val_losses, label="Val Loss", marker='o')
        plt.title("Loss Curve")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.grid(True)
        plt.legend()
        plt.savefig(os.path.join(save_dir, "loss_curve.png"))
        plt.close()

        # --- Accuracy Plot ---
        plt.figure(figsize=(10,5))
        plt.plot(epochs, train_accuracies, label="Train Accuracy", marker='o')
        plt.plot(epochs, val_accuracies, label="Val Accuracy", marker='o')
        plt.title("Accuracy Curve")
        plt.xlabel("Epoch")
        plt.ylabel("Accuracy")
        plt.grid(True)
        plt.legend()
        plt.savefig(os.path.join(save_dir, "accuracy_curve.png"))
        plt.close()

        print(f"Saved training graphs to {save_dir}")

    train_losses = []
    train_accuracies = []
    val_losses = []
    val_accuracies = []

    for epoch in range(num_epochs):
        print(f"\n=== Epoch {epoch+1}/{num_epochs} ===")

        train_loss, train_acc = train_one_epoch(model, processor, train_loader, optimizer, epoch, num_epochs)
        train_losses.append(train_loss)
        train_accuracies.append(train_acc)

        print(f"[Train] Loss: {train_loss:.4f} | Acc: {train_acc:.4f}")

        val_loss, val_acc = evaluate(model, processor, val_loader, desc="Validation")
        val_losses.append(val_loss)
        val_accuracies.append(val_acc)

        scheduler.step()

        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            model.save_pretrained(save_dir)
            processor.save_pretrained(save_dir)
            print(f"Saved improved model → {save_dir}")

    # Final test evaluation
    print("\n=== FINAL TEST ===")
    evaluate(model, processor, test_loader, desc="Test")

    plot_and_save_metrics(train_losses, val_losses, train_accuracies, val_accuracies, save_dir)

if __name__ == "__main__":
    main()
