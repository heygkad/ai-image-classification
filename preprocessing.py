import os
from PIL import Image

FER_LABEL_MAP = {
    "angry": 0,
    "disgust": 1,
    "fear": 2,
    "happy": 3,
    "sad": 4,
    "surprise": 5,
    "neutral": 6
}

def load_split(split_root):
    images = []
    labels = []

    for class_name, class_idx in FER_LABEL_MAP.items():
        class_dir = os.path.join(split_root, class_name)

        if not os.path.isdir(class_dir):
            continue

        for img_file in os.listdir(class_dir):
            img_path = os.path.join(class_dir, img_file)

            # open grayscale 48x48 → convert to RGB for HF processor
            img = Image.open(img_path).convert("RGB")
            images.append(img)
            labels.append(class_idx)

    return images, labels


def load_train_test(dataset_root):
    train_root = os.path.join(dataset_root, "train")
    test_root  = os.path.join(dataset_root, "test")

    train_images, train_labels = load_split(train_root)
    test_images, test_labels   = load_split(test_root)

    return train_images, train_labels, test_images, test_labels
