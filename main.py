from preprocessing import load_train_test
from inference import EmotionModel
from tuned_model_inference import EmotionModelTuned
import os

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_ROOT = os.path.join(CURRENT_DIR, "FER-2013")

# pipeline that loads images, loads the model, and evaluates the model accuracy
def main():
    print(f"Loading images from {DATASET_ROOT}...\n")
    train_images, train_labels, test_images, test_labels = load_train_test(DATASET_ROOT)

    print(f"# of training samples: {len(train_images)}")
    print(f"# of testing samples:  {len(test_images)}\n")

    print("Loading model...\n")
    model = EmotionModel()
    # tuned_model = EmotionModelTuned()

    print("Evaluating on testing set")
    acc = model.evaluate(test_images, test_labels)
    # acc2 = tuned_model.evaluate(test_images, test_labels)

    print(f"\nBase Model Accuracy: {acc}")
    # print(f"\nTuned Model Accuracy: {acc2}")


# runs main function
if __name__ == "__main__":
    main()