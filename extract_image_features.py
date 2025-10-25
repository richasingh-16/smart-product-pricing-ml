import os
import pandas as pd
import numpy as np
from tqdm import tqdm
from PIL import Image
import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
import joblib

# ============== CONFIG =================
DATA_DIR = './'
IMAGE_DIR = 'cache/images'
FEATURE_DIR = 'features'
BATCH_SIZE = 256       # lower this to 128 if you face memory issues
SAVE_INTERVAL = 10000  # save after every 10k images
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
# =======================================


def load_image(img_path):
    """Safely loads an image and applies transformations."""
    try:
        image = Image.open(img_path).convert('RGB')
        return transform(image).unsqueeze(0)
    except Exception:
        return None


def extract_features(model, img_paths):
    """Extract features for a list of image paths."""
    features = []
    model.eval()

    with torch.no_grad():
        for path in tqdm(img_paths, desc="Extracting image features"):
            tensor = load_image(path)
            if tensor is None:
                # If image failed to load, append a zero vector
                features.append(np.zeros(2048))
                continue

            tensor = tensor.to(DEVICE)
            output = model(tensor)
            features.append(output.cpu().numpy().squeeze())

    return np.array(features)


def process_and_save(df, mode='train'):
    """Process image set in batches and save progress."""
    all_features = []
    start_idx = 0

    # Check for resume
    save_path = os.path.join(FEATURE_DIR, f"{mode}_image_features_partial.pkl")
    if os.path.exists(save_path):
        print(f"🔁 Resuming from previous run for {mode}...")
        saved_data = joblib.load(save_path)
        all_features = saved_data['features']
        start_idx = saved_data['processed']
        print(f"✅ Resumed from index {start_idx}")

    img_paths = [
        os.path.join(IMAGE_DIR, os.path.basename(link))
        for link in df['image_link'].tolist()
    ]

    print(f"🚀 Starting {mode} image feature extraction from index {start_idx}...")
    remaining_paths = img_paths[start_idx:]

    for i in range(0, len(remaining_paths), SAVE_INTERVAL):
        batch_paths = remaining_paths[i:i + SAVE_INTERVAL]
        batch_features = extract_features(model, batch_paths)
        all_features.extend(batch_features)

        # Save progress after each chunk
        partial_data = {
            'features': all_features,
            'processed': start_idx + i + len(batch_paths)
        }
        joblib.dump(partial_data, save_path)
        print(f"💾 Saved progress after {start_idx + i + len(batch_paths)} images.")

    # Final save
    final_path = os.path.join(FEATURE_DIR, f"{mode}_image_features.pkl")
    joblib.dump(np.array(all_features), final_path)
    print(f"✅ Saved final features to {final_path}")
    os.remove(save_path) if os.path.exists(save_path) else None


# ============== MAIN =================
if __name__ == "__main__":
    os.makedirs(FEATURE_DIR, exist_ok=True)

    print(f"🖥️ Using device: {DEVICE}")
    print("📦 Loading dataset...")

    train_df = pd.read_csv(os.path.join(DATA_DIR, "train.csv"))
    test_df = pd.read_csv(os.path.join(DATA_DIR, "test.csv"))

    # Define transforms (resize + normalize for ResNet)
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])

    # Load pretrained ResNet50 and remove final classification layer
    print("⚙️ Loading pretrained ResNet50 model...")
    base_model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
    model = nn.Sequential(*list(base_model.children())[:-1])
    model.to(DEVICE)

    # Extract features for train and test sets
    process_and_save(train_df, mode='train')
    process_and_save(test_df, mode='test')

    print("🎉 Image feature extraction complete for both train and test sets.")
