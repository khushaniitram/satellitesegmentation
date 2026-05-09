import torch
import matplotlib.pyplot as plt
import cv2
import pandas as pd
import numpy as np
import os
import random

# --- CONFIGURATION ---
# (Make sure these match your training config!)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MODEL_PATH = "multiclass_unet.pth"
IMG_DIR = 'dataset/train_images_512/'
MASK_DIR = 'dataset/train_masks_512/'
CLASS_DICT_PATH = 'class_dict.csv'

# --- 1. SETUP COLORS ---
class_df = pd.read_csv(CLASS_DICT_PATH)
class_rgb_values = class_df[['r', 'g', 'b']].values.tolist()
n_classes = len(class_rgb_values)

# --- 2. DEFINE MODEL ARCHITECTURE ---
class DoubleConv(torch.nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.double_conv = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            torch.nn.BatchNorm2d(out_channels),
            torch.nn.ReLU(inplace=True),
            torch.nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            torch.nn.BatchNorm2d(out_channels),
            torch.nn.ReLU(inplace=True),
        )
    def forward(self, x): return self.double_conv(x)

class UNet(torch.nn.Module):
    def __init__(self, n_classes, in_channels=3, features=[32, 64, 128, 256]):
        super().__init__()
        self.ups = torch.nn.ModuleList()
        self.downs = torch.nn.ModuleList()
        self.pool = torch.nn.MaxPool2d(kernel_size=2, stride=2)
        for feature in features:
            self.downs.append(DoubleConv(in_channels, feature))
            in_channels = feature
        for feature in reversed(features):
            self.ups.append(torch.nn.ConvTranspose2d(feature * 2, feature, kernel_size=2, stride=2))
            self.ups.append(DoubleConv(feature * 2, feature))
        self.bottleneck = DoubleConv(features[-1], features[-1] * 2)
        self.final_conv = torch.nn.Conv2d(features[0], n_classes, kernel_size=1)
    def forward(self, x):
        skip_connections = []

        # 1. Encoder (Downsampling)
        for down in self.downs:
            x = down(x)
            skip_connections.append(x)
            x = self.pool(x)

        # 2. Bottleneck (The "Brain")
        x = self.bottleneck(x)
        
        # --- NEW: CAPTURE FEATURES FOR VQA ---
        # We save 'x' here because this is the deepest, most semantic representation
        bottleneck_features = x 
        # -------------------------------------

        # 3. Decoder (Upsampling)
        skip_connections = skip_connections[::-1]
        
        for idx in range(0, len(self.ups), 2):
            x = self.ups[idx](x)
            skip_connection = skip_connections[idx//2]
            
            if x.shape != skip_connection.shape:
                x = torch.nn.functional.interpolate(x, size=skip_connection.shape[2:])
                
            concat_skip = torch.cat((skip_connection, x), dim=1)
            x = self.ups[idx+1](concat_skip)

        # 4. Return BOTH: The mask (for segmentation) and features (for VQA)
        return self.final_conv(x), bottleneck_features

# --- 3. LOAD THE TRAINED WEIGHTS ---
model = UNet(n_classes=n_classes).to(DEVICE)

if os.path.exists(MODEL_PATH):
    print(f"Loading weights from {MODEL_PATH}...")
    model.load_state_dict(torch.load(MODEL_PATH))
    model.eval()
    print("✅ Model loaded! Ready to predict.")
else:
    print("⏳ Waiting for training to finish...")

# --- 4. PREDICTION FUNCTION ---
def predict_random_image():
    if not os.path.exists(MODEL_PATH): return

    # Pick random file
    files = [f for f in os.listdir(IMG_DIR) if f.endswith('.png')]
    if not files: return
    img_name = random.choice(files)
    
    # Load Image
    img_path = os.path.join(IMG_DIR, img_name)
    image = cv2.imread(img_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Preprocess
    input_tensor = image / 255.0
    input_tensor = np.transpose(input_tensor, (2, 0, 1)).astype(np.float32)
    input_tensor = torch.from_numpy(input_tensor).unsqueeze(0).to(DEVICE)
    
    # Predict
    with torch.no_grad():
        output = model(input_tensor)
        pred_idx = torch.argmax(output, dim=1).squeeze().cpu().numpy()
    
    # Colorize Prediction
    h, w = pred_idx.shape
    rgb_mask = np.zeros((h, w, 3), dtype=np.uint8)
    for i, color in enumerate(class_rgb_values):
        rgb_mask[pred_idx == i] = color
        
    # Load Ground Truth
    mask_path = os.path.join(MASK_DIR, img_name)
    gt_mask = cv2.imread(mask_path)
    gt_mask = cv2.cvtColor(gt_mask, cv2.COLOR_BGR2RGB)
    
    # Display
    fig, ax = plt.subplots(1, 3, figsize=(15, 5))
    
    # --- NEW: SHOW FILENAME AT THE TOP ---
    fig.suptitle(f"FILE: {img_name}", fontsize=16, fontweight='bold')
    
    ax[0].imshow(image)
    ax[0].set_title("Input Satellite Image")
    ax[0].axis('off')
    
    ax[1].imshow(gt_mask)
    ax[1].set_title("True Mask (Target)")
    ax[1].axis('off')
    
    ax[2].imshow(rgb_mask)
    ax[2].set_title("AI Prediction")
    ax[2].axis('off')
    
    plt.tight_layout()
    plt.show()

# Run the prediction check
predict_random_image()