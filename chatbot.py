import torch
import torch.nn as nn
from torchvision import transforms
import cv2
import json
import os
import random
import matplotlib.pyplot as plt
import numpy as np
import time

# --- CONFIGURATION ---
IMAGE_DIR = r"C:\Users\Khushan\Desktop\segment\dataset\train_images_512" 
MODEL_PATH = "vqa_model_final.pth"
ANSWERS_FILE = "vqa_answers.json"
DATASET_FILE = "vqa_dataset.json" # Needed to rebuild vocab if json missing
# ---------------------

# --- 1. DEFINE ARCHITECTURE (Must match training) ---
class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DoubleConv, self).__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )
    def forward(self, x): return self.double_conv(x)

class UNet(nn.Module):
    def __init__(self, n_classes, in_channels=3, features=[32, 64, 128, 256]):
        super(UNet, self).__init__()
        self.ups = nn.ModuleList()
        self.downs = nn.ModuleList()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        for feature in features:
            self.downs.append(DoubleConv(in_channels, feature))
            in_channels = feature
        for feature in reversed(features):
            self.ups.append(nn.ConvTranspose2d(feature*2, feature, kernel_size=2, stride=2))
            self.ups.append(DoubleConv(feature*2, feature))
        self.bottleneck = DoubleConv(features[-1], features[-1]*2)
        self.final_conv = nn.Conv2d(features[0], n_classes, kernel_size=1)

    def forward(self, x):
        skip_connections = []
        for down in self.downs:
            x = down(x)
            skip_connections.append(x)
            x = self.pool(x)
        x = self.bottleneck(x)
        return None, x # Return bottleneck features

class SatelliteVQA(nn.Module):
    def __init__(self, vocab_size, num_classes=7):
        super(SatelliteVQA, self).__init__()
        self.unet = UNet(n_classes=num_classes)
        self.q_emb = nn.Embedding(vocab_size, 64)
        self.q_rnn = nn.LSTM(64, 128, batch_first=True)
        self.fusion = nn.Sequential(
            nn.Linear(512 + 128, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 20) 
        )

    def forward(self, images, question_indices):
        _, features = self.unet(images) 
        features = torch.mean(features, dim=[2, 3]) 
        emb = self.q_emb(question_indices)
        _, (h_n, _) = self.q_rnn(emb)
        q_feat = h_n[-1] 
        combined = torch.cat((features, q_feat), dim=1)
        return self.fusion(combined)

# --- 2. HELPER FUNCTIONS ---
def get_vocab_and_answers():
    # Load Answers Map
    with open(ANSWERS_FILE, "r") as f:
        idx_to_answer = json.load(f)
        idx_to_answer = {int(k): v for k, v in idx_to_answer.items()}

    # Rebuild Vocab from dataset
    with open(DATASET_FILE, "r") as f:
        data = json.load(f)
    words = set()
    for item in data:
        for w in item['question'].lower().replace("?", "").split():
            words.add(w)
    vocab = {w: i+1 for i, w in enumerate(words)}
    return vocab, idx_to_answer

def prepare_question(text, vocab, device):
    indices = [vocab.get(w, 0) for w in text.lower().replace("?", "").split()]
    max_len = 10
    if len(indices) < max_len: indices += [0] * (max_len - len(indices))
    else: indices = indices[:max_len]
    return torch.tensor(indices, dtype=torch.long).unsqueeze(0).to(device)

def prepare_image(path, device):
    img = cv2.imread(path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
    ])
    return transform(img).unsqueeze(0).to(device), img

# --- 3. CHATBOT LOOP ---
def start_chat():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Initializing Satellite Bot on {device}...")

    # Load Resources
    vocab, answers = get_vocab_and_answers()
    model = SatelliteVQA(vocab_size=len(vocab)+1).to(device)
    
    if os.path.exists(MODEL_PATH):
        model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
        model.eval()
        print("✅ Model loaded! I am ready.")
    else:
        print("❌ Model file not found.")
        return

    # Enable interactive mode for plots so they don't block the chat
    plt.ion() 
    
    all_images = [f for f in os.listdir(IMAGE_DIR) if f.endswith(('.png', '.jpg'))]
    
    while True:
        # 1. Pick a new image context
        img_name = random.choice(all_images)
        img_path = os.path.join(IMAGE_DIR, img_name)
        img_tensor, display_img = prepare_image(img_path, device)
        
        # Show image
        plt.close('all') # Close previous
        plt.figure(figsize=(5, 5))
        plt.imshow(display_img)
        plt.title(f"Looking at: {img_name}")
        plt.axis('off')
        plt.show()
        plt.pause(0.1) # Give it a moment to render

        print(f"\n[SYSTEM] I am looking at image: {img_name}")
        print("[SYSTEM] Ask me anything! (Type 'next' for new image, 'exit' to quit)")

        # 2. Question Loop for this specific image
        while True:
            user_input = input("\n👤 You: ")
            
            if user_input.lower() in ['exit', 'quit']:
                print("👋 Goodbye!")
                return
            
            if user_input.lower() == 'next':
                print("🔄 Loading next image...")
                break # Breaks inner loop, goes back to picking new image
            
            # Prediction
            try:
                q_tensor = prepare_question(user_input, vocab, device)
                with torch.no_grad():
                    output = model(img_tensor, q_tensor)
                    pred_idx = torch.argmax(output, dim=1).item()
                
                answer = answers.get(pred_idx, "Unknown")
                print(f"🤖 Bot: {answer}")
            except Exception as e:
                print(f"❌ Error: {e}")

if __name__ == "__main__":
    start_chat()