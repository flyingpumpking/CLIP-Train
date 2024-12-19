import clip
from clip import model
import torch
import json
from PIL import Image
from torch import nn, optim

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model, preprocess = clip.load(name="ViT-B/16", device=device, jit=False)

checkpoint_path = r"D:\Projects\caches\clip\clip.pt"
checkpoint = torch.load(checkpoint_path, map_location=device)

model.load_state_dict(checkpoint['model_state_dict'])
model.eval()
print(model)
print("Model Loaded")
