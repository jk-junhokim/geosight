import torch
import numpy as np
from dreamsim import dreamsim
from PIL import Image
import cv2
import glob
import zipfile
import os

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model, preprocess = dreamsim(pretrained=True, device=device)

# TODO: 
zip_path = ""  # Path to .zip file
extract_path = ""  #

if not os.path.exists(extract_path): 
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(extract_path) 
else:
    pass
    
model, preprocess = dreamsim(pretrained=True, device=device)
extractor = model.extractor_list[0] # DreamSim - DINO

image_files = glob.glob(os.path.join(extract_path, "*.jpg"))

N = 5 
selected_images = image_files[:N]

for i, img_path in enumerate(selected_images):
    img = Image.open(img_path)
    img_tensor = preprocess(img).to(device)

    with torch.no_grad():
        attn_weights = extractor.model.get_last_selfattention(img_tensor)
        attn_weights = attn_weights.mean(dim=1)
        attn_map = attn_weights[0, 0, 1:]
        attn_map = attn_map.reshape(14,14).cpu().numpy()

        attn_map = (attn_map - attn_map.min()) / (attn_map.max() - attn_map.min())

        heatmap = cv2.resize(attn_map, (224,224))
        heatmap = cv2.applyColorMap(np.uint8(255*heatmap), cv2.COLORMAP_JET)

        img_resized = img.resize((224,224))
        img_np = np.array(img_resized)
        overlay = cv2.addWeighted(img_np, 0.6, heatmap, 0.4, 0)
        
        # TODO: Define Save Path
        cv2.imwrite(f".jpg", overlay) 