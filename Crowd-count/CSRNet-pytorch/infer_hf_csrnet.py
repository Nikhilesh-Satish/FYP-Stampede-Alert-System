import torch
import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
from model import CSRNet   # This comes from CSRNet-pytorch repo

# ----------------------------
# IMPORTANT: Set your weights path here
# ----------------------------
WEIGHTS_PATH = r"C:\Users\praagna\.cache\huggingface\hub\models--rootstrap-org--crowd-counting\snapshots\1c8d69ebfdc4380baaaade2fb0b0f006f8bc04c2\weights.pth"

# Image you want to test (put test.jpg inside Crowd-count folder)
IMG_PATH = r"..\test.jpg"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def load_image(img_path):
    img = cv2.imread(img_path)
    if img is None:
        raise FileNotFoundError(f"Image not found: {img_path}")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    return img

def preprocess(img):
    img_t = np.transpose(img, (2,0,1)).astype(np.float32)
    img_t = torch.from_numpy(img_t).unsqueeze(0)
    return img_t

def load_weights(model, path, device):
    ckpt = torch.load(path, map_location=device)

    # if checkpoint has 'state_dict'
    if isinstance(ckpt, dict) and "state_dict" in ckpt:
        ckpt = ckpt["state_dict"]

    # remove "module." prefix if present
    new_state = {}
    for k,v in ckpt.items():
        new_key = k.replace("module.", "") if k.startswith("module.") else k
        new_state[new_key] = v

    model.load_state_dict(new_state, strict=False)
    return model

def inference(model, img_t, device):
    model = model.to(device)
    model.eval()
    img_t = img_t.to(device)
    with torch.no_grad():
        den = model(img_t)
        den_up = torch.nn.functional.interpolate(
            den,
            size=(img_t.shape[2], img_t.shape[3]),
            mode='bilinear',
            align_corners=False
        )
        count = den_up.sum().item()
    return den_up.squeeze().cpu().numpy(), count


if __name__ == "__main__":
    print("Loading model...")
    model = CSRNet()
    model = load_weights(model, WEIGHTS_PATH, DEVICE)
    print("Model loaded!")

    print("Loading image...")
    img = load_image(IMG_PATH)

    print("Running inference...")
    img_t = preprocess(img)
    den_map, pred_count = inference(model, img_t, DEVICE)

    print(f"\nPredicted crowd count = {pred_count:.2f}\n")

    # save results
    os.makedirs("outputs", exist_ok=True)
    plt.imsave("outputs/density_map.png", den_map, cmap="jet")

    den_norm = (den_map - den_map.min()) / (den_map.max() - den_map.min() + 1e-8)
    heat = cv2.resize(den_norm, (img.shape[1], img.shape[0]))
    heatmap = (plt.cm.jet(heat)[:,:,:3] * 255).astype(np.uint8)
    overlay = (0.6 * heatmap + 0.4 * (img * 255).astype(np.uint8)).astype(np.uint8)

    cv2.imwrite("outputs/overlay.png", cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))

    print("Saved density_map.png and overlay.png in outputs folder.")
