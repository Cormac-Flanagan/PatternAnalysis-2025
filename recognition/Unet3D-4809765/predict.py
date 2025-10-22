from modules import Unet3D
from utils import load_data_3D
from matplotlib import pyplot as plt
import torch
import numpy as np
import os
from tqdm import tqdm

def sliding_window_predict(model, volume, window_size=(128, 128, 128), stride=(128, 64, 64), num_classes=6):
    model.eval()
    device = next(model.parameters()).device
    _, H, W, D = volume.shape
    output = torch.zeros((num_classes, H, W, D), device=device)
    count_map = torch.zeros_like(output)

    for z in range(0, D - window_size[0] + 1, stride[0]):
        for y in tqdm(range(0, H - window_size[1] + 1, stride[1])):
            for x in tqdm(range(0, W - window_size[2] + 1, stride[2]), leave=False):
                patch = volume[:, x:x+window_size[2],
                                     y:y+window_size[1],
                                     z:z+window_size[0]]
                patch = torch.from_numpy(patch).unsqueeze(0).to(device)
                with torch.no_grad(), torch.autocast(device_type="cuda"):
                    pred = model(patch)              # [1, 6, Dz, Dy, Dx]
                output[:, x:x+window_size[2],
                                     y:y+window_size[1],
                                     z:z+window_size[0]] += pred[0]
                count_map[:, x:x+window_size[2],
                                     y:y+window_size[1],
                                     z:z+window_size[0]] += 1

    output /= count_map
    prediction = torch.argmax(output, dim=0)  # convert to label map

    return prediction.cpu().numpy()

def render_segments(segments, output_path, base):
    vmin = np.min(segments)
    vmax = np.max(segments)
    os.makedirs(output_path, exist_ok=True)
    for i in range(segments.shape[2]):
        plt.imshow(segments[:, :, i], cmap='Set2', vmin=vmin, vmax=vmax)
        plt.axis('off')
        plt.savefig(os.path.join(output_path, f"{base}_{i:03d}.png"))
        plt.close()


def predict(model_path: str, input_path, output_path: str, base = "layer"):
    model = Unet3D(num_classes=6)
    model.load_state_dict(torch.load(model_path, weights_only=True))
    model.eval()
    print("Loading Data")
    data = load_data_3D([input_path], normImage=True, categorical=False)

    print("Predicting...")
    segments = sliding_window_predict(model, data)
    print("Prediction done")
    render_segments(segments, output_path, base)

if __name__ == "__main__":
    predict("output/model.pth", "data/semantic_MRs/B006_Week0_LFOV.nii.gz", "output/segments")
