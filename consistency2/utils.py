import numpy as np
import os
import torch
import PIL.Image as Image

def save_torch_img(tensor, path):
    #img has to be [H, W, (1 or 3 or 4)]
    tensor_np = tensor.detach().cpu().numpy()
    tensor_np = (tensor_np * 255).astype("uint8")
    Image.fromarray(tensor_np).save(path)

def load_torch_img(path, dtype=torch.float32, device= "cuda"):
    tensor_np = np.array(Image.open(path)).astype("float32")
    tensor = torch.tensor(tensor_np/255, dtype = dtype, device = device)
    return tensor

def create_dir(dir_name):
    os.makedirs(dir_name, exist_ok=True)
    return dir_name


def import_config_key(config, key, default=""):
    return config.get(key, default)


def exponential_decay_list(init_weight, decay_rate, num_steps):
    weights = [init_weight * (decay_rate ** i) for i in range(num_steps)]
    return torch.tensor(weights)
