import cv2
import sys
import numpy as np
from argparse import Namespace


import torch
from torchvision import transforms

from models import initialize_model
from utils import set_device, colorize, load_params, get_data_transform, DEFAULT_PARAMS_PATH


def inference(file_path: str, model_path: str, params: Namespace):
    # prepare model
    device = set_device(params.cuda)
    transform = get_data_transform("eval", params.image_resize, params.mean, params.std)
    model, _ = initialize_model(params, device)
    model.load_state_dict(torch.load(model_path))
    model.eval()

    image = cv2.imread(file_path)
    hight, width, _ = image.shape
    resize = transforms.Compose([transforms.Resize((hight, width))])
    transformed_image = transform(image=image)['image'].to(device)

    with torch.set_grad_enabled(False):
        output = model(transformed_image.unsqueeze(0))

    output = output.argmax(1).cpu()
    resized_image = resize(output)
    return resized_image.squeeze(0).numpy()


if __name__ == "__main__":
    params_path = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_PARAMS_PATH
    config = Namespace(**load_params(filepath=params_path))
    file_path = config.image_path
    model_path = config.model_path

    image = cv2.imread(file_path)
    output = inference(file_path, model_path, config)
    color_image = colorize(output).permute(1, 2, 0).numpy()

    print(color_image.shape)
    cv2.imwrite('output.png', color_image)
