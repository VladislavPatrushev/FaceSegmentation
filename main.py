import sys
from argparse import Namespace

import torch
import torch.optim as optim
import segmentation_models_pytorch as smp

from train import train
from models import initialize_model
from data import create_df, create_splited_dataset
from utils import set_seed, set_device, load_params, DEFAULT_PARAMS_PATH

if __name__ == "__main__":
    torch.cuda.empty_cache()
    set_seed()

    params_path = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_PARAMS_PATH
    config = Namespace(**load_params(filepath=params_path))

    df = create_df(config.path_to_images, config.path_to_masks)
    ds_train, ds_val, ds_test = create_splited_dataset(df, config)

    device = set_device(True)

    model, model_name = initialize_model(config, device)

    criterion = smp.losses.DiceLoss(mode='multiclass')
    optimizer = optim.Adam(model.parameters(), config.lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min')

    history = train(ds_train, ds_val,
                    model=model,
                    epochs=config.epochs,
                    batch_size=config.batch_size,
                    opt=optimizer,
                    criterion=criterion,
                    scheduler=scheduler,
                    model_name=model_name,
                    device=device)
