from tqdm import tqdm

import torch
from torch.utils.data import DataLoader

from utils import mIOU


def fit_epoch(model, train_loader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    processed_data = 0
    sum_iou = 0
    num_iter = 0

    for inputs, labels in train_loader:
        inputs = inputs.to(device)
        labels = labels.to(device)
        optimizer.zero_grad()

        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * inputs.size(0)
        sum_iou += mIOU(labels.cpu(), outputs.cpu().detach())
        processed_data += inputs.size(0)
        num_iter += 1

    train_loss = running_loss / processed_data
    iou_mean = sum_iou / num_iter
    return train_loss, iou_mean


def eval_epoch(model, val_loader, criterion, device):
    model.eval()
    running_loss = 0.0
    processed_data = 0
    sum_iou = 0
    num_iter = 0

    for inputs, labels in val_loader:
        inputs = inputs.to(device)
        labels = labels.to(device)

        with torch.set_grad_enabled(False):
            outputs = model(inputs)
            loss = criterion(outputs, labels)

        running_loss += loss.item() * inputs.size(0)
        sum_iou += mIOU(labels.cpu(), outputs.cpu().detach())
        processed_data += inputs.size(0)
        num_iter += 1

    train_loss = running_loss / processed_data
    iou_mean = sum_iou / num_iter
    return train_loss, iou_mean


def train(ds_train,
          ds_val,
          model,
          epochs,
          batch_size,
          opt,
          criterion,
          scheduler,
          model_name,
          device
          ):

    train_loader = DataLoader(ds_train, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(ds_val, batch_size=batch_size, shuffle=False)

    history = []
    log_template = "\nEpoch {ep:03d} train_loss: {t_loss:0.4f} " \
                   "val_loss {v_loss:0.4f} train_iou {t_iou:0.4f} " \
                   "val_iou {v_iou:0.4f}"
    best_score = 0

    with tqdm(desc="epoch", total=epochs) as pbar_outer:
        for epoch in range(epochs):
            train_loss, train_mean_iou = fit_epoch(model,
                                                   train_loader,
                                                   criterion,
                                                   opt,
                                                   device)
            print("loss", train_loss)

            val_loss, val_mean_iou = eval_epoch(model,
                                                val_loader,
                                                criterion,
                                                device)
            scheduler.step(val_loss)

            history.append((train_loss, train_mean_iou, val_loss, val_mean_iou))

            if val_mean_iou > best_score:
                torch.save(model.state_dict(), './weights/' + model_name + '.pth')
                best_score = val_mean_iou

            pbar_outer.update(1)
            tqdm.write(log_template.format(ep=epoch + 1,
                                           t_loss=train_loss,
                                           v_loss=val_loss,
                                           t_iou=train_mean_iou,
                                           v_iou=val_mean_iou))

    return history
