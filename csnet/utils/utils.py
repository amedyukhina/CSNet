import datetime
import os

import torch
import wandb


def get_paired_file_list(dir1, dir2):
    files = os.listdir(dir1)
    files.sort()
    files1 = [os.path.join(dir1, fn) for fn in files
              if fn in os.listdir(dir2)]
    files2 = [os.path.join(dir2, fn) for fn in files
              if fn in os.listdir(dir2)]
    return files1, files2


def get_data_dict(dir1, dir2, image_key='image', label_key='label'):
    files = [{image_key: image_name, label_key: label_name}
             for image_name, label_name in zip(*get_paired_file_list(dir1, dir2))
             ]
    return files


def get_device():
    return torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')


def save_model(model, model_path, model_name='best_model.pth'):
    fn_out = os.path.join(model_path, model_name)
    os.makedirs(model_path, exist_ok=True)
    torch.save(model.state_dict(), fn_out)
    print(rf"Saved new best model to: {fn_out}")


def get_model_name(log_progress):
    if log_progress:
        model_name = wandb.run.name
    else:
        model_name = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    return model_name
