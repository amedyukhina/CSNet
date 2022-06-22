import torch
from skimage import io
from torch.utils.data import Dataset

from .utils import get_paired_file_list


class CS_Dataset(Dataset):
    def __init__(self, dir_img, dir_gt, transform=None):
        self.image_fns, self.mask_fns = get_paired_file_list(dir_img, dir_gt)
        self.transform = transform

    def __len__(self):
        return len(self.image_fns)

    def __getitem__(self, idx):
        image = io.imread(self.image_fns[idx])
        mask = io.imread(self.mask_fns[idx])

        image = image * 1. / image.max()
        mask = mask * 1. / mask.max()
        image = torch.from_numpy(image).unsqueeze(0)
        mask = torch.from_numpy(mask).unsqueeze(0)

        return image.to(torch.float), mask.to(torch.int64)

