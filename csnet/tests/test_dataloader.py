from torch.utils.data import DataLoader

from ..dataset import CS_Dataset


def test_dataloader(data_paths):
    dir_img, dir_gt = data_paths
    ds = CS_Dataset(dir_img, dir_gt)
    dl = DataLoader(ds, batch_size=2, num_workers=2, shuffle=False)
    images, masks = next(iter(dl))
    assert len(images.shape) == 5
    assert len(masks.shape) == 5
