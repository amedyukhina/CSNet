from monai.data import DataLoader, Dataset

from csnet.transforms.default import get_default_train_transforms
from csnet.utils import get_data_dict


def test_dataloader(data_paths):
    dir_img, dir_gt = data_paths
    train_files = get_data_dict(dir_img, dir_gt)
    assert len(train_files) == 10
    assert 'image' in train_files[0].keys()
    assert 'label' in train_files[0].keys()
    train_transforms = get_default_train_transforms(roi_size=(8, 16, 16), return_val=False)
    ds = Dataset(data=train_files, transform=train_transforms)
    assert len(ds) == 10
    dl = DataLoader(ds, batch_size=2)
    batch = next(iter(dl))
    assert len(batch['image'].shape) == 5
    assert len(batch['label'].shape) == 5
