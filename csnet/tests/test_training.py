from torch import optim
from torch.utils.data import DataLoader

from ..dataset import CS_Dataset
from ..losses import WeightedCrossEntropyLoss, DiceLoss
from ..model import CSNet3D


def test_training(data_paths):
    dir_img, dir_gt = data_paths
    ds = CS_Dataset(dir_img, dir_gt)
    dl = DataLoader(ds, batch_size=2, num_workers=2, shuffle=False)
    net = CSNet3D(classes=2, channels=1).cuda()
    optimizer = optim.Adam(net.parameters(), lr=0.0001, weight_decay=0.0005)

    wce_loss = WeightedCrossEntropyLoss().cuda()
    dice_loss = DiceLoss().cuda()

    for epoch in range(1):
        net.train()
        for idx, batch in enumerate(dl):
            image = batch[0].cuda()
            label = batch[1].cuda()
            optimizer.zero_grad()
            pred = net(image)
            loss = (0.6 * wce_loss(pred, label.squeeze(1)) + 0.4 * dice_loss(pred, label))
            loss.backward()
            optimizer.step()
