import os.path

import tensorboardX
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from PIL import Image
import torch
from model import Generator
from dataset import ImageDataset
import itertools
from torchvision.utils import save_image


def main():
    # 判断输出路径是否存在
    if not os.path.exists("outputs/A"):
        os.mkdir("outputs/A")
    if not os.path.exists("outputs/B"):
        os.mkdir("outputs/B")
    
    size = 256
    selected_epoch = 7

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    netG_A2B = Generator().to(device)
    netG_B2A = Generator().to(device)

    netG_A2B.load_state_dict(torch.load('./models/netG_A2B_epoch{}.pth'.format(selected_epoch)))
    netG_B2A.load_state_dict(torch.load('./models/netG_B2A_epoch{}.pth'.format(selected_epoch)))

    netG_A2B.eval()
    netG_B2A.eval()

    input_A = torch.ones([1, 3, size, size], dtype=torch.float).to(device)
    input_B = torch.ones([1, 3, size, size], dtype=torch.float).to(device)

    transforms_ = [
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ]

    data_root = 'apple2orange'
    loader = DataLoader(ImageDataset(data_root, transforms_, "test"), batch_size=1, shuffle=False, num_workers=10)

    for i, batch in enumerate(loader):
        real_A = torch.tensor(input_A.copy_(batch['A']), dtype=torch.float).to(device)
        real_B = torch.tensor(input_B.copy_(batch['B']), dtype=torch.float).to(device)

        fake_B = 0.5 * (netG_A2B(real_A).data + 1.0)
        fake_A = 0.5 * (netG_B2A(real_B).data + 1.0)

        save_image(fake_A, 'outputs/A/{}.png'.format(i))
        save_image(fake_B, 'outputs/B/{}.png'.format(i))


if __name__ == "__main__":
    main()

