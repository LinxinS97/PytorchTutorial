import tensorboardX
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from PIL import Image
import torch
from model import Discriminator, Generator
from utils import ReplayBuffer, LambdaLR, weights_init_normal
from dataset import ImageDataset
import itertools

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# HYPER PARAMETER
batch_size = 1
size = 256
lr = 0.0002
n_epoch = 200
decay_epoch = 100


def main():
    epoch = 0
    # MODEL
    netG_A2B = Generator().to(device)
    netG_B2A = Generator().to(device)
    netD_A = Discriminator().to(device)
    netD_B = Discriminator().to(device)


    # LOSS
    loss_GAN = torch.nn.MSELoss()
    loss_cycle = torch.nn.L1Loss()
    loss_identity = torch.nn.L1Loss()

    # OPTIMIZER and LR
    opt_G = torch.optim.Adam(
        itertools.chain(netG_A2B.parameters(), netG_B2A.parameters()),
        lr=lr,
        betas=(0.5, 0.999))
    opt_DA = torch.optim.Adam(netD_A.parameters(), lr=lr, betas=(0.5, 0.999))
    opt_DB = torch.optim.Adam(netD_B.parameters(), lr=lr, betas=(0.5, 0.999))

    # 学习率衰减
    lr_scheduler_G = torch.optim.lr_scheduler.LambdaLR(opt_G, lr_lambda=LambdaLR(n_epoch, epoch, decay_epoch).step)
    lr_scheduler_DA = torch.optim.lr_scheduler.LambdaLR(opt_DA, lr_lambda=LambdaLR(n_epoch, epoch, decay_epoch).step)
    lr_scheduler_DB = torch.optim.lr_scheduler.LambdaLR(opt_DB, lr_lambda=LambdaLR(n_epoch, epoch, decay_epoch).step)

    # 输入、标签、buffer、tensorboardX
    data_root = "apple2orange"
    input_A = torch.ones([1, 3, size, size], dtype=torch.float).to(device)
    input_B = torch.ones([1, 3, size, size], dtype=torch.float).to(device)
    label_real = torch.ones([1], requires_grad=False, dtype=torch.float).to(device)
    label_fake = torch.zeros([1], requires_grad=False, dtype=torch.float).to(device)
    fake_A_buffer = ReplayBuffer()
    fake_B_buffer = ReplayBuffer()
    log_path = "logs"
    writer_log = tensorboardX.SummaryWriter(log_path)
    transforms_ = [
        transforms.Resize(int(256 * 1.12), Image.BICUBIC),
        transforms.RandomCrop(256),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ]

    # DataLoader
    loader = DataLoader(ImageDataset(data_root, transforms_), batch_size=batch_size, shuffle=True, num_workers=10)


    # TRAIN
    step = 0
    for epoch in range(n_epoch):
        print("epoch:{}".format(epoch+1))
        for i, batch in enumerate(loader):
            real_A = torch.tensor(input_A.copy_(batch['A']), dtype=torch.float).to(device)
            real_B = torch.tensor(input_B.copy_(batch['B']), dtype=torch.float).to(device)

            # #################Generator#################
            # 用AB生成对应AB，优化loss
            opt_G.zero_grad()
            same_B = netG_A2B(real_A)  # 根据A生成B的Generator
            loss_identity_B = loss_identity(same_B, real_B) * 5.0  # A-->B and 真实B 的 L1 loss

            same_A = netG_B2A(real_B)  # 根据B生成A的Generator
            loss_identity_A = loss_identity(same_A, real_A) * 5.0  # B-->A and 真实A 的 L1 loss

            # 用真A生成假B，真B生成假A，并且用判别器判断
            fake_B = netG_A2B(real_A)
            pred_fake = netD_B(fake_B)
            loss_GAN_A2B = loss_GAN(pred_fake, label_real)  # 根据A真标签生成假B，把假B丢给B的判别器，并优化他与真标签之间的MSE

            fake_A = netG_B2A(real_B)
            pred_fake = netD_A(fake_A)
            loss_GAN_B2A = loss_GAN(pred_fake, label_real)  # 根据B真标签生成假A，把假A丢给A的判别器，并优化他与真标签之间的MSE

            # cycle loss: 优化从假B还原成A的loss以及假A还原成B的loss
            recovered_A = netG_B2A(fake_B)
            loss_cycle_ABA = loss_cycle(recovered_A, real_A) * 10.0
            recovered_B = netG_B2A(fake_A)
            loss_cycle_BAB = loss_cycle(recovered_B, real_B) * 10.0

            # 生成器整体loss
            loss_G = loss_identity_A + loss_identity_B + \
                     loss_GAN_A2B + loss_GAN_B2A + loss_cycle_ABA + loss_cycle_BAB

            loss_G.backward()
            opt_G.step()

            # #################Discriminator#################
            opt_DA.zero_grad()
            pred_real = netD_A(real_A)  # 判别器对于真实结果的预测值
            loss_D_real = loss_GAN(pred_real, label_real)  # 优化判别器对于真实结果的预测与判别器对于生成结果的预测之间的loss
            fake_A = fake_A_buffer.push_and_pop(fake_A)  # 抽取生成器生成的A
            pred_fake = netD_A(fake_A.detach())  # 将生成A扔给判别器，让判别器算出生成A的分数（.detach()对梯度进行截断）
            loss_D_fake = loss_GAN(pred_fake, label_fake)  # 计算生成A与预测值的loss
            # total loss
            loss_D_A = (loss_D_real + loss_D_fake) * 0.5
            loss_D_A.backward()
            opt_DA.step()

            # 针对B，下同
            opt_DB.zero_grad()
            pred_real = netD_B(real_B)
            loss_D_real = loss_GAN(pred_real, label_real)
            fake_B = fake_B_buffer.push_and_pop(fake_B)
            pred_fake = netD_B(fake_B.detach())
            loss_D_fake = loss_GAN(pred_fake, label_fake)
            # total loss
            loss_D_B = (loss_D_real + loss_D_fake) * 0.5
            loss_D_B.backward()
            opt_DB.step()

            print("loss_G:{}, loss_G_identity:{}, loss_G_GAN:{}, loss_G_cycle:{}, loss_D_A:{}, loss_D_B:{}".format(
                loss_G,
                loss_identity_A + loss_identity_B,
                loss_GAN_A2B + loss_GAN_B2A,
                loss_cycle_BAB + loss_cycle_ABA,
                loss_D_A,
                loss_D_B
            ))

            writer_log.add_scalar("loss_G", loss_G, global_step=step + 1)
            writer_log.add_scalar("loss_G_identity", loss_identity_A + loss_identity_B, global_step=step + 1)
            writer_log.add_scalar("loss_G_GAN", loss_GAN_A2B + loss_GAN_B2A, global_step=step + 1)
            writer_log.add_scalar("loss_G_cycle", loss_cycle_BAB + loss_cycle_ABA, global_step=step + 1)
            writer_log.add_scalar("loss_D_A", loss_D_A, global_step=step + 1)
            writer_log.add_scalar("loss_D_B", loss_D_B, global_step=step + 1)

            step += 1

        lr_scheduler_G.step()
        lr_scheduler_DA.step()
        lr_scheduler_DB.step()

        torch.save(netG_A2B.state_dict(), 'models/netG_A2B_epoch{}.pth'.format(epoch + 1))
        torch.save(netG_B2A.state_dict(), 'models/netG_B2A_epoch{}.pth'.format(epoch + 1))
        torch.save(netD_A.state_dict(), 'models/netD_A_epoch{}.pth'.format(epoch + 1))
        torch.save(netD_B.state_dict(), 'models/netD_B_epoch{}.pth'.format(epoch + 1))


if __name__ == "__main__":
    main()
