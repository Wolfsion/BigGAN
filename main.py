import os
import torch.nn as nn
from torch import optim
from torch.nn.utils import spectral_norm
import torch.nn.functional as F
import torch
import torchvision
import numpy as np
from matplotlib import pyplot as plt
import torchvision.utils as vutils

from DataLoader import get_data_loader, VDataSet
from Discriminator import Discriminator


# 定义归一化函数
def batchnorm_2d(in_features, eps=1e-4, momentum=0.1, affine=True):
    return nn.BatchNorm2d(in_features, eps=eps, momentum=momentum, affine=affine, track_running_stats=True)


# sn全连接层(这样的目的是为了更加稳定)
def snlinear(in_features, out_features, bias=True):
    return spectral_norm(nn.Linear(in_features=in_features, out_features=out_features, bias=bias), eps=1e-6)


# sn卷积层(这样的目的是为了更加稳定)
def snconv2d(in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True):
    return spectral_norm(nn.Conv2d(in_channels=in_channels,
                                   out_channels=out_channels,
                                   kernel_size=kernel_size,
                                   stride=stride,
                                   padding=padding,
                                   dilation=dilation,
                                   groups=groups,
                                   bias=bias),
                         eps=1e-6)


# sn编码(这样的目的是为了更加稳定)
def sn_embedding(num_embeddings, embedding_dim):
    return spectral_norm(nn.Embedding(num_embeddings=num_embeddings, embedding_dim=embedding_dim), eps=1e-6)


# 归一化层？
class ConditionalBatchNorm2d(nn.Module):
    def __init__(self, in_features=148, out_features=384):
        super().__init__()
        self.in_features = in_features
        self.bn = batchnorm_2d(out_features, eps=1e-4, momentum=0.1, affine=False)

        self.gain = snlinear(in_features=in_features, out_features=out_features, bias=False)
        self.bias = snlinear(in_features=in_features, out_features=out_features, bias=False)

    def forward(self, x, y):
        gain = (1 + self.gain(y)).view(y.size(0), -1, 1, 1)
        bias = self.bias(y).view(y.size(0), -1, 1, 1)
        out = self.bn(x)
        return out * gain + bias


# 自我注意机制？
class SelfAttention(nn.Module):
    def __init__(self, in_channels, is_generator):
        super(SelfAttention, self).__init__()
        self.in_channels = in_channels

        if is_generator:
            self.conv1x1_theta = snconv2d(in_channels=in_channels, out_channels=in_channels // 8, kernel_size=1,
                                          stride=1, padding=0, bias=False)
            self.conv1x1_phi = snconv2d(in_channels=in_channels, out_channels=in_channels // 8, kernel_size=1,
                                        stride=1, padding=0, bias=False)
            self.conv1x1_g = snconv2d(in_channels=in_channels, out_channels=in_channels // 2, kernel_size=1,
                                      stride=1, padding=0, bias=False)
            self.conv1x1_attn = snconv2d(in_channels=in_channels // 2, out_channels=in_channels, kernel_size=1,
                                         stride=1, padding=0, bias=False)
        else:
            self.conv1x1_theta = snconv2d(in_channels=in_channels, out_channels=in_channels // 8, kernel_size=1,
                                          stride=1, padding=0, bias=False)
            self.conv1x1_phi = snconv2d(in_channels=in_channels, out_channels=in_channels // 8, kernel_size=1,
                                        stride=1, padding=0, bias=False)
            self.conv1x1_g = snconv2d(in_channels=in_channels, out_channels=in_channels // 2, kernel_size=1,
                                      stride=1, padding=0, bias=False)
            self.conv1x1_attn = snconv2d(in_channels=in_channels // 2, out_channels=in_channels, kernel_size=1,
                                         stride=1, padding=0, bias=False)

        self.maxpool = nn.MaxPool2d(2, stride=2, padding=0)
        self.softmax = nn.Softmax(dim=-1)
        self.sigma = nn.Parameter(torch.zeros(1), requires_grad=True)

    def forward(self, x):
        _, ch, h, w = x.size()
        # Theta path
        theta = self.conv1x1_theta(x)
        theta = theta.view(-1, ch // 8, h * w)
        # Phi path
        phi = self.conv1x1_phi(x)
        phi = self.maxpool(phi)
        phi = phi.view(-1, ch // 8, h * w // 4)
        # Attn map
        attn = torch.bmm(theta.permute(0, 2, 1), phi)
        attn = self.softmax(attn)
        # g path
        g = self.conv1x1_g(x)
        g = self.maxpool(g)
        g = g.view(-1, ch // 2, h * w // 4)
        # Attn_g
        attn_g = torch.bmm(g, attn.permute(0, 2, 1))
        attn_g = attn_g.view(-1, ch // 2, h, w)
        attn_g = self.conv1x1_attn(attn_g)
        return x + self.sigma * attn_g


# 一个块
class GenBlock(nn.Module):
    def __init__(self):
        super(GenBlock, self).__init__()
        in_features = 148
        self.bn1 = ConditionalBatchNorm2d(in_features=in_features)
        self.bn2 = ConditionalBatchNorm2d(in_features=in_features)
        self.activation = nn.ReLU(inplace=True)
        self.conv2d0 = snconv2d(384, 384, kernel_size=(1, 1), stride=(1, 1))
        self.conv2d1 = snconv2d(384, 384, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.conv2d2 = snconv2d(384, 384, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))

    def forward(self, x, affine):
        x0 = x
        x = self.bn1(x, affine)
        x = self.activation(x)
        x = F.interpolate(x, scale_factor=2, mode="nearest")
        x = self.conv2d1(x)

        x = self.bn2(x, affine)
        x = self.activation(x)
        x = self.conv2d2(x)

        x0 = F.interpolate(x0, scale_factor=2, mode="nearest")
        x0 = self.conv2d0(x0)
        out = x + x0
        return out


# 生成网络 输入inputs.shape = tensor.size[(batch_size, 80)]
class Generator(nn.Module):
    def __init__(self, classes: int):
        super(Generator, self).__init__()
        self.linear0 = snlinear(in_features=20, out_features=6144, bias=True)
        self.shared = nn.Embedding(classes, 128)
        # 主要块
        self.blocks = nn.ModuleList()
        self.blocks.append(nn.ModuleList([GenBlock()]))  # (0): ModuleList
        self.blocks.append(nn.ModuleList([GenBlock()]))  # (1): ModuleList
        self.blocks.append(nn.ModuleList([SelfAttention(in_channels=384, is_generator=True)]))  # (2): ModuleList
        self.blocks.append(nn.ModuleList([GenBlock()]))  # (3): ModuleList

        self.bn4 = batchnorm_2d(in_features=384)
        self.activation = nn.ReLU(inplace=True)
        self.conv2d5 = snconv2d(384, 3, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.tanh = nn.Tanh()

        # 关于cifar10的默认参数
        self.bottom = 4

    def forward(self, z, label=None):
        affine_list = []
        z0 = z
        zs = torch.split(z, 20, 1)
        z = zs[0]

        shared_label = self.shared(label)
        affine_list.append(shared_label)
        affines = [torch.cat(affine_list + [item], 1) for item in zs[1:]]

        act = self.linear0(z)
        act = act.view(-1, 384, self.bottom, self.bottom)
        counter = 0
        for index, blocklist in enumerate(self.blocks):
            for block in blocklist:
                if isinstance(block, SelfAttention):
                    act = block(act)
                else:
                    act = block(act, affines[counter])
                    counter += 1

        act = self.bn4(act)
        act = self.activation(act)
        act = self.conv2d5(act)
        out = self.tanh(act)
        return out


def imshow(imgs):
    imgs = imgs / 2 + 0.5  # 逆归一化，像素值从[-1, 1]回到[0, 1]
    imgs = imgs.cpu().detach().numpy().transpose((1, 2, 0))  # 图像从(C, H, W)转回(H, W, C)的numpy矩阵
    plt.axis("off")
    plt.imshow(imgs)


# custom weights initialization called on netG and netD
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname == 'BatchNorm2d':
        if m.affine:
            m.weight.data.normal_(1.0, 0.02)
            m.bias.data.fill_(0)


def train_model(dataloader, classes: int):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    netG = Generator(classes).cuda(0)
    netG.apply(weights_init)
    # load weights to test the model
    # netG.load_state_dict(torch.load('weights/netG_epoch_24.pth'))
    print(netG)

    netD = Discriminator(n_classes=classes).cuda(0)
    netD.apply(weights_init)
    # load weights to test the model
    # netD.load_state_dict(torch.load('weights/netD_epoch_24.pth'))
    print(netD)

    criterion = nn.BCELoss()

    # setup optimizer
    optimizerD = optim.Adam(netD.parameters(), lr=2e-4, betas=(0.5, 0.999))
    optimizerG = optim.Adam(netG.parameters(), lr=2e-4, betas=(0.5, 0.999))

    real_label = 1
    fake_label = 0

    niter = 100
    g_loss = []
    d_loss = []

    for epoch in range(niter):
        for i, data in enumerate(dataloader, 0):
            ############################
            # (1) Update D network: maximize log(D(x)) + log(1 - D(G(z)))
            ###########################
            # train with real
            netD.zero_grad()
            real_cpu = data[0].cuda(0)
            real_target = data[1].cuda(0)
            batch_size = real_cpu.size(0)
            label = torch.full((batch_size,), real_label).float().cuda(0)

            output = netD(real_cpu, real_target)
            errD_real = criterion(output, label)

            errD_real.backward()
            D_x = output.mean().item()

            # train with fake
            fixed_noise = torch.tensor(np.random.RandomState(2022).randn(batch_size, 80)).to(torch.float32).cuda(0)
            label = torch.randint(low=0, high=classes, size=(batch_size,), dtype=torch.long).cuda(0)

            fake = netG(fixed_noise, label)
            label.fill_(fake_label).float().cuda(0)

            output = netD(fake.detach(), label)
            errD_fake = criterion(output, label)
            errD_fake.backward()

            D_G_z1 = output.mean().item()
            errD = errD_real + errD_fake
            optimizerD.step()

            ############################
            # (2) Update G network: maximize log(D(G(z)))
            ###########################
            netG.zero_grad()
            label.fill_(real_label).float()  # fake labels are real for generator cost
            output = netD(fake)
            errG = criterion(output, label)
            errG.backward()
            D_G_z2 = output.mean().item()
            optimizerG.step()

            print('[%d/%d][%d/%d] Loss_D: %.4f Loss_G: %.4f D(x): %.4f D(G(z)): %.4f / %.4f' % (
                epoch, niter, i, len(dataloader), errD.item(), errG.item(), D_x, D_G_z1, D_G_z2))

            # save the output
            if i % 100 == 0:
                print('saving the output')
                vutils.save_image(real_cpu, 'output/real_samples.png', normalize=True)
                fake = netG(fixed_noise)
                vutils.save_image(fake.detach(), 'output/fake_samples_epoch_%03d.png' % epoch, normalize=True)

        # Check pointing for every epoch

        if epoch % 5 == 0:
            torch.save(netG.state_dict(), 'weights/netG_epoch_%d.pth' % epoch)
            torch.save(netD.state_dict(), 'weights/netD_epoch_%d.pth' % epoch)


# 路径需要自己手动创建 to refine
def create_imgs(path: str, imgs_cnt: int, label_idx: int):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    batch = 128

    # 以下两行需要根据自己的模型以及模型参数路径进行修改
    G = Generator().cuda(0)
    G.load_state_dict(torch.load("./model.pth", map_location=device)["state_dict"])
    G.eval()

    for idx in range(imgs_cnt // batch):
        z = torch.tensor(np.random.RandomState(2022).randn(batch, 80)).to(torch.float32).cuda(0)
        label = torch.zeros(batch).type(torch.long).cuda(0) + label_idx
        img = G(z, label)

        for j in range(batch):
            img_path = os.path.join(path, f"{label_idx}-{idx}-{j}.png")
            vutils.save_image(img.detach()[j:j + 1], img_path, normalize=True)


if __name__ == '__main__':
    batch_size = 32
    num_classes = 100
    dataloader = get_data_loader(VDataSet.CIFAR100, data_type="train",
                                 batch_size=batch_size, shuffle=True)
    train_model(dataloader, classes=num_classes)

    # # 生成数据集
    # base_path = "/home/xd/la/datasets/CIFAR10-GAN"
    # num_classes = 10
    # cnt_imgs = 5120
    #
    # for i in range(num_classes):
    #     create_imgs(os.path.join(base_path, str(i)), cnt_imgs, i)
    #     print(f"{i} class create done.")
