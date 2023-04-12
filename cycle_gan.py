from pathlib import Path

import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
from tqdm import tqdm
from pydantic import BaseModel
import wandb


# [yunjey/mnist-svhn-transfer: PyTorch Implementation of CycleGAN and SSGAN for Domain Transfer (Minimal)]
# (https://github.com/yunjey/mnist-svhn-transfer)


class Settings(BaseModel):
    project_name: str = 'CycleGAN'
    device: str = 'cuda'
    epoch: int = 5000
    batch: int = 64  # 16
    learning_rate: float = 2e-4
    image_size: list[int] = [32, 32]
    sample_interval: int = 500

    dataset_path: str = "/home/hpds/yucheng/gan-pytorch/dataset"
    dataset_domain_a: str = 'draw'
    dataset_domain_b: str = 'game'

    beta1 = 0.5
    beta2 = 0.999

    real_label = 1.
    fake_label = 0.

    g_conv_dim = 64
    d_conv_dim = 64


def deconv(c_in, c_out, k_size, stride=2, pad=1, bn=True):
    """Custom deconvolutional layer for simplicity."""
    layers = []
    layers.append(nn.ConvTranspose2d(c_in, c_out, k_size, stride, pad, bias=False))
    if bn:
        layers.append(nn.BatchNorm2d(c_out))
    return nn.Sequential(*layers)


def conv(c_in, c_out, k_size, stride=2, pad=1, bn=True):
    """Custom convolutional layer for simplicity."""
    layers = []
    layers.append(nn.Conv2d(c_in, c_out, k_size, stride, pad, bias=False))
    if bn:
        layers.append(nn.BatchNorm2d(c_out))
    return nn.Sequential(*layers)


class G12(nn.Module):
    """Generator for transfering from mnist to svhn"""

    def __init__(self, conv_dim=64):
        super(G12, self).__init__()
        # encoding blocks
        self.conv1 = conv(3, conv_dim, 4)
        self.conv2 = conv(conv_dim, conv_dim*2, 4)

        # residual blocks
        self.conv3 = conv(conv_dim*2, conv_dim*2, 3, 1, 1)
        self.conv4 = conv(conv_dim*2, conv_dim*2, 3, 1, 1)

        # decoding blocks
        self.deconv1 = deconv(conv_dim*2, conv_dim, 4)
        self.deconv2 = deconv(conv_dim, 3, 4, bn=False)

    def forward(self, x):
        out = nn.functional.leaky_relu(self.conv1(x), 0.05)      # (?, 64, 16, 16)
        out = nn.functional.leaky_relu(self.conv2(out), 0.05)    # (?, 128, 8, 8)

        out = nn.functional.leaky_relu(self.conv3(out), 0.05)    # ( " )
        out = nn.functional.leaky_relu(self.conv4(out), 0.05)    # ( " )

        out = nn.functional.leaky_relu(self.deconv1(out), 0.05)  # (?, 64, 16, 16)
        out = torch.tanh(self.deconv2(out))              # (?, 3, 32, 32)
        return out


class G21(nn.Module):
    """Generator for transfering from svhn to mnist"""

    def __init__(self, conv_dim=64):
        super(G21, self).__init__()
        # encoding blocks
        self.conv1 = conv(3, conv_dim, 4)
        self.conv2 = conv(conv_dim, conv_dim*2, 4)

        # residual blocks
        self.conv3 = conv(conv_dim*2, conv_dim*2, 3, 1, 1)
        self.conv4 = conv(conv_dim*2, conv_dim*2, 3, 1, 1)

        # decoding blocks
        self.deconv1 = deconv(conv_dim*2, conv_dim, 4)
        self.deconv2 = deconv(conv_dim, 3, 4, bn=False)

    def forward(self, x):
        out = nn.functional.leaky_relu(self.conv1(x), 0.05)      # (?, 64, 16, 16)
        out = nn.functional.leaky_relu(self.conv2(out), 0.05)    # (?, 128, 8, 8)

        out = nn.functional.leaky_relu(self.conv3(out), 0.05)    # ( " )
        out = nn.functional.leaky_relu(self.conv4(out), 0.05)    # ( " )

        out = nn.functional.leaky_relu(self.deconv1(out), 0.05)  # (?, 64, 16, 16)
        out = torch.tanh(self.deconv2(out))              # (?, 1, 32, 32)
        return out


class D1(nn.Module):
    """Discriminator for mnist."""

    def __init__(self, conv_dim=64, use_labels=False):
        super(D1, self).__init__()
        self.conv1 = conv(3, conv_dim, 4, bn=False)
        self.conv2 = conv(conv_dim, conv_dim*2, 4)
        self.conv3 = conv(conv_dim*2, conv_dim*4, 4)
        n_out = 11 if use_labels else 1
        self.fc = conv(conv_dim*4, n_out, 4, 1, 0, False)

    def forward(self, x):
        out = nn.functional.leaky_relu(self.conv1(x), 0.05)    # (?, 64, 16, 16)
        out = nn.functional.leaky_relu(self.conv2(out), 0.05)  # (?, 128, 8, 8)
        out = nn.functional.leaky_relu(self.conv3(out), 0.05)  # (?, 256, 4, 4)
        out = self.fc(out).squeeze()
        return out


class D2(nn.Module):
    """Discriminator for svhn."""

    def __init__(self, conv_dim=64, use_labels=False):
        super(D2, self).__init__()
        self.conv1 = conv(3, conv_dim, 4, bn=False)
        self.conv2 = conv(conv_dim, conv_dim*2, 4)
        self.conv3 = conv(conv_dim*2, conv_dim*4, 4)
        n_out = 11 if use_labels else 1
        self.fc = conv(conv_dim*4, n_out, 4, 1, 0, False)

    def forward(self, x):
        out = nn.functional.leaky_relu(self.conv1(x), 0.05)    # (?, 64, 16, 16)
        out = nn.functional.leaky_relu(self.conv2(out), 0.05)  # (?, 128, 8, 8)
        out = nn.functional.leaky_relu(self.conv3(out), 0.05)  # (?, 256, 4, 4)
        out = self.fc(out).squeeze()
        return out


class MyDataset(Dataset):
    def __init__(self, **kwargs) -> None:
        self.args = kwargs
        self.dataset_path = Path(self.args['dataset_path'])
        self.normalize_mean = 0.5
        self.normalize_std = 0.5
        self.transforms = transforms.Compose([
            transforms.Resize(self.args['image_size']),
            transforms.CenterCrop(self.args['image_size']),
            transforms.ToTensor(),  # [0, 255] => [0, 1]
            self.normalize()  # [0, 1] => [-1, 1]
        ])
        self.anime_path = self.read_dataset(self.args['dataset_domain_a'])
        self.face_path = self.read_dataset(self.args['dataset_domain_b'])

        self.anime_image = self.open_image(self.anime_path)
        self.face_image = self.open_image(self.face_path)

        self.length = len(self.anime_path)

    def normalize(self):
        mean = self.normalize_mean
        std = self.normalize_std
        return transforms.Normalize(mean=[mean, mean, mean], std=[std, std, std])

    def de_normalize(self, tensor):
        mean = -self.normalize_mean / self.normalize_std
        std = 1 / self.normalize_std
        t = transforms.Normalize(mean=[mean, mean, mean], std=[std, std, std])
        return t(tensor)

    def read_dataset(self, dir):
        image_path = []
        target_path = self.dataset_path.joinpath(dir)
        for path in target_path.glob("**/*"):
            if path.suffix in ['.png', '.jpg']:
                image_path += [path]
        return image_path

    def open_image(self, image_path):
        image = []
        for key, value in enumerate(image_path):
            image += [Image.open(value).copy()]
            image[-1] = image[-1].convert("RGB")
            if self.transforms is not None:
                image[-1] = self.transforms(image[-1]).numpy()
        return image

    def __getitem__(self, index):
        return self.anime_image[index], self.face_image[index]

    def __len__(self):
        return self.length


class CycleGAN():
    def __init__(self) -> None:
        self.args = Settings()
        wandb.init(project=self.args.project_name, config=self.args.dict(), save_code=True)

    def load_dataset(self):
        self.dataset = MyDataset(**self.args.dict())
        self.loader = DataLoader(dataset=self.dataset, batch_size=self.args.batch)

    def load_model(self):
        self.g12_model, self.g21_model, self.g_optim = self.generator()
        self.d1_model, self.d2_model, self.d_optim = self.discriminator()
        # self.criterion = nn.CrossEntropyLoss()

    def generator(self):
        g12 = G12(conv_dim=self.args.g_conv_dim).to(self.args.device)
        g21 = G21(conv_dim=self.args.g_conv_dim).to(self.args.device)
        g_params = list(g12.parameters()) + list(g21.parameters())
        g_optimizer = torch.optim.Adam(g_params, self.args.learning_rate, [self.args.beta1, self.args.beta2])
        return g12, g21, g_optimizer

    def discriminator(self):
        d1 = D1(conv_dim=self.args.d_conv_dim, use_labels=False).to(self.args.device)
        d2 = D2(conv_dim=self.args.d_conv_dim, use_labels=False).to(self.args.device)
        d_params = list(d1.parameters()) + list(d2.parameters())
        d_optimizer = torch.optim.Adam(d_params, self.args.learning_rate, [self.args.beta1, self.args.beta2])
        return d1, d2, d_optimizer

    def reset_grad(self):
        self.g_optim.zero_grad()
        self.d_optim.zero_grad()

    def train_one_epoch(self, i_epoch: int):
        self.metric = {
            self.train_d.__name__: {},
            self.train_g.__name__: {},
        }
        self.i_epoch = i_epoch
        self.g12_model.train(mode=True)
        self.g21_model.train(mode=True)
        self.d1_model.train(mode=True)
        self.d2_model.train(mode=True)
        bar = tqdm(self.loader, unit='batch', leave=True)
        for i_batch, (domain_a, domain_b) in enumerate(bar):
            self.step = (i_epoch * len(self.loader) + i_batch)

            domain_a = domain_a.to(self.args.device)
            domain_b = domain_b.to(self.args.device)

            self.train_d(domain_a, domain_b)

            fake_domain_b, reconst_domain_a, fake_domain_a, reconst_domain_b = self.train_g(domain_a, domain_b)

            loss = {
                'd_loss_1': self.metric[self.train_d.__name__]['loss_1'][-1],
                'd_loss_2': self.metric[self.train_d.__name__]['loss_2'][-1],
                'g_loss_1': self.metric[self.train_g.__name__]['loss_1'][-1],
                'g_loss_2': self.metric[self.train_g.__name__]['loss_2'][-1],
            }
            wandb.log({**loss, **{'i_epoch': self.i_epoch, 'step': self.step, }}, step=self.step)

            self.show_result(domain_a, fake_domain_b, reconst_domain_a, domain_b, fake_domain_a, reconst_domain_b)

            bar.set_description(f'Epoch [{self.i_epoch + 1}/{self.args.epoch}]')
            bar.set_postfix(**loss)

        d_loss_1 = sum(self.metric[self.train_d.__name__]['loss_1']) / len(self.loader)
        d_loss_2 = sum(self.metric[self.train_d.__name__]['loss_2']) / len(self.loader)
        g_loss_1 = sum(self.metric[self.train_g.__name__]['loss_1']) / len(self.loader)
        g_loss_2 = sum(self.metric[self.train_g.__name__]['loss_2']) / len(self.loader)
        print(f"d_loss_1: {d_loss_1}")
        print(f"d_loss_2: {d_loss_2}")
        print(f"g_loss_1: {g_loss_1}")
        print(f"g_loss_2: {g_loss_2}")
        return 0

    def show_result(self, domain_a, fake_domain_b, reconst_domain_a, domain_b, fake_domain_a, reconst_domain_b):
        if self.step % self.args.sample_interval == 0:
            wandb.log({
                'domain_a': [wandb.Image(im.permute(1,2,0).detach().cpu().numpy()) for im in domain_a],
                'fake_domain_b': [wandb.Image(im.permute(1,2,0).detach().cpu().numpy()) for im in fake_domain_b],
                'reconst_domain_a': [wandb.Image(im.permute(1,2,0).detach().cpu().numpy()) for im in reconst_domain_a],
                'domain_b': [wandb.Image(im.permute(1,2,0).detach().cpu().numpy()) for im in domain_b],
                'fake_domain_a': [wandb.Image(im.permute(1,2,0).detach().cpu().numpy()) for im in fake_domain_a],
                'reconst_domain_b': [wandb.Image(im.permute(1,2,0).detach().cpu().numpy()) for im in reconst_domain_b],
            }, step=self.step)
            pass
        return 0

    def train_d(self, domain_a, domain_b):
        # train with real images
        out = self.d1_model.forward(domain_a)  # draw is real
        d1_loss = torch.mean((out-1)**2)

        out = self.d2_model.forward(domain_b)  # game is real
        d2_loss = torch.mean((out-1)**2)

        d_domain_a_loss = d1_loss
        d_domain_b_loss = d2_loss
        d_real_loss = d1_loss + d2_loss

        self.reset_grad()
        torch.autograd.backward(d_real_loss)
        self.d_optim.step()

        # train with fake images
        fake_domain_b = self.g12_model.forward(domain_a)
        out = self.d2_model.forward(fake_domain_b)
        d2_loss = torch.mean(out**2)

        fake_domain_a = self.g21_model.forward(domain_b)
        out = self.d1_model.forward(fake_domain_a)
        d1_loss = torch.mean(out**2)

        d_fake_loss = d1_loss + d2_loss

        self.reset_grad()
        torch.autograd.backward(d_fake_loss)
        self.d_optim.step()

        self.metric[self.train_d.__name__].setdefault('loss_1', []).append(d_real_loss.item())
        self.metric[self.train_d.__name__].setdefault('loss_2', []).append(d_fake_loss.item())
        return 0

    def train_g(self, domain_a, domain_b):
        # train a-b-a cycle
        fake_domain_b = self.g12_model.forward(domain_a)
        out = self.d2_model.forward(fake_domain_b)
        reconst_domain_a = self.g21_model.forward(fake_domain_b)
        g_loss = torch.mean((out-1)**2)

        g_loss += torch.mean((domain_a - reconst_domain_a)**2)

        self.reset_grad()
        torch.autograd.backward(g_loss)
        self.g_optim.step()

        self.metric[self.train_g.__name__].setdefault('loss_1', []).append(g_loss.item())

        # train b-a-b cycle
        fake_domain_a = self.g21_model.forward(domain_b)
        out = self.d1_model.forward(fake_domain_a)
        reconst_domain_b = self.g12_model.forward(fake_domain_a)
        g_loss = torch.mean((out-1)**2)

        g_loss += torch.mean((domain_b - reconst_domain_b)**2)

        self.reset_grad()
        torch.autograd.backward(g_loss)
        self.g_optim.step()

        self.metric[self.train_g.__name__].setdefault('loss_2', []).append(g_loss.item())
        return fake_domain_b, reconst_domain_a, fake_domain_a, reconst_domain_b

    def train(self):
        wandb.watch(self.g12_model, criterion=None, log='gradients', log_freq=1000, idx=0)
        wandb.watch(self.g21_model, criterion=None, log='gradients', log_freq=1000, idx=1)
        wandb.watch(self.d1_model, criterion=None, log='gradients', log_freq=1000, idx=2)
        wandb.watch(self.d2_model, criterion=None, log='gradients', log_freq=1000, idx=3)
        for i_epoch in range(self.args.epoch):
            self.train_one_epoch(i_epoch)
            self.validation()
            self.save_model()
        return 0

    def validation(self):
        return 0

    def test(self):
        return 0

    def save_model(self):
        return 0


if __name__ == "__main__":
    deep_conv_gan = CycleGAN()
    deep_conv_gan.load_dataset()
    deep_conv_gan.load_model()
    deep_conv_gan.train()
