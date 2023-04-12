from pathlib import Path

import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
from tqdm import tqdm
from pydantic import BaseModel
import wandb


# [[Pytorch] 搭建 GAN 模型產生虛假的 MNIST 圖片 - Clay-Technology World]
# (https://clay-atlas.com/blog/2020/01/09/pytorch-chinese-tutorial-mnist-generator-discriminator-mnist/)


class Settings(BaseModel):
    project_name: str = 'DenseGAN'
    device: str = 'cuda'
    epoch: int = 2000
    batch: int = 8
    learning_rate: float = 2e-4
    image_size: int = 50
    sample_interval: int = 500

    dataset_path: str = "/home/hpds/yucheng/gan-pytorch/dataset/FYC"


class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.main = nn.Sequential(
            nn.Linear(2500, 1024),
            nn.LeakyReLU(0.2),
            nn.Linear(1024, 512),
            nn.LeakyReLU(0.2),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2),
            nn.Linear(256, 1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        return self.main(x)


class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.layer1 = nn.Linear(128, 256)
        self.layer2 = nn.LeakyReLU(negative_slope=0.2)
        self.layer3 = nn.BatchNorm1d(256, momentum=0.8)
        self.layer4 = nn.Linear(256, 512)
        self.layer5 = nn.LeakyReLU(negative_slope=0.2)
        self.layer6 = nn.BatchNorm1d(512, momentum=0.8)
        self.layer7 = nn.Linear(512, 1024)
        self.layer8 = nn.LeakyReLU(negative_slope=0.2)
        self.layer9 = nn.BatchNorm1d(1024, momentum=0.8)
        self.layer10 = nn.Linear(1024, 2500)
        self.layer11 = nn.Tanh()

    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.layer5(out)
        out = self.layer6(out)
        out = self.layer7(out)
        out = self.layer8(out)
        out = self.layer9(out)
        out = self.layer10(out)
        out = self.layer11(out)
        return out


class MyDataset(Dataset):
    def __init__(self, **kwargs) -> None:
        self.args = kwargs
        self.dataset_path = Path(self.args['dataset_path'])
        self.normalize_mean = 0.5
        self.normalize_std = 0.5
        self.transforms = transforms.Compose(
            [
                transforms.Resize((self.args['image_size'], self.args['image_size'])),
                transforms.CenterCrop((self.args['image_size'], self.args['image_size'])),
                transforms.Grayscale(1),
                transforms.ToTensor(),
                self.normalize()
            ]
        )
        self.image_path = self.read_dataset()
        self.image = self.open_image(self.image_path)

        self.length = len(self.image_path)

    def normalize(self):
        mean = self.normalize_mean
        std = self.normalize_std
        return transforms.Normalize(mean=[mean], std=[std])

    def de_normalize(self, tensor):
        mean = -self.normalize_mean / self.normalize_std
        std = 1 / self.normalize_std
        t = transforms.Normalize(mean=[mean], std=[std])
        return t(tensor)

    def read_dataset(self):
        image_path = []
        for path in self.dataset_path.glob("**/*"):
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
        return self.image[index]

    def __len__(self):
        return self.length


class DenseGAN():
    def __init__(self) -> None:
        self.args = Settings()
        wandb.init(project=self.args.project_name, config=self.args.dict(), save_code=True)

    def load_dataset(self):
        self.dataset = MyDataset(**self.args.dict())
        self.loader = DataLoader(dataset=self.dataset, batch_size=self.args.batch)

    def load_model(self):
        self.g_model, self.g_loss, self.g_optim = self.generator()
        self.d_model, self.d_loss, self.d_optim = self.discriminator()

    def generator(self):
        model = Generator().to(self.args.device)
        criterion = nn.BCELoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=self.args.learning_rate)
        return model, criterion, optimizer

    def discriminator(self):
        model = Discriminator().to(self.args.device)
        criterion = nn.BCELoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=self.args.learning_rate)
        return model, criterion, optimizer

    def train_one_epoch(self, i_epoch: int):
        self.metric = {
            self.train_d.__name__: {},
            self.train_g.__name__: {},
        }
        self.i_epoch = i_epoch
        # self.d_model.train(mode=True)
        # self.g_model.train(mode=True)
        bar = tqdm(self.loader, unit='batch', leave=True)
        for i_batch, (data) in enumerate(bar):
            self.step = (i_epoch * len(self.loader) + i_batch)
            data = data.to(self.args.device)

            self.d_model.train(mode=True)
            self.g_model.train(mode=False)
            self.train_d(data)

            self.d_model.train(mode=False)
            self.g_model.train(mode=True)
            fake_inputs = self.train_g()

            loss = {
                'd_loss': self.metric[self.train_d.__name__]['loss'][-1],
                'g_loss': self.metric[self.train_g.__name__]['loss'][-1],
            }
            wandb.log({**loss, **{'i_epoch': self.i_epoch, 'step': self.step, }}, step=self.step)

            self.show_result(fake_inputs.view(-1, 1, self.args.image_size, self.args.image_size))

            bar.set_description(f'Epoch [{self.i_epoch + 1}/{self.args.epoch}]')
            bar.set_postfix(**loss)

        d_loss = sum(self.metric[self.train_d.__name__]['loss']) / len(self.loader)
        g_loss = sum(self.metric[self.train_g.__name__]['loss']) / len(self.loader)
        print(f"d_loss: {d_loss}")
        print(f"g_loss: {g_loss}")
        return 0

    def show_result(self, fake_inputs):
        if self.step % self.args.sample_interval == 0:
            wandb.log({
                'fake_inputs': [wandb.Image(im.detach().cpu().numpy()) for im in fake_inputs]
            }, step=self.step)
        return 0

    def train_d(self, data):
        self.d_optim.zero_grad()

        # feed real data
        real_inputs = data.view(-1, 2500)
        real_outputs = self.d_model.forward(real_inputs)
        real_label = torch.ones(real_inputs.shape[0], 1).to(self.args.device)

        # feed fake data
        noise = ((torch.rand(real_inputs.shape[0], 128) - 0.5) / 0.5).to(self.args.device)  # [-1, 1]
        fake_inputs = self.g_model.forward(noise)
        fake_outputs = self.d_model.forward(fake_inputs)
        fake_label = torch.zeros(fake_inputs.shape[0], 1).to(self.args.device)

        # loss
        outputs = torch.cat((real_outputs, fake_outputs), 0)
        targets = torch.cat((real_label, fake_label), 0)
        d_loss = self.d_loss.forward(outputs, targets)

        # backward
        # torch.autograd.backward(d_loss)
        d_loss.backward()

        self.d_optim.step()
        self.metric[self.train_d.__name__].setdefault('loss', []).append(d_loss.item())
        return 0

    def train_g(self):
        self.g_optim.zero_grad()

        # generate fake data, and feed fake data
        noise = ((torch.rand(self.args.batch, 128) - 0.5) / 0.5).to(self.args.device)
        fake_inputs = self.g_model.forward(noise)
        fake_outputs = self.d_model.forward(fake_inputs)

        # loss
        g_loss = self.d_loss(fake_outputs, (torch.ones([fake_outputs.shape[0], 1])).to(self.args.device))

        # backward
        # torch.autograd.backward(g_loss)
        g_loss.backward()

        self.g_optim.step()
        self.metric[self.train_g.__name__].setdefault('loss', []).append(g_loss.item())
        return fake_inputs

    def train(self):
        wandb.watch(self.d_model, criterion=None, log='gradients', log_freq=1000, idx=0)
        wandb.watch(self.g_model, criterion=None, log='gradients', log_freq=1000, idx=1)
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
    dense_gan = DenseGAN()
    dense_gan.load_dataset()
    dense_gan.load_model()
    dense_gan.train()
