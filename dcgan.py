from pathlib import Path

import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
from tqdm import tqdm
from pydantic import BaseModel
import wandb


# [DCGAN Tutorial — PyTorch Tutorials 1.13.1+cu117 documentation]
# (https://pytorch.org/tutorials/beginner/dcgan_faces_tutorial.html#implementation)


class Settings(BaseModel):
    project_name: str = 'DCGAN'
    device: str = 'cuda'
    epoch: int = 2000
    batch: int = 8
    learning_rate: float = 2e-4
    image_size: int = 64
    sample_interval: int = 500

    dataset_path: str = "/home/hpds/yucheng/gan-pytorch/dataset/anime"

    # Size of z latent vector (i.e. size of generator input)
    nz = 100
    # Size of feature maps in generator
    ngf = 64
    # Number of channels in the training images. For color images this is 3
    nc = 3
    # Size of feature maps in discriminator
    ndf = 64
    # Beta1 hyperparam for Adam optimizers
    beta1 = 0.5
    # Establish convention for real and fake labels during training
    real_label = 1.
    fake_label = 0.


class Discriminator(nn.Module):
    def __init__(self, nc=3, ndf=64):
        super(Discriminator, self).__init__()
        self.main = nn.Sequential(
            nn.Conv2d(nc, ndf, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf) x 32 x 32
            nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*2) x 16 x 16
            nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*4) x 8 x 8
            nn.Conv2d(ndf * 4, ndf * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 8),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*8) x 4 x 4
            nn.Conv2d(ndf * 8, 1, 4, 1, 0, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.main(x)


class Generator(nn.Module):
    def __init__(self, nz=100, ngf=64, nc=3):
        super(Generator, self).__init__()
        self.main = nn.Sequential(
            # input is Z, going into a convolution
            nn.ConvTranspose2d(nz, ngf * 8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(ngf * 8),
            nn.ReLU(True),
            # state size. (ngf*8) x 4 x 4
            nn.ConvTranspose2d(ngf * 8, ngf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 4),
            nn.ReLU(True),
            # state size. (ngf*4) x 8 x 8
            nn.ConvTranspose2d(ngf * 4, ngf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 2),
            nn.ReLU(True),
            # state size. (ngf*2) x 16 x 16
            nn.ConvTranspose2d(ngf * 2, ngf, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf),
            nn.ReLU(True),
            # state size. (ngf) x 32 x 32
            nn.ConvTranspose2d(ngf, nc, 4, 2, 1, bias=False),
            nn.Tanh()
            # state size. (nc) x 64 x 64
        )

    def forward(self, x):
        return self.main(x)


class MyDataset(Dataset):
    def __init__(self, **kwargs) -> None:
        self.args = kwargs
        self.dataset_path = Path(self.args['dataset_path'])
        self.normalize_mean = 0.5
        self.normalize_std = 0.5
        self.transforms = transforms.Compose([
            transforms.Resize(self.args['image_size']),
            transforms.CenterCrop(self.args['image_size']),
            transforms.ToTensor(),
            self.normalize()
        ])
        self.image_path = self.read_dataset()
        self.image = self.open_image(self.image_path)

        self.length = len(self.image_path)

    def normalize(self):
        mean = self.normalize_mean
        std = self.normalize_std
        return transforms.Normalize(mean=[mean, mean, mean], std=[std, std, std])

    def de_normalize(self, tensor):
        mean = -self.normalize_mean / self.normalize_std
        std = 1 / self.normalize_std
        t = transforms.Normalize(mean=[mean, mean, mean], std=[std, std, std])
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


class DCGAN():
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
        model.apply(self.weights_init)
        criterion = nn.BCELoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=self.args.learning_rate, betas=(self.args.beta1, 0.999))
        return model, criterion, optimizer

    def discriminator(self):
        model = Discriminator().to(self.args.device)
        model.apply(self.weights_init)
        criterion = nn.BCELoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=self.args.learning_rate, betas=(self.args.beta1, 0.999))
        return model, criterion, optimizer

    def weights_init(self, m):
        # custom weights initialization called on netG and netD
        classname = m.__class__.__name__
        if classname.find('Conv') != -1:
            nn.init.normal_(m.weight.data, 0.0, 0.02)
        elif classname.find('BatchNorm') != -1:
            nn.init.normal_(m.weight.data, 1.0, 0.02)
            nn.init.constant_(m.bias.data, 0)

    def train_one_epoch(self, i_epoch: int):
        self.metric = {
            self.train_d.__name__: {},
            self.train_g.__name__: {},
        }
        self.i_epoch = i_epoch
        self.d_model.train(mode=True)
        self.g_model.train(mode=True)
        bar = tqdm(self.loader, unit='batch', leave=True)
        for i_batch, (data) in enumerate(bar):
            self.step = (i_epoch * len(self.loader) + i_batch)
            data = data.to(self.args.device)

            fake = self.train_d(data)

            self.train_g(fake)

            loss = {
                'd_loss': self.metric[self.train_d.__name__]['loss'][-1],
                'g_loss': self.metric[self.train_g.__name__]['loss'][-1],
            }
            wandb.log({**loss, **{'i_epoch': self.i_epoch, 'step': self.step, }}, step=self.step)

            self.show_result(fake)

            bar.set_description(f'Epoch [{self.i_epoch + 1}/{self.args.epoch}]')
            bar.set_postfix(**loss)

        d_loss = sum(self.metric[self.train_d.__name__]['loss']) / len(self.loader)
        g_loss = sum(self.metric[self.train_g.__name__]['loss']) / len(self.loader)
        print(f"d_loss: {d_loss}")
        print(f"g_loss: {g_loss}")
        return 0

    def show_result(self, fake):
        if self.step % self.args.sample_interval == 0:
            wandb.log({
                'fake': [wandb.Image(im.permute(1,2,0).detach().cpu().numpy()) for im in fake]
            }, step=self.step)
        return 0

    def train_d(self, data):
        ############################
        # (1) Update D network: maximize log(D(x)) + log(1 - D(G(z)))
        ###########################
        # Train with all-real batch
        self.d_optim.zero_grad()
        # Format batch
        real_inputs = data.to(self.args.device)
        b_size = real_inputs.size(0)
        label = torch.full((b_size,), self.args.real_label, dtype=torch.float, device=self.args.device)
        # Forward pass real batch through D
        output = self.d_model.forward(real_inputs).view(-1)
        # Calculate loss on all-real batch
        errD_real = self.d_loss.forward(output, label)
        # Calculate gradients for D in backward pass
        torch.autograd.backward(errD_real)
        D_x = output.mean().item()

        # Train with all-fake batch
        # Generate batch of latent vectors
        noise = torch.randn(b_size, self.args.nz, 1, 1, device=self.args.device)
        # Generate fake image batch with G
        fake = self.g_model.forward(noise)
        label.fill_(self.args.fake_label)
        # Classify all fake batch with D
        output = self.d_model.forward(fake.detach()).view(-1)
        # Calculate D's loss on the all-fake batch
        errD_fake = self.d_loss.forward(output, label)
        # Calculate the gradients for this batch, accumulated (summed) with previous gradients
        torch.autograd.backward(errD_fake)
        D_G_z1 = output.mean().item()
        # Compute error of D as sum over the fake and the real batches
        errD = errD_real + errD_fake
        # Update D
        self.d_optim.step()
        self.metric[self.train_d.__name__].setdefault('loss', []).append(errD.item())
        return fake

    def train_g(self, fake):
        ############################
        # (2) Update G network: maximize log(D(G(z)))
        ###########################
        self.g_optim.zero_grad()
        # fake labels are real for generator cost
        label = torch.full((fake.size(0),), self.args.real_label, dtype=torch.float, device=self.args.device)
        # Since we just updated D, perform another forward pass of all-fake batch through D
        output = self.d_model.forward(fake).view(-1)
        # Calculate G's loss based on this output
        errG = self.d_loss(output, label)
        # Calculate gradients for G
        torch.autograd.backward(errG)
        D_G_z2 = output.mean().item()
        # Update G
        self.g_optim.step()
        self.metric[self.train_g.__name__].setdefault('loss', []).append(errG.item())
        return 0

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
    deep_conv_gan = DCGAN()
    deep_conv_gan.load_dataset()
    deep_conv_gan.load_model()
    deep_conv_gan.train()
