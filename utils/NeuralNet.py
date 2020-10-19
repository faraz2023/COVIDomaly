import torch
import torch.nn as nn
import torch.optim as optim


class Flatten(nn.Module):
    def forward(self, input):
        return input.reshape(input.size(0), -1)


class Generator(nn.Module):
    def __init__(self, height, width, channel, device, ngpu, ksize, z_dim, learning_rate=1e-3):
        super(Generator, self).__init__()

        self.height, self.width, self.channel = height, width, channel
        self.device, self.ngpu = device, ngpu
        self.ksize, self.z_dim, self.learning_rate = ksize, z_dim, learning_rate

        # Encoder Layers
        self.enc_bn1 = nn.BatchNorm2d(1)
        self.conv1 = nn.Conv2d(in_channels=self.channel, out_channels=8, kernel_size=self.ksize, stride=1,
                               padding=self.ksize // 2)

        self.enc_bn2 = nn.BatchNorm2d(8)
        self.conv2 = nn.Conv2d(in_channels=8, out_channels=16, kernel_size=self.ksize, stride=1,
                               padding=self.ksize // 2)

        self.enc_bn3 = nn.BatchNorm2d(16)
        self.conv3 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=self.ksize, stride=1,
                               padding=self.ksize // 2)

        self.flat = Flatten()

        self.en_linear1 = nn.Linear((self.height // (2 ** 3)) * (self.width // (2 ** 3)) * self.channel * 32, 256)
        self.en_linear2 = nn.Linear(256, self.z_dim)

        # Decoder Layers
        self.dec_linear1 = nn.Linear(self.z_dim, 256)
        self.dec_linear2 = nn.Linear(256, (self.height // (2 ** 3)) * (self.width // (2 ** 3)) * self.channel * 32)

        self.dec_bn1 = nn.BatchNorm2d(32)
        self.convT1 = nn.ConvTranspose2d(in_channels=32, out_channels=16, kernel_size=self.ksize, stride=1,
                                         padding=self.ksize // 2)

        self.dec_bn2 = nn.BatchNorm2d(16)
        self.convT2 = nn.ConvTranspose2d(in_channels=16, out_channels=8, kernel_size=self.ksize, stride=1,
                                         padding=self.ksize // 2)

        self.dec_bn3 = nn.BatchNorm2d(8)
        self.convT3 = nn.ConvTranspose2d(in_channels=8, out_channels=self.channel, kernel_size=self.ksize, stride=1,
                                         padding=self.ksize // 2)

        # Other Layers
        self.leaky_relu = nn.LeakyReLU(0.2)
        self.maxpool = nn.MaxPool2d(2)
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

    def encode(self, x):
        x = self.enc_bn1(x)
        x = self.conv1(x)
        x = self.leaky_relu(x)
        x = self.maxpool(x)

        x = self.enc_bn2(x)
        x = self.conv2(x)
        x = self.leaky_relu(x)
        x = self.maxpool(x)

        x = self.enc_bn3(x)
        x = self.conv3(x)
        x = self.leaky_relu(x)
        x = self.maxpool(x)

        x = self.flat(x)
        x = self.en_linear1(x)
        x = self.leaky_relu(x)
        x = self.en_linear2(x)

        return (x)

    def decode(self, x):
        x = self.dec_linear1(x)
        x = self.leaky_relu(x)
        x = self.dec_linear2(x)

        x = x.reshape(x.size(0), 32, (self.height // (2 ** 3)), (self.height // (2 ** 3)))

        x = self.dec_bn1(x)
        x = self.convT1(x)
        x = self.leaky_relu(x)
        x = self.upsample(x)

        x = self.dec_bn2(x)
        x = self.convT2(x)
        x = self.leaky_relu(x)
        x = self.upsample(x)

        x = self.dec_bn3(x)
        x = self.convT3(x)
        x = self.leaky_relu(x)
        x = self.upsample(x)

        return x

    def forward(self, x):
        x = self.encode(x)
        x = self.decode(x)

        return x