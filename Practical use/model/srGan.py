import torch
import torch.nn as nn

class ResidualBlock(nn.Module):
  def __init__(self, in_channels, out_channels):
    super(ResidualBlock, self).__init__()
    self.layers = nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1),
        nn.BatchNorm2d(out_channels),
        nn.PReLU(),
        nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1),
        nn.BatchNorm2d(out_channels)
    )
  def forward(self, x):
    x_ = x
    x = self.layers(x)
    x = x_ + x
    return x_
  
class UpSample(nn.Sequential):
  def __init__(self, in_channels, out_channels):
    super(UpSample, self).__init__(
        nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1),
        nn.PixelShuffle(upscale_factor=2),
        nn.PReLU()
    )

class Generator(nn.Module):
  def __init__(self):
    super(Generator, self).__init__()
    self.conv1 = nn.Sequential(
        nn.Conv2d(3, 64, kernel_size=9, stride=1, padding=4),
        nn.PReLU()
    )
    self.res_blocks = nn.Sequential(
        ResidualBlock(in_channels=64, out_channels=64),
        ResidualBlock(in_channels=64, out_channels=64),
        ResidualBlock(in_channels=64, out_channels=64)
    )
    self.conv2 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
    self.bn2 = nn.BatchNorm2d(64)
    self.upsample_blocks = nn.Sequential(
        UpSample(in_channels=64, out_channels=256)
    )
    self.conv3 = nn.Conv2d(64, 3, kernel_size=9, stride=1, padding=4)
  
  def forward(self, x):
    x = self.conv1(x)
    x_ = x
    x = self.res_blocks(x)
    x = self.conv2(x)
    x = self.bn2(x)
    x = x + x_
    x = self.upsample_blocks(x)
    x = self.conv3(x)
    return x
   
class DiscBlock(nn.Module):
  def __init__(self, in_channels, out_channels):
    super(DiscBlock, self).__init__()
    self.layers = nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=2, padding=1),
        nn.BatchNorm2d(out_channels),
        nn.LeakyReLU()
    )
  def forward(self, x):
    return self.layers(x)

class Discriminator(nn.Module):
  def __init__(self):
    super(Discriminator, self).__init__()
    self.conv1 = nn.Sequential(
        nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1),
        nn.LeakyReLU()
    )
    self.blocks = DiscBlock(in_channels=64, out_channels=64)
    self.fc1 = nn.Linear(65536, 1024)
    self.activation = nn.LeakyReLU()
    self.fc2 = nn.Linear(1024, 1)
    self.sigmoid = nn.Sigmoid()

  def forward(self, x):
    x = self.conv1(x)
    x = self.blocks(x)
    x = torch.flatten(x, start_dim=1)
    x = self.fc1(x)
    x = self.activation(x)
    x = self.fc2(x)
    x = self.sigmoid(x)
    return x