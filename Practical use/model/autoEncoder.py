import torch.nn as nn
import torch

class BasicBlock(nn.Module):
  def __init__(self, in_channels, out_channels, hidden_dim):
    super(BasicBlock, self).__init__()

    self.conv1 = nn.Conv2d(in_channels, hidden_dim, kernel_size=3, padding=1)
    self.conv2 = nn.Conv2d(hidden_dim, out_channels, kernel_size=3, padding=1)
    self.relu = nn.ReLU()
  
  def forward(self, x):
    x = self.conv1(x)
    x = self.relu(x)
    x = self.conv2(x)
    x = self.relu(x)

    return x
  
class Encoder(nn.Module):
  def __init__(self):
    super(Encoder, self).__init__()

    self.conv1 = BasicBlock(in_channels=1, out_channels=16, hidden_dim=16)
    self.conv2 = BasicBlock(in_channels=16, out_channels=8, hidden_dim=8)
    self.pool = nn.AvgPool2d(kernel_size=2, stride=2)

  def forward(self, x):
    x = self.conv1(x)
    x = self.pool(x)
    x = self.conv2(x)
    x = self.pool(x)

    return x

class Decoder(nn.Module):
  def __init__(self):
    super(Decoder, self).__init__()
    self.conv1 = BasicBlock(in_channels=8, out_channels=8, hidden_dim=8)
    self.conv2 = BasicBlock(in_channels=8, out_channels=16, hidden_dim=16)
    self.conv3 = nn.Conv2d(in_channels=16, out_channels=1, kernel_size=3, padding=1)
    self.upsample1 = nn.ConvTranspose2d(8, 8, kernel_size=2, stride=2)
    self.upsample2 = nn.ConvTranspose2d(16, 16, kernel_size=2, stride=2)

  def forward(self, x):
    x = self.conv1(x)
    x = self.upsample1(x)
    x = self.conv2(x)
    x = self.upsample2(x)
    x = self.conv3(x)

    return x

class CAE(nn.Module):
  def __init__(self):
    super(CAE, self).__init__()
    self.enc = Encoder()
    self.dec = Decoder()

  def forward(self, x):
    x = self.enc(x)
    x = self.dec(x)
    x = torch.squeeze(x)

    return x