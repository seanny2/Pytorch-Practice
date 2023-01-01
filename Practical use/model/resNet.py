import torch
import torch.nn as nn

class BasicBlock(nn.Module):
  def  __init__(self, in_channels, out_channels, kernel_size=3):
    super(BasicBlock, self).__init__()
    
    self.c1 = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=1)
    self.c2 = nn.Conv2d(out_channels, out_channels, kernel_size=kernel_size, padding=1)
    self.downsample = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    self.bn1 = nn.BatchNorm2d(num_features=out_channels)
    self.bn2 = nn.BatchNorm2d(num_features=out_channels)

    self.relu = nn.ReLU()

  def forward(self, x):
    x_ = x

    x = self.c1(x)
    x = self.bn1(x)
    x = self.relu(x)
    x = self.c2(x)
    x = self.bn2(x)

    x_ = self.downsample(x_)

    x += x_
    x = self.relu(x)

    return x
    
class ResNet(nn.Module):
  def __init__(self, num_classes=10):
    super(ResNet, self).__init__()

    self.b1 = BasicBlock(in_channels=3, out_channels=64)
    self.b2 = BasicBlock(in_channels=64, out_channels=128)
    self.b3 = BasicBlock(in_channels=128, out_channels=256)

    self.pool = nn.AvgPool2d(kernel_size=2, stride=2)

    self.fc1 = nn.Linear(in_features=4096, out_features=2048)
    self.fc2 = nn.Linear(in_features=2048, out_features=512)
    self.fc3 = nn.Linear(in_features=512, out_features=num_classes)

    self.relu = nn.ReLU()

  def forward(self, x):
    x = self.b1(x)
    x = self.pool(x)
    x = self.b2(x)
    x = self.pool(x)
    x = self.b3(x)
    x = self.pool(x)

    x = torch.flatten(x, start_dim=1)

    x = self.fc1(x)
    x = self.relu(x)
    x = self.fc2(x)
    x = self.relu(x)
    x = self.fc3(x)

    return x