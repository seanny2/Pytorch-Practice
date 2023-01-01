import torch
import torch.nn as nn

class LowLevel(nn.Module):
  def __init__(self):
    super(LowLevel, self).__init__()

    self.low1 = nn.Conv2d(1, 64, kernel_size=3, stride=2, padding=1)

    self.lb1 = nn.BatchNorm2d(64)
    self.low2 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)

    self.lb2 = nn.BatchNorm2d(128)
    self.low3 = nn.Conv2d(128, 128, kernel_size=3, stride=2, padding=1)

    self.lb3 = nn.BatchNorm2d(128)
    self.low4 = nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1)

    self.lb4 = nn.BatchNorm2d(256)
    self.low5 = nn.Conv2d(256, 256, kernel_size=3, stride=2, padding=1)

    self.lb5 = nn.BatchNorm2d(256)
    self.low6 = nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1)

    self.lb6 = nn.BatchNorm2d(512)
    
    self.sigmoid = nn.Sigmoid()
  
  def forward(self, x):
    low = self.low1(x)
    low = self.lb1(low)
    low = self.sigmoid(low)

    low = self.low2(low)
    low = self.lb2(low)
    low = self.sigmoid(low)

    low = self.low3(low)
    low = self.lb3(low)
    low = self.sigmoid(low)

    low = self.low4(low)
    low = self.lb4(low)
    low = self.sigmoid(low)

    low = self.low5(low)
    low = self.lb5(low)
    low = self.sigmoid(low)

    low = self.low6(low)
    low = self.lb6(low)
    low = self.sigmoid(low)

    return low

class MidLevel(nn.Module):
  def __init__(self):
    super(MidLevel, self).__init__()

    self.mid1 = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1)
    
    self.mb1 = nn.BatchNorm2d(512)
    self.mid2 = nn.Conv2d(512, 256, kernel_size=3, stride=1, padding=1)

    self.mb2 = nn.BatchNorm2d(256)

    self.sigmoid = nn.Sigmoid()

  def forward(self, x):
    mid = self.mid1(x)
    mid = self.mb1(mid)
    mid = self.sigmoid(mid)

    mid = self.mid2(mid)
    mid = self.mb2(mid)
    mid = self.sigmoid(mid)

    return mid

class GlobalLevel(nn.Module):
  def __init__(self):
    super(GlobalLevel, self).__init__()

    self.glob1 = nn.Conv2d(512, 512, kernel_size=3, stride=2, padding=1)

    self.gb1 = nn.BatchNorm2d(512)
    self.glob2 = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1)

    self.gb2 = nn.BatchNorm2d(512)
    self.glob3 = nn.Conv2d(512, 512, kernel_size=3, stride=2, padding=1)

    self.gb3 = nn.BatchNorm2d(512)
    self.glob4 = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1)

    self.gb4 = nn.BatchNorm2d(512)

    self.fc1 = nn.Linear(in_features=32768, out_features=1024)
    self.fc2 = nn.Linear(in_features=1024, out_features=512)
    self.fc3 = nn.Linear(in_features=512, out_features=256)

    self.sigmoid = nn.Sigmoid()

  def forward(self, x):
    glo = self.glob1(x)
    glo = self.gb1(glo)
    glo = self.sigmoid(glo)

    glo = self.glob2(glo)
    glo = self.gb2(glo)
    glo = self.sigmoid(glo)

    glo = self.glob3(glo)
    glo = self.gb3(glo)
    glo = self.sigmoid(glo)

    glo = self.glob4(glo)
    glo = self.gb4(glo)
    glo = self.sigmoid(glo)

    glo = torch.flatten(glo, start_dim=1)
    glo = self.fc1(glo)
    glo = self.sigmoid(glo)
    glo = self.fc2(glo)
    glo = self.sigmoid(glo)
    glo = self.fc3(glo)
    glo = self.sigmoid(glo)

    return glo

class Colorization(nn.Module):
  def __init__(self):
    super(Colorization, self).__init__()

    self.color1 = nn.ConvTranspose2d(256, 128, 3, 1, 1)
    self.cb1 = nn.BatchNorm2d(128)

    self.color2 = nn.ConvTranspose2d(128, 64, 2, 2)
    self.cb2 = nn.BatchNorm2d(64)

    self.color3 = nn.ConvTranspose2d(64, 64, 3, 1, 1)
    self.cb3 = nn.BatchNorm2d(64)

    self.color4 = nn.ConvTranspose2d(64, 32, 2, 2)
    self.cb4 = nn.BatchNorm2d(32)
    
    self.color5 = nn.ConvTranspose2d(32, 2, 2, 2)

    self.sigmoid = nn.Sigmoid()

  def forward(self, x):
    color = self.color1(x)
    color = self.cb1(color)
    color = self.sigmoid(color)

    color = self.color2(color)
    color = self.cb2(color)
    color = self.sigmoid(color)

    color = self.color3(color)
    color = self.cb3(color)
    color = self.sigmoid(color)

    color = self.color4(color)
    color = self.cb4(color)
    color = self.sigmoid(color)

    color = self.color5(color)

    return color

class AutoColoringModel(nn.Module):
  def __init__(self):
    super(AutoColoringModel, self).__init__()

    self.low = LowLevel()
    self.mid = MidLevel()
    self.glob = GlobalLevel()
    self.fusion = nn.Conv2d(512, 256, kernel_size=3, stride=1, padding=1)
    self.color = Colorization()
    self.sigmoid = nn.Sigmoid()

  def forward(self, x):
    low = self.low(x)

    mid = self.mid(low)
    glo = self.glob(low)

    fusion = glo.repeat(1, mid.shape[2]*mid.shape[2])
    fusion = torch.reshape(fusion, (-1, 256, mid.shape[2], mid.shape[2]))
    fusion = torch.cat([mid, fusion], dim=1)
    fusion = self.fusion(fusion)
    fusion = self.sigmoid(fusion)

    color = self.color(fusion)

    return color