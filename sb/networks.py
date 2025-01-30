import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.utils.data import Dataset
import sb.utils

class AdaIN(nn.Module):
    def __init__(self, feature_channels, condition_dim):
        super(AdaIN, self).__init__()
        self.norm = nn.InstanceNorm2d(feature_channels, affine=False)
        self.fc = nn.Linear(condition_dim, feature_channels * 2)

    def forward(self, x, t):
        # x: [batch_size, feature_channels, height, width]
        # t: [batch_size, condition_dim]
        h = self.fc(t)  # [batch_size, feature_channels * 2]
        h = h.view(h.size(0), h.size(1), 1, 1)  # [batch_size, feature_channels * 2, 1, 1]
        gamma, beta = torch.chunk(h, 2, dim=1)  # 分成gamma和beta
        x = self.norm(x)  # 实例归一化
        x = gamma * x + beta  # 应用条件归一化
        return x

# 定义两层UNet
class SimpleUNet(nn.Module):
    def __init__(self, condition_dim=32):
        super(SimpleUNet, self).__init__()
        # 编码器
        self.encoder_conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1).double()
        self.encoder_conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1).double()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2).double()

        # AdaIN层
        self.adain1 = AdaIN(32, condition_dim).double()
        self.adain2 = AdaIN(64, condition_dim).double()

        # 解码器
        self.upconv1 = nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2).double()  # 上采样
        self.decoder_conv1 = nn.Conv2d(96, 32, kernel_size=3, padding=1).double()  # 解码器卷积
        self.upconv2 = nn.ConvTranspose2d(32, 32, kernel_size=2, stride=2).double()  # 上采样
        self.decoder_conv2 = nn.Conv2d(64, 32, kernel_size=3, padding=1).double()  # 解码器卷积
        self.final_conv = nn.Conv2d(32, 1, kernel_size=3, padding=1).double()  # 输出层

    def forward(self, x_vector, t):
        x = x_vector.reshape(x_vector.shape[0], 1, 32, 32)
        x1 = F.leaky_relu(self.encoder_conv1(x))  # [batch_size, 32, 32, 32]
        x1 = self.adain1(x1, t)  # 应用AdaIN
        x2 = self.pool(x1)  # [batch_size, 32, 16, 16]

        x2 = F.leaky_relu(self.encoder_conv2(x2))  # [batch_size, 64, 16, 16]
        x2 = self.adain2(x2, t)  # 应用AdaIN
        x3 = self.pool(x2)  # [batch_size, 64, 8, 8]

        # 解码器部分
        x4 = self.upconv1(x3)  # [batch_size, 32, 16, 16]
        x4 = torch.cat([x4, x2], dim=1)  # 跳跃连接 [batch_size, 64, 16, 16]
        x4 = F.leaky_relu(self.decoder_conv1(x4))  # [batch_size, 32, 16, 16]

        x5 = self.upconv2(x4)  # [batch_size, 32, 32, 32]
        x5 = torch.cat([x5, x1], dim=1)  # 跳跃连接 [batch_size, 64, 32, 32]
        x5 = F.leaky_relu(self.decoder_conv2(x5))  # [batch_size, 32, 32, 32]
        output = self.final_conv(x5)
        return output.reshape(x_vector.shape[0], -1)

class scale_model_muti(nn.Module):
    def __init__(self,output_size,hidden_size=16,input_size=2):
        super(scale_model_muti, self).__init__()
        self.fc1 = nn.Linear(input_size,hidden_size).double()
        self.fc2 = nn.Linear(hidden_size,output_size).double()
    def forward(self,t,stage):
        B=t.shape[0]
        stage=torch.from_numpy(np.repeat(stage,B)).reshape(-1,1).cuda()
        x=torch.cat([t,stage],dim=1).double()
        x=F.leaky_relu(self.fc1(x))
        x=self.fc2(x)
        return x
    
class CustomDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]