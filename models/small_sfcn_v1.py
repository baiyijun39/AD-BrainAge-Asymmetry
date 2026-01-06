import torch.nn as nn
import torch

class Small_SFCN(nn.Module):
    def __init__(self, input_channels=1, channel_number=[24, 48, 64, 128, 64, 48], output_dim=50, dropout=0.1):
        super(Small_SFCN, self).__init__()
        n_layer = len(channel_number)
        self.feature_extractor = nn.Sequential()
        for i in range(n_layer):
            if i == 0:
                in_channel = input_channels
            else:
                in_channel = channel_number[i-1]
            out_channel = channel_number[i]
            if i < n_layer-1:
                self.feature_extractor.add_module('conv_%d' % i,
                                                  self.conv_layer(in_channel,
                                                                  out_channel,
                                                                  maxpool=True,
                                                                  kernel_size=3,
                                                                  padding=1))
            else:
                self.feature_extractor.add_module('conv_%d' % i,
                                                  self.conv_layer(in_channel,
                                                                  out_channel,
                                                                  maxpool=False,
                                                                  kernel_size=1,
                                                                  padding=0))
        self.gap = nn.AdaptiveAvgPool3d((1, 1, 1)) 
        self.dropout = nn.Dropout(dropout)
        
        in_channel = channel_number[-1]
        self.regressor = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_channel, 64),  
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1), 
        )

    @staticmethod
    def conv_layer(in_channel, out_channel, maxpool=True, kernel_size=3, padding=0, maxpool_stride=2):
        if maxpool:
            layer = nn.Sequential(
                nn.Conv3d(in_channel, out_channel, padding=padding, kernel_size=kernel_size),
                nn.BatchNorm3d(out_channel),
                nn.MaxPool3d(2, stride=maxpool_stride),
                nn.ReLU(),
            )
        else:
            layer = nn.Sequential(
                nn.Conv3d(in_channel, out_channel, padding=padding, kernel_size=kernel_size),
                nn.BatchNorm3d(out_channel),
                nn.ReLU()
            )
        return layer

    def forward(self, x):
       
        x_f = self.feature_extractor(x)
        x_f = self.gap(x_f)
        x_f = self.dropout(x_f)
        out = self.regressor(x_f)
        return out

