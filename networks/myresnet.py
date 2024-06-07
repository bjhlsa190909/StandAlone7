import torch
import torch.nn as nn

class InputPart(nn.Module):
    def __init__(self):
        super().__init__()
        self.module1 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=64,
                      kernel_size=7, stride=2, padding=3),
            nn.BatchNorm2d(64),
            nn.ReLU()
        )

        self.module2 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
    
    def forward(self, x):
        x = self.module1(x)
        x = self.module2(x)
        return x

class OutputPart(nn.Module):
    def __init__(self, linear_input, linear_output):
        super().__init__()
        self.module1 = nn.AdaptiveAvgPool2d(output_size=(1,1))
        self.module2 = nn.Linear(linear_input, linear_output)

    def forward(self, x):
        x = self.module1(x)
        x = self.module1(x)
        return x
    
class MiddlePart(nn.Module):
    def __init__(self, layer_numa = [2,2,2,2]):
        super().__init__()
        self.layer1 = self.make_layer(64, 64, layer_numa[0], False)
        self.layer2 = self.make_layer(64, 128, layer_numa[1], True)
        self.layer3 = self.make_layer(128, 256, layer_numa[2], True)
        self.layer4 = self.make_layer(256, 512, layer_numa[3], True)
    
    def make_layer(self, in_channels, out_channels, _num, size_matching):
        # num횟수만큼 동일 channel의 block의 개수 만듦
        
        