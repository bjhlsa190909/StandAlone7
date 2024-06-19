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
        layer = [self.target(in_channels=in_channels, out_channels=out_channels,
               size_matching=size_matching)] 
        for idx, _ in enumerate(range(_num -1)):
            layer.append(self.target(in_channels=in_channels, out_channels=out_channels))
        layer = nn.Sequential(*layer)
        return layer

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        return x

class Block(nn.Module):
    def __init__(self, in_channels, out_channels, size_matching=False):
        super().__init__()
        self.size_matching = size_matching
        stride = 1
        if self.size_matching:
            self.size_conv = nn.Sequential(
                nn.Conv2d(in_channels=in_channels, out_channels=out_channels,
                          kernel_size=3, stride=2, padding=1),
                nn.BatchNorm2d(out_channels),
                nn.ReLU()
            )
            stride = 2

        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels,
                          kernel_size=3, stride=stride, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()

        )

        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels,
                          kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(out_channels)
                          
        )
        self.relu = nn.ReLU()

    def forward(self, x):
        _x = x.clone()

        x = self.conv1(x)
        x = self.conv2(x)

        if self.size_matching:
            _x = self.size_conv(_x)

        x = x + _x
        x = self.relu(x)
        return x
    
class BottleNeck(nn.Module):
    def __init__(self, in_channels, out_channels, size_matching = False):
        super().__init__()
        middle_channels = out_channels // 4

        stride = 1
        if size_matching:
            stride = 2
        
        self.need_channel_matching = False
        if in_channels != out_channels:
            self.need_channel_matching = True
            self.size_conv = nn.Sequential(
                nn.Conv2d(in_channels=in_channels,
                          out_channels=out_channels,
                          kernel_size=1,
                          stride=stride,
                          padding=0),
                nn.BatchNorm2d(out_channels)
                
            )

        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=in_channels,out_channels=out_channels,
                    kernel_size = 1, stride=1, padding=0),
            nn.BatchNorm2d(middle_channels),
            nn.ReLU()
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels=in_channels,out_channels=out_channels,
                    kernel_size = 3, stride=stride, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(in_channels=middle_channels,out_channels=out_channels,
                    kernel_size = 1, stride=1, padding=0),
            nn.BatchNorm2d(out_channels),
        )
        self.relu = nn.ReLU()

    def forward(self, x):
        _x = x.clone()

        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)

        if self.need_channel_matching:
            _x = self.size_conv(_x)
        
        x = x + _x
        x = self.relu(x)
        return x
    
class resnet(nn.Module):
    def __init__(self, resnet_type, num_classes):
        super().__init__()

        target, channels, layer_nums, linear_input = self.get_infos(resnet_type)

        self.input_part = InputPart()
        self.middle_part = MiddlePart(target, channels, layer_nums)
        self.output_part = OutputPart(linear_input=linear_input,
                                      linear_output=num_classes)
        
    def get_infos(self, resnet_type): 
        if resnet_type == '18': 
            layer_nums = [2,2,2,2]
            channels=[64, 64, 128, 256, 512]
            target=Block
            linear_input = 512
        elif resnet_type == '34': 
            layer_nums = [3,4,6,3]
            channels=[64, 64, 128, 256, 512]
            target=Block
            linear_input = 512
        elif resnet_type == '50': 
            layer_nums = [3,4,6,3]
            channels=[64, 256, 512, 1024, 2048]
            target=BottleNeck
            linear_input = 2048
        elif resnet_type == '101': 
            layer_nums = [3,4,23,3]
            channels=[64, 256, 512, 1024, 2048]
            target=BottleNeck
            linear_input = 2048
        elif resnet_type == '152': 
            layer_nums = [3,8,36,3]
            channels=[64, 256, 512, 1024, 2048]
            target=BottleNeck
            linear_input = 2048
        return target, channels, layer_nums, linear_input

    def forward(self, x):
        x = self.input_part(x)
        x = self.middle_part(x)
        x = self.output_part(x)
        return x