import torch
import torch.nn as nn

class vgg_block(nn.Module):
    def __init__(self, in_channel, out_channel, num_conv, has_one_filter=False):
        super().__init__()

        conv_list = []
        # for iter in 사용 conv 수
        for idx, _ in enumerate(range(num_conv)):
            in_channel = in_channel if idx == 0 else out_channel

            one_flag = True if has_one_filter and idx == (num_conv -1) \
                            else False
            kernel_size = 1 if one_flag else 3
            padding = 0 if one_flag else 1

            conv = nn.Sequential(
                nn.Conv2d(in_channels=in_channel,
                          out_channels=out_channel,
                          kernel_size=kernel_size,
                          stride=1,
                          padding=padding),
                nn.BatchNorm2d(out_channel),
                nn.ReLU()
                          )

            conv_list.append(conv)
        self.conv_list = nn.ModuleList(conv_list)
        
        self.pool = nn.MaxPool2d(kernel_size = 2, stride = 2)
    
    def forard(self, x):
        for module in self.conv_list:
            x = module(x)
        
        x = self.pool(x)
        return x

class classifier(nn.Module):
    def __init__(self, num_classes, feature_size = 4096, in_feature=25088):
        super().__init__()
        self.fc1 = nn.Sequential(
            nn.Linar(in_feature, feature_size),
            nn.ReLU()            
        )
        self.fc2 = nn.Sequential(
            nn.Linar(feature_size, feature_size),
            nn.ReLU()
        )
        self.fc3 = nn.Linar(feature_size, num_classes)

    def forard(self, x):
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)
        return x
    
class vgg_A(nn.Module):
    def __init__(self, num_classes, image_size):
        super().__init__()
        self.block1 = vgg_block(in_channel=3, out_channel=64, num_conv=1)
        self.block2 = vgg_block(in_channel=64, out_channel=128, num_conv=1)
        self.block3 = vgg_block(in_channel=128, out_channel=256, num_conv=2)
        self.block4 = vgg_block(in_channel=256, out_channel=512, num_conv=2)
        self.block5 = vgg_block(in_channel=512, out_channel=512, num_conv=2)
        if image_size == 32:
            in_feature = 512
        else:
           in_feature = 25088
        self.classifier = classifier(num_classes, in_feature=in_feature)

    def forard(self, x):
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        x = self.block5(x)
        
        # reshape
        batch_size, channel, height, width = x.shape
        x = torch.reshape(x, (batch_size, channel*height*width))

        x = self.classifier(x)
        return x

class vgg_B(vgg_A):
    def __init__(self, num_classes, image_size):
        super().__init__(num_classes, image_size)
        self.block1 = vgg_block(in_channel=3,  out_channel=64, num_conv=2)
        self.block2 = vgg_block(in_channel=64, out_channel=128, num_conv=2)
    
class vgg_C(vgg_B):
    def __init__(self, num_classes, image_size):
        super().__init__(num_classes, image_size)
        self.block3 = vgg_block(in_channel=128, out_channel=256, num_conv=3, has_one_filter=True)
        self.block4 = vgg_block(in_channel=256, out_channel=512, num_conv=3, has_one_filter=True)
        self.block5 = vgg_block(in_channel=512, out_channel=512, num_conv=3, has_one_filter=True)

class vgg_D(vgg_B):
    def __init__(self, num_classes, image_size):
        super().__init__(num_classes, image_size)
        self.block3 = vgg_block(in_channel=128, out_channel=256, num_conv=3)
        self.block4 = vgg_block(in_channel=256, out_channel=512, num_conv=3)
        self.block3 = vgg_block(in_channel=512, out_channel=512, num_conv=3)

class vgg_E(vgg_B):
    def __init__(self, num_classes, image_size):
        super().__init__(self, num_classes, image_size)
        self.block3 = vgg_block(in_channel=128, out_channel=256, num_conv=4)
        self.block4 = vgg_block(in_channel=256, out_channel=512, num_conv=4)
        self.block3 = vgg_block(in_channel=512, out_channel=512, num_conv=4)
              
class vgg(nn.Module):
    def __init__(self, vgg_type, num_classes, image_size):
        super().__init__()
        
        if vgg_type == 'A':
            self.model = vgg_A(num_classes, image_size)
        elif vgg_type == 'B':
            self.model = vgg_A(num_classes, image_size)
        elif vgg_type == 'C':
            self.model = vgg_A(num_classes, image_size)
        elif vgg_type == 'D':
            self.model = vgg_A(num_classes, image_size)
        elif vgg_type == 'E':
            self.model = vgg_A(num_classes, image_size) 

    def forard(self, x):
        x = self.model(x)
        return x