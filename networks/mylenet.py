import torch
import torch.nn as nn

class lenet(nn.Module):
    def __init__(self, num_classes):
        super().__init__()

        # 필요한 모듈 생성
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=6,
                      kernel_size=5, stride=1, padding=0),
             # 까먹지말고 conv 뒤에는 bn이랑 activate 함수 넣기! 
            nn.BatchNorm2d(6),
            nn.ReLU()        
        )
        self.pool1 = nn.MaxPool2d(2,2)
        self.conv2 = nn.Sequential(
            nn.Conv2d(6, 16, 5, 1, 0),
            nn.BatchNorm2d(16),
            nn.ReLU()        
        )
        self.pool2 = nn.MaxPool2d(2,2)
        # fully connect 적용 전 reshape부분은 forward에서 진행
        self.fc1 = nn.Sequential(
            nn.Linear(400, 120),
            nn.ReLU()
        )
        self.fc2 = nn.Sequential(
            nn.Linear(120, 84),
            nn.ReLU()
        )
        self.fc3 = nn.Linear(84, num_classes)
    
    def forward(self, x):
        # 데이터가 x라고 들어온다
        # x의 크기는 batch, 3, 32, 32 임(cifar)
        batch_size, channel, height, width = x.shape
                
        # 이미지 크기 그대로 앞쪽 모듈부터 진행됨 
        x = self.conv1(x)
        x = self.pool1(x)
        x = self.conv2(x)
        x = self.pool2(x)

        _, channel, height, width = x.shape
        # 중간에 fc를 태우기 전에 reshape 해야하고
        x = torch.reshape(x, (batch_size, channel* height*width))
        # reshape 된 feature가 다시 모듈에 들어가서 출력 됨
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)
        
        return x