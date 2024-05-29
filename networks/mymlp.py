import torch.nn as nn

## 모델 만들기 ##
class mlp(nn.Module):
    def __init__(self, input_size=28*28, hidden_size= 500, output_size= 10):
        super().__init__()
        ## 모델 설계도 만들기 (fc1 ~ fc4)
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, hidden_size)
        self.fc4 = nn.Linear(hidden_size, output_size)
    
    def forward(self, x):     # x : 데이터(이미지, batch size * channel * height * width)
        batch_size, channel, height, width = x.shape
    
    ## 실제 데이터가 흘러가는 줄기 만들기 
        # 데이터 펼치기 -> fc1 -> fc2 -> fc3 -> fc4 -> 출력 

        x = x.reshape(batch_size, height*width)
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)
        x = self.fc4(x)
        return x
        