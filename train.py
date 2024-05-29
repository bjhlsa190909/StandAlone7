# 패키지 임포트 
import torch
import torch.nn as nn
from torch.optim import Adam
import os
from networks.mymlp import mlp
from utils.parsing import train_parser_args
from module.get_dataloader import get_target_loader
from utils.utils import evaluate, save_args, make_folder, get_models

def main():
    args = train_parser_args()

    # 저장할 폴더 위치를 잡아주고
    args.save_folder = make_folder(args.save_folder_name)
    # 그 위치에 파싱한 args를 저장한다
    save_args(args)

    args.device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # 데이터 준비
    train_loader, test_loader = get_target_loader(args)

    # 모델 객체를 생성
    model = get_models(args).to(args.device)

    # Loss 계산하는 계산기 (분류 문제)
    criteria = nn.CrossEntropyLoss()

    # Optimizer (Adam)
    optim = Adam(params=model.parameters(), lr=args.lr)


    best_acc = 0
    # for loop를 돌면서 데이터를 불러오기 
    for epoch in range(args.epochs):

        for idx, (data, label) in enumerate(train_loader):
            # args.device로 casting
            data = data.to(args.device)
            label = label.to(args.device)
        
            # 불러온 데이터를 모델에 넣기 
            output = model(data)   
            # 나온 출력물(output)로 loss를 계산 
            loss = criteria(output, label)
            # Loss로 back prop 진행 
            loss.backward()
            # optimizer를 이용해 최적화를 수행
            optim.step()
            optim.zero_grad()

            if idx % 100 == 0:
                print('Loss value :', loss.item())

                # 학습 중간에 평가를 진행
                acc = evaluate(model, test_loader, args)
                # 성능이 좋은 경우
                if best_acc < acc:
                    best_acc = acc
                    print('Best acc :', acc)
                    #모델의 weight 저장
                    torch.save(model.state_dict(), 
                            os.path.join(args.save_folder, 'best_model.ckpt'))
                    #필요시 meta data를 저장

if __name__ == '__main__':
    main()













