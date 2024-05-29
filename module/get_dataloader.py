from torchvision.transforms import ToTensor
from torchvision.datasets import MNIST, CIFAR10
from torch.utils.data import DataLoader
from torchvision.transforms import Compose, Resize, Normalize

# CIFAR 이미지 관련 숫자
cifar_size = 32
cifar_mean = [0.491, 0.482, 0.447]
cifar_std = [0.247, 0.244, 0.262]

# ImageNet 이미지의 평균, 표준편차
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

def get_target_dataset(args):
    # TODO: MNIST dataset 모듈 불러오기
    if args.dataset == 'mnist':
        train_dataset = MNIST(root='../../data', train=True, download=True, transform=ToTensor())
        test_dataset = MNIST(root='../../data', train=False, download=True, transform=ToTensor())
    
    # TODO: CIFAR10 dataset 모듈 불러오기
    elif args.dataset == 'cifar':
        ## TODO: transform 모듈 구현 
        transform = Compose([Resize((cifar_size, cifar_size)),
                            ToTensor(),
                            Normalize(mean=cifar_mean, std=cifar_std)
                            ])        
        train_dataset = CIFAR10(root='../../data', train=True, download=True,transform=transform)
        test_dataset = CIFAR10(root='../../data', train=True, download=True,transform=transform)
    else:
        raise TypeError('타겟 데이터셋 오류')
    return train_dataset, test_dataset

def get_target_loader(args): 
    ## 데이터 불러오기 ## 
    train_dataset, test_dataset = get_target_dataset(args)
      
    train_loader = DataLoader(dataset=train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=0)
    test_loader = DataLoader(dataset=test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)

    return train_loader, test_loader