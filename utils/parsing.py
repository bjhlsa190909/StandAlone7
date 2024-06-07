import argparse

def train_parser_args():
    parser = argparse.ArgumentParser()
    # hyper-parameters 설정
    ## 모델 관련
    parser.add_argument('--model_type',type=str,default='lenet',
                        choices=['lenet', 'mlp', 'vgg'], help='사용 모델 종류')

    ### MLP
    parser.add_argument('--input_size',type=int, default=28*28, help='모델 입력의 크기')
    parser.add_argument('--hidden_size', type=int, default=50, help='hidden layer크기')
    parser.add_argument('--output_size', type=int, default=10, help='출력의 크기')
      
    ### LeNet 
    parser.add_argument('--mid_feature', type=int, default=2048, help='LeNet Linear 모델에서 중간 feature 크기를 의미함')
    
    ### VGG
    parser.add_argument('--vgg_type', type=str, default='A', help='A~E까지 vgg type선택')

    ## 데이터 관련
    parser.add_argument('--batch_size', type=int, default=100)
    parser.add_argument('--dataset', type=str,default='cifar', choices=['mnist','cifar'])
    parser.add_argument('--num_classes',type=int,default=10)
    parser.add_argument('--image_size', type=int, default=32)
    ## 학습 관련
    parser.add_argument('lr', type=int, default=0.001, help='학습률')
    parser.add_argument('epochs', type=int, default=10, help='에폭')
    ## 폴더 셋팅 관련
    parser.add_argument('args.save_folder_name', type=str, default='save', help='저장 폴더 묶어주는 상위폴더')
    parser = parser.parse_args()
    
    return parser

def test_parser_args():
    parser = argparse.ArgumentParser()
    # 어떤 데이터를 쓸건지
    parser.add_argument('--example_image_path', type=str, default='mnist_example.jpeg')
    # 어떤 모델을 사용할 것인지
    parser.add_argument('--args.train_folder_path', type=str, default='save/2')
    parser = parser.parse_args()
    return parser