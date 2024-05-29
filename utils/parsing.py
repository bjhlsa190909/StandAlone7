import argparse

def train_parser_args():
    parser = argparse.ArgumentParser()
    # hyper-parameters 설정
    ## 모델 관련
    parser.add_argument('--input_size',type=int, default=28*28, help='모델 입력의 크기')
    parser.add_argument('--hidden_size', type=int, default=50, help='hidden layer크기')
    parser.add_argument('--output_size', type=int, default=10, help='출력의 크기')
    # TODO: 과제 작성에 필요한 argument 추가하기
    parser.add_argument('--model_type',type=str,default='lenet',choices=['lenet', 'mlp'], help='사용 모델 종류')
    ## 데이터 관련
    parser.add_argument('--batch_size', type=int, default=100)
    # TODO: 과제 작성에 필요한 argument 추가하기
    parser.add_argument('--dataset', type=str,default='cifar', choices=['mnist','cifar'])
    parser.add_argument('--num_classes',type=int,default=10)
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