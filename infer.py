import torch
import argparse
from networks.mymlp import mlp
from utils.parsing import test_parser_args
from utils.utils import (get_target_infer_image, get_hparam, 
                        get_trained_weight, postprocess_image, get_models)

def main():
    args = test_parser_args()
   
    # args를 json으로 변경하고, arg와 hparams를 합쳐서 새로운 NameSpace객체 생성
    json_args = vars(args)
    json_args.update(args)
    args = argparse.Namespace(**json_args)


    # device 설정
    args.device = 'cuda' if torch.cuda.is_available() else 'cpu'

    weight = get_trained_weight(args)

    # 추론에 필요한 이미지 준비
    image_tensor = get_target_infer_image(args)

    model = get_models(args)

    # 속이 빈 모델에 학습된 모델의 weight 덮어 씌우기
    model.load_state_dict(weight)
    model = model.to(args.device)

    model.eval()

    #준비된 데이터를 모델에 넣기 
    output = model(image_tensor)

    prob, predict = postprocess_image(output)
    print(f'이미지를 보고 모델은 {prob:.2f}의 확률로 {predict}라고 답했다')
    
if __name__ == '__main__':
    main()