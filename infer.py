import torch
from networks._mlp import mlp
from utils.parsing import test_parser_args
from utils.utils import (get_target_infer_image, get_hparam, 
                        get_trained_weight, postprocess_image)

def main():
    args = test_parser_args()

    image_tensor = get_target_infer_image(args)
    ##학습이 완료된 최고의 모델을 준비하기
    #저장된 폴더를 지정받고
    hparam = get_hparam(args)
    weight = get_trained_weight(args)

    # device 설정
    args.device = 'cuda' if torch.cuda.is_available() else 'cpu'

    model = mlp(hparam['input_size'], hparam['hidden_size'], hparam['output_size'])
    # 속이 빈 모델에 학습된 모델의 weight 덮어 씌우기
    model.load_state_dict(weight)
    model = model.to(args.device)

    #준비된 데이터를 모델에 넣기 
    output = model(image_tensor)

    prob, predict = postprocess_image(output)
    print(f'이미지를 보고 모델은 {prob:.2f}의 확률로 {predict}라고 답했다')
    
if __name__ == '__main__':
    main()