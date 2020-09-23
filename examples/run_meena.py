import argparse
import random

import numpy as np
import torch

import sentencepiece as spm
from dialogue_generation_models.configuration_meena import MeenaConfig
from dialogue_generation_models.modeling_meena import MeenaForConditionalGeneration


parser = argparse.ArgumentParser()
parser.add_argument(
    "--pretrained-model-path",
    default="/nas/models/meena/torch_weights/base_filter.pth",
    type=str,
    help="사전학습된 모델 경로",
)
parser.add_argument(
    "--model-config-path",
    default="./configs/base_meena_config.json",
    type=str,
    help="모델 설정 파일",
)
parser.add_argument(
    "--tokenizer-model-path",
    default="./tokenizer/tok.model",
    type=str,
    help="sentencepiece model 경로",
)
parser.add_argument(
    "--decoding-method",
    default="beam_search",
    type=str,
    help="디코딩 방법 (beam_search or top_p)",
)


def main(args):
    np.random.seed(42)
    torch.manual_seed(42)
    random.seed(42)
    tokenizer = spm.SentencePieceProcessor(model_file=args.tokenizer_model_path)

    config = MeenaConfig.from_json(args.model_config_path)
    model = MeenaForConditionalGeneration(config)
    model.load_state_dict(torch.load(args.pretrained_model_path, map_location="cpu"))
    model.eval()

    contexts = [
        ["나 야나두 해보려고ㅋㅋ"],
        ["나 야나두 해보려고ㅋㅋ", "영어 배우게?"],
        ["나 야나두 해보려고ㅋㅋ", "영어 배우게?", "웅웅"],
        ["나 야나두 해보려고ㅋㅋ", "영어 배우게?", "웅웅", "야나두 괜찮나?"],
        ["나 야나두 해보려고ㅋㅋ", "영어 배우게?", "웅웅", "야나두 괜찮나?", "주변에서 추천을 많이해줘서 한번 해보게"],
        ["나 언제 데리러 올거야?"],
        ["나 언제 데리러 올거야?", "수업 끝나고 바로 갈게"],
        ["나 언제 데리러 올거야?", "수업 끝나고 바로 갈게", "얼마나 걸려?"],
        ["나 언제 데리러 올거야?", "수업 끝나고 바로 갈게", "얼마나 걸려?", "한 10분 안에 끝날 거 같은데?"],
        [
            "나 언제 데리러 올거야?",
            "수업 끝나고 바로 갈게",
            "얼마나 걸려?",
            "한 10분 안에 끝날 거 같은데?",
            "그럼 건물 앞에서 기다릴게",
        ],
    ]

    for context in contexts:
        # 컨텍스트 문장 사이에 [SEPT] 토큰을 입력
        input_id = [
            token_id
            for raw_str in context
            for token_id in tokenizer.encode(raw_str, out_type=int) + [config.sept_token_id]
        ] + [config.bos_token_id]

        input_tensor = torch.tensor([input_id])

        if args.decoding_method == "top_p":
            outputs = model.generate(
                input_ids=input_tensor,
                max_length=256,
                min_length=8,
                temperature=1.0,
                do_sample=True,
                top_p=0.8,
                pad_token_id=config.pad_token_id,
                bos_token_id=config.bos_token_id,
                eos_token_id=config.eos_token_id,
                repetition_penalty=1.3,
                no_repeat_ngram_size=3,
                num_return_sequences=10,
            )
        elif args.decoding_method == "beam_search":
            outputs = model.generate(
                input_ids=input_tensor,
                max_length=256,
                min_length=8,
                num_beams=10,
                pad_token_id=config.pad_token_id,
                bos_token_id=config.bos_token_id,
                eos_token_id=config.eos_token_id,
                repetition_penalty=1.3,
                no_repeat_ngram_size=3,
                num_return_sequences=5,
            )
        else:
            raise ValueError("올바른 디코딩 방법을 입력해주세요.")

        context_str = " [SEPT] ".join(context)
        for output in outputs.tolist():
            decoded_response = tokenizer.decode(output)
            print(f"Context: {context_str} \t Reply: {decoded_response}")


if __name__ == "__main__":
    args = parser.parse_args()
    main(args)
