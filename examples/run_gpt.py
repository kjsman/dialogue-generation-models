import argparse
import random

import numpy as np
import torch

import sentencepiece as spm
from dialogue_generation_models.configuration_gpt import GPT2Config
from dialogue_generation_models.modeling_gpt import GPT2LMHeadModel


parser = argparse.ArgumentParser()
parser.add_argument("--pretrained-model-path", type=str, help="Path to pre-trained model", required=True)
parser.add_argument(
    "--model-config-path",
    default="./configs/base_gpt_config.json",
    type=str,
    help="Path to model configuration file",
)
parser.add_argument(
    "--tokenizer-model-path",
    default="./tokenizer/kr_spm.model",
    type=str,
    help="Path to Sentencepiece model",
)
parser.add_argument(
    "--decoding-method",
    default="top_p",
    type=str,
    help="Decoding method (beam_search or top_p)",
)


def main(args):
    np.random.seed(42)
    torch.manual_seed(42)
    random.seed(42)
    tokenizer = spm.SentencePieceProcessor(model_file=args.tokenizer_model_path)

    config = GPT2Config.from_json(args.model_config_path)
    model = GPT2LMHeadModel(config)
    model.load_state_dict(torch.load(args.pretrained_model_path, map_location="cpu"))
    model.eval()

    contexts = [
        ["나 야나두 해보려고ㅋㅋ"],
        ["나 야나두 해보려고ㅋㅋ", "영어 배우게?"],
        ["나 야나두 해보려고ㅋㅋ", "영어 배우게?", "웅웅"],
        ["나 야나두 해보려고ㅋㅋ", "영어 배우게?", "웅웅", "야나두 괜찮나?"],
        ["나 야나두 해보려고ㅋㅋ", "영어 배우게?", "웅웅", "야나두 괜찮나?", "주변에서 추천을 많이해줘서 한번 해보게"],
    ]

    for context in contexts:
        # insert [SEPT] between input utterances
        input_ids = torch.tensor(
            [config.bos_token_id]
            + [
                token_id
                for raw_str in context
                for token_id in tokenizer.encode(raw_str, out_type=int) + [config.sept_token_id]
            ]
        ).unsqueeze(0)

        if args.decoding_method == "top_p":
            outputs = model.generate(
                input_ids=input_ids,
                max_length=48,
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
                bad_words_ids=[[config.sept_token_id]],
            )
        elif args.decoding_method == "beam_search":
            outputs = model.generate(
                input_ids=input_ids,
                max_length=48,
                min_length=8,
                num_beams=10,
                pad_token_id=config.pad_token_id,
                bos_token_id=config.bos_token_id,
                eos_token_id=config.eos_token_id,
                repetition_penalty=1.3,
                no_repeat_ngram_size=3,
                num_return_sequences=10,
                bad_words_ids=[[config.sept_token_id]],
            )
        else:
            raise ValueError("올바른 디코딩 방법을 입력해주세요.")

        context_str = " [SEPT] ".join(context)

        for output in outputs.tolist():
            decoded_response = tokenizer.decode(output[input_ids.size()[1] :])
            print(f"Context: {context_str} \t Reply: {decoded_response}")


if __name__ == "__main__":
    args = parser.parse_args()
    main(args)
