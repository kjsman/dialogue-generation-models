# Dialogue Generation Models

## Introduction

* This is a repository of pretrained dialogue generation models (GPT-2 and Meena) of Pingpong, ScatterLab.
* You can refer to our [blog post](https://blog.pingpong.us/generation-model/) for detailed pre-training processes and experiment results.

## Downloads

* You can download the pretrained GPT-2 and Meena models from the link below.
    - **[ `GPT-base-trained-on-dialogue-data-only-KR` ]()**
        - 한국어 대화 데이터로만 학습한 base size GPT-2
    - **[ `GPT-large-trained-on-dialogue-data-only-KR` ]()**
        - 한국어 대화 데이터로만 학습한 large size GPT-2
    - **[ `GPT-base-trained-on-dialogue-and-wiki-KR` ]()**
        - 한국어 대화 데이터, 위키피디아, 나무위키로 학습한 base size GPT-2
    - **[ `GPT-large-trained-on-dialogue-and-wiki-KR` ]()** (**Recommend**)
        - 한국어 대화 데이터, 위키피디아, 나무위키로 학습한 large size GPT-2
    - **[ `Meena-base-trained-on-filtered-dialogue-data-KR` ]()**
        - 필터링된 한국어 대화 데이터로 학습한 base size Meena
    - **[ `Meena-large-trained-on-filtered-dialogue-data-KR` ]()** (**Recommend**)
        - 필터링된 한국어 대화 데이터로 학습한 large size Meena
    - **[ `Meena-base-trained-on-no-filtered-dialogue-data-KR` ]()**
        - 필터링을 거치지 않은 한국어 대화 데이터로 학습한 base size Meena
    - **[ `Meena-large-trained-on-no-filtered-dialogue-data-KR` ]()**
        - 필터링을 거치지 않은 한국어 대화 데이터로 학습한 large size Meena
    - **[ `Meena-base-trained-on-dialogue-data-JP` ]()**
        - 約5億件の日本語日常会話データで学習したbase sizeのMeena

## Install

``` sh
git clone https://github.com/pingpong-ai/dialogue-generation-models.git
cd dialogue-generation-models
pip install -r requirements.txt
```

## Usage

* GPT 답변 생성 예시

``` sh
PYTHONPATH=. python examples/run_gpt.py --pretrained-model-path $PRETRAINED_MODEL_PATH --model-config-path $MODEL_CONFIG_PATH --tokenizer-model-path $TOKENIZER_MODEL_PATH --decoding-method $DECODING_METHOD
```

* Meena 답변 생성 예시

``` sh
PYTHONPATH=. python examples/run_meena.py --pretrained-model-path $PRETRAINED_MODEL_PATH --model-config-path $MODEL_CONFIG_PATH --tokenizer-model-path $TOKENIZER_MODEL_PATH --decoding-method $DECODING_METHOD
```

* 예시 코드에는 답변의 디코딩 전략으로 Top-p Sampling (Nucleus Sampling), Beam Search을 구현해두었습니다.
    - 두 디코딩 방식에는 정확성 (Sensibleness)과 다양성 (Specificity)의 트레이드 오프가 존재합니다.
    - 답변의 정확성이 요구되는 경우는 Beam Search, 답변의 다양성이 요구되는 경우는 top-p sampling의 p를 높여서 사용하는 것을 권장드립니다.

## Caution

* 본 레포는 모델의 학습 로직을 포함하고 있지 않습니다.
* 공개한 모델은 원 논문에서 제안된 GPT-2 및 Meena 모델과 사이즈 및 구조적으로 일부 차이가 있습니다.
* 본 학습은 대량의 카톡 데이터를 이용한 사전학습만 완료한 상태이기 때문에 실사용을 할 때는 모델을 원하는 목적에 맞게 파인튜닝한 뒤 사용하시는 것을 권장드립니다.
* 모델의 생성 결과는 학습을 바탕으로 한 예측 결과에 불과할 뿐 절대 사실이 아니며 회사의 의견과 무관함을 알려드립니다.
* 본 회사는 공개한 모델의 생성 결과에 대한 책임을 일절 지지 않습니다.
* 본 레포에서 공개한 모델을 상업적으로 이용하고 싶은 경우는 회사로 문의 부탁드립니다.

## License

The pretrained models and the codes in this repository are distributed under the terms of the [Apache-2.0 License](https://www.apache.org/licenses/LICENSE-2.0).

## Citation

If you use our software for research, please cite:

``` bibtex
@misc{pingpong2020dial_gen_models,
  author = {Chaehun Park, Sangwoo Seo, Dawoon Jung},
  title = {dialogue-generation-models},
  year = {2019},
  publisher = {GitHub},
  howpublished = {\url{https://github.com/pingpong-ai/dialogue-generation-models}}
}
```

## References

``` bibtex
@techreport{radford2019gpt2,
    title={Language Models are Unsupervised Multitask Learners},
    author={Alec Radford, Jeffrey Wu, Rewon Child, David Luan, Dario Amodei, Ilya Sutskever},
    institution={OpenAI},
    year={2019}
}
```

``` bibtex
@misc{adiwardana2020meena,
    title={Towards a Human-like Open-Domain Chatbot},
    author={Daniel Adiwardana, Minh-Thang Luong, David R So, Jamie Hall, Noah Fiedel, Romal Thoppilan, Zi Yang, Apoorv Kulshreshtha, Gaurav Nemade, Yifeng Lu},
    year={2020},
    eprint={2001.09977},
    archivePrefix={arXiv},
    primaryClass={cs.CL}
}
```

## Acknowledgments

For training models, we used Cloud TPUs provided by [TensorFlow Research Cloud](https://www.tensorflow.org/tfrc/) program.
