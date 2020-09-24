# Dialogue Generation Models

![Lint and Format Python](https://github.com/pingpong-ai/dialogue-generation-models/workflows/Lint%20and%20Format%20Python/badge.svg)

## Introduction

* This is a repository of pretrained dialogue generation models (GPT-2 and Meena) of Pingpong, ScatterLab.
* You can refer to our [blog post](https://blog.pingpong.us/generation-model/) for detailed pre-training processes and experiment results.

## Downloads

* You can download the pretrained GPT-2 and Meena models from [Release page](https://github.com/pingpong-ai/dialogue-generation-models/releases/tag/v1.0).
    - **`GPT-base-trained-on-dialogue-data-only-KR`**
        - 한국어 대화 데이터로만 학습한 base size GPT-2
    - **`GPT-large-trained-on-dialogue-data-only-KR`**
        - 한국어 대화 데이터로만 학습한 large size GPT-2
    - **`GPT-base-trained-on-dialogue-and-wiki-KR`**
        - 한국어 대화 데이터, 위키피디아, 나무위키로 학습한 base size GPT-2
    - **`GPT-large-trained-on-dialogue-and-wiki-KR`** (**Recommend**)
        - 한국어 대화 데이터, 위키피디아, 나무위키로 학습한 large size GPT-2
    - **`Meena-base-trained-on-filtered-dialogue-data-KR`**
        - 필터링된 한국어 대화 데이터로 학습한 base size Meena
    - **`Meena-large-trained-on-filtered-dialogue-data-KR`** (**Recommend**)
        - 필터링된 한국어 대화 데이터로 학습한 large size Meena
    - **`Meena-base-trained-on-no-filtered-dialogue-data-KR`**
        - 필터링을 거치지 않은 한국어 대화 데이터로 학습한 base size Meena
    - **`Meena-large-trained-on-no-filtered-dialogue-data-KR`**
        - 필터링을 거치지 않은 한국어 대화 데이터로 학습한 large size Meena
    - **`Meena-base-trained-on-dialogue-data-JP`**
        - 約5億件の日本語日常会話データで学習したbase sizeのMeena

## Usage

* GPT

``` sh
PYTHONPATH=. python examples/run_gpt.py \
        --pretrained-model-path $PRETRAINED_MODEL_PATH \
        --model-config-path $MODEL_CONFIG_PATH \
        --tokenizer-model-path $TOKENIZER_MODEL_PATH \
        --decoding-method $DECODING_METHOD
```

* Meena

``` sh
PYTHONPATH=. python examples/run_meena.py \
        --pretrained-model-path $PRETRAINED_MODEL_PATH \
        --model-config-path $MODEL_CONFIG_PATH \
        --tokenizer-model-path $TOKENIZER_MODEL_PATH \
        --decoding-method $DECODING_METHOD
```

* We implement two decoding methods called **Top-p Sampling** and **Beam Search** as examples.
    - There is a trade-off of **Accuracy** (Sensibleness) and **Diversity** (Specificity) between two decoding methods.
    - **Beam Search** is a good choice if you prefer the accuracy of the answer, and **Top-p Sampling** is a good choice if you prefer the diversity of the answer.

## Cautions

### Korean

* 모델의 생성 결과는 통계적 학습을 바탕으로 한 예측 결과에 불과할 뿐, 절대 사실이 아니며 회사의 의견과 무관함을 알려드립니다.
* 본 회사는 공개한 모델의 생성 결과에 대한 책임을 일절 지지 않습니다.
* 본 레포는 모델의 학습 로직을 포함하고 있지 않습니다.
* 공개한 모델은 원 논문에서 제안된 GPT-2 및 Meena 모델과 사이즈 및 구조적으로 일부 차이가 있습니다.
* 공개한 모델은 대량의 카톡 데이터를 이용한 사전학습만 완료한 상태이기 때문에 실사용을 할 때는 모델을 원하는 목적에 맞게 파인튜닝한 뒤 사용하시는 것을 권장드립니다.

### Japanese

* モデルの生成結果は統計的機械学習を用いた予測結果であり、事実とは無関係な発話文が生成される可能性があります。この結果は当社の意思決定・判断を示すものではありません。
* 当社は、公開したモデルの使用によって生じる損失、損害等について、いかなる場合においても一切責任を負いません。
* 本レポジトリにはモデルの事前学習に関するソースコードが含まれておりません。
* 公開したモデルには、オリジナル論文で提案されたGPT-2、Meenaとはサイズやモデルの構造において一部異なる部分が含まれております。
* 公開したモデルは日常会話データを用いた事前学習のみを完了したものであり、実際に利用する場合には目的によって追加学習を行ってから利用することをお勧めします。

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
