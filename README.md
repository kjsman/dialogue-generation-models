# 안내

* 본 레포지토리의 소유자는 스캐터랩/핑퐁팀과 어떠한 관계도 없습니다.
  * [핑퐁팀의 원래 레포지토리](https://github.com/pingpong-ai/dialogue-generation-models)는 (삭제/비공개로 전환)되었습니다. [GitHub에서 Fork 기능이 작동하는 방식으로 인해](https://docs.github.com/en/github/collaborating-with-issues-and-pull-requests/what-happens-to-forks-when-a-repository-is-deleted-or-changes-visibility), Fork의 Parent가 이 레포지토리로 변경된 것입니다.
* [개인정보와 관련하여 이슈가 되는 점](https://namu.wiki/w/%EC%9D%B4%EB%A3%A8%EB%8B%A4(%EC%9D%B8%EA%B3%B5%EC%A7%80%EB%8A%A5)#s-7.3)을 고려하여, 사전 학습된 모델은 **공개하지 않습니다**.
  * 즉, **이 레포지토리를 Fork/Clone하셔도, 대화 가능한 챗봇을 재현하실 수 없습니다**.
* 다만 연구 자체가 가지는 학술적인 의미를 고려하여, 개인정보가 남아있지 않은 순수한 코드를 배포합니다.
* 아래 내용은 기존의 `README.md`에서 일부 내용을 수정한 것입니다.

# Dialogue Generation Models

## Introduction

* This is a repository of pretrained dialogue generation models (GPT-2 and Meena) of Pingpong, ScatterLab.
* You can refer to our [blog post (archive)](http://web.archive.org/web/20210113023101/https://blog.pingpong.us/generation-model/) for detailed pre-training processes and experiment results.

## Downloads

* You **can't** download the pretrained GPT-2 and Meena models; Original models have been deleted due to privacy concerns.

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

## Notes

### Korean

* 모델의 생성 결과는 학습을 바탕으로 한 예측 결과이며 스캐터랩/핑퐁팀의 의견과 무관합니다.
* 모델의 생성 결과는 가상의 대화 생성 결과이며 사실 여부를 담보하지 않습니다.
* 스캐터랩/핑퐁팀은 공개한 모델의 생성 결과에 대한 책임을 지지 않습니다.
* 본 레포지토리는 모델의 사전 학습 코드를 포함하고 있지 않습니다.
* 모델의 상업적 활용에 대해서는 support@pingpong.us로 문의 부탁드립니다.

### Japanese

* モデルの生成結果は統計的機械学習を用いた予測結果であり、事実とは無関係な発話文が生成される可能性があります。この結果は当社の意思決定や判断を示すものではありません。
* 当社は、公開したモデルの使用によって生じる損失、損害等について、いかなる場合においても一切責任を負いません。
* 本レポジトリにはモデルの事前学習に関するソースコードが含まれておりません。
* モデルの商業的利用に関しては、support@pingpong.usから問い合わせお願いします。

## License

The codes in this repository are distributed under the terms of the [Apache-2.0 License](https://www.apache.org/licenses/LICENSE-2.0).

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
