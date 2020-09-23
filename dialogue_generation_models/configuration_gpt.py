# coding=utf-8
# Copyright 2018 The OpenAI Team Authors and HuggingFace Inc. team.
# Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
""" OpenAI GPT-2 configuration """
import json
from typing import Type, TypeVar

from transformers.configuration_utils import PretrainedConfig


T = TypeVar("T")


class GPT2Config(PretrainedConfig):
    """
    This is the configuration class to store the configuration of a :class:`GPT2Model`.

    Args:
        vocab_size (:obj:`int`, optional, defaults to 50257):
            Vocabulary size of the GPT-2 model. Defines the different tokens that
            can be represented by the `inputs_ids` passed to the forward method of :class:`~transformers.GPT2Model`.
        n_positions (:obj:`int`, optional, defaults to 1024):
            The maximum sequence length that this model might ever be used with.
            Typically set this to something large just in case (e.g., 512 or 1024 or 2048).
        n_ctx (:obj:`int`, optional, defaults to 1024):
            Dimensionality of the causal mask (usually same as n_positions).
        n_embd (:obj:`int`, optional, defaults to 768):
            Dimensionality of the embeddings and hidden states.
        n_layer (:obj:`int`, optional, defaults to 12):
            Number of hidden layers in the Transformer encoder.
        n_head (:obj:`int`, optional, defaults to 12):
            Number of attention heads for each attention layer in the Transformer encoder.
        n_inner (:obj:`int`, optional, defaults to None):
            Dimensionality of the inner feed-forward layers. :obj:`None` will set it to 4 times n_embd
        activation_function (:obj:`str`, optional, defaults to 'gelu'):
            Activation function selected in the list ["relu", "swish", "gelu", "tanh", "gelu_new"].
        resid_pdrop (:obj:`float`, optional, defaults to 0.1):
            The dropout probability for all fully connected layers in the embeddings, encoder, and pooler.
        embd_pdrop (:obj:`int`, optional, defaults to 0.1):
            The dropout ratio for the embeddings.
        attn_pdrop (:obj:`float`, optional, defaults to 0.1):
            The dropout ratio for the attention.
        layer_norm_epsilon (:obj:`float`, optional, defaults to 1e-5):
            The epsilon to use in the layer normalization layers
        initializer_range (:obj:`float`, optional, defaults to 0.02):
            The standard deviation of the truncated_normal_initializer for initializing all weight matrices.
        pad_token_id (:obj:`int`, optional, defaults to 0)
            Padding token id.
        unk_token_id (:obj:`int`, optional, defaults to 1)
            Unknown of stream token id.
        bos_token_id (:obj:`int`, optional, defaults to 2)
            Beginning of stream token id.
        eos_token_id (:obj:`int`, optional, defaults to 3)
            End of stream token id.
        sept_token_id (:obj:`int`, optional, defaults to 4)
            Turn separator of stream token id.
    """

    model_type = "gpt2"

    def __init__(
        self,
        vocab_size: int = 32000,
        n_positions: int = 256,
        n_ctx: int = 256,
        n_embd: int = 768,
        n_layer: int = 12,
        n_head: int = 12,
        n_inner: int = 3072,
        resid_pdrop: float = 0.0,
        embd_pdrop: float = 0.0,
        attn_pdrop: float = 0.0,
        layer_norm_epsilon: float = 1e-5,
        initializer_range: float = 0.02,
        pad_token_id: int = 0,
        unk_token_id: int = 1,
        bos_token_id: int = 2,
        eos_token_id: int = 3,
        sept_token_id: int = 4,
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.vocab_size = vocab_size
        self.n_ctx = n_ctx
        self.n_positions = n_positions
        self.n_embd = n_embd
        self.n_layer = n_layer
        self.n_head = n_head
        self.n_inner = n_inner
        self.resid_pdrop = resid_pdrop
        self.embd_pdrop = embd_pdrop
        self.attn_pdrop = attn_pdrop
        self.layer_norm_epsilon = layer_norm_epsilon
        self.initializer_range = initializer_range

        self.pad_token_id = pad_token_id
        self.unk_token_id = unk_token_id
        self.bos_token_id = bos_token_id
        self.eos_token_id = eos_token_id
        self.sept_token_id = sept_token_id

    @property
    def max_position_embeddings(self):
        return self.n_positions

    @property
    def hidden_size(self):
        return self.n_embd

    @property
    def num_attention_heads(self):
        return self.n_head

    @property
    def num_hidden_layers(self):
        return self.n_layer

    @classmethod
    def from_json(cls: Type[T], json_file_path: str, **kwargs) -> T:
        """
        Json으로부터 Config 클래스를 생성합니다.
        """
        with open(json_file_path, "r") as f:
            return cls.from_dict(json.load(f), **kwargs)
