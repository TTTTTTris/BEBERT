# coding=utf-8
# 2020.04.20 - Add&replace quantization modules
#              Huawei Technologies Co., Ltd <zhangwei379@huawei.com>
# Copyright (c) 2020, Huawei Technologies Co., Ltd.  All rights reserved.
# Copyright 2018 The Google AI Language Team Authors and The HuggingFace Inc. team.
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
"""PyTorch BERT model."""

from __future__ import absolute_import, division, print_function, unicode_literals

import pdb
import matplotlib.pyplot as plt
import seaborn as sns
import math
import numpy as np
import sys

import logging
import math
import os

import torch
from torch import nn
from torch.autograd import Variable
from .configuration import BertConfig
from .utils_quant import QuantizeEmbedding, SymQuantizer, BinaryQuantizer, ZMeanBinaryQuantizer, QuantizeLinear
from torch.nn import Parameter

logger = logging.getLogger(__name__)

CONFIG_NAME = "config.json"
WEIGHTS_NAME = "pytorch_model.bin"
cnt_epoch = -1
last_epoch = -1

g_num = 0

def gelu(x):
    """Implementation of the gelu activation function.
        For information: OpenAI GPT's gelu is slightly different (and gives slightly different results):
        0.5 * x * (1 + torch.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * torch.pow(x, 3))))
        Also see https://arxiv.org/abs/1606.08415
    """
    return x * 0.5 * (1.0 + torch.erf(x / math.sqrt(2.0)))


class BertEmbeddings(nn.Module):
    def __init__(self, config):
        super(BertEmbeddings, self).__init__()
        self.word_embeddings = QuantizeEmbedding(config.vocab_size,
                                                 config.hidden_size,
                                                 padding_idx=0,
                                                 config=config,
                                                 type="word_embd")
        self.position_embeddings = nn.Embedding(
            config.max_position_embeddings, config.hidden_size)
        self.token_type_embeddings = nn.Embedding(
            config.type_vocab_size, config.hidden_size)
        
        self.LayerNorm = nn.LayerNorm(config.hidden_size,
                                      eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, input_ids, token_type_ids):
        seq_length = input_ids.size(1)
        position_ids = torch.arange(seq_length,
                                    dtype=torch.long,
                                    device=input_ids.device)
        position_ids = position_ids.unsqueeze(0).expand_as(input_ids)

        words_embeddings = self.word_embeddings(input_ids)
        position_embeddings = self.position_embeddings(position_ids)
        token_type_embeddings = self.token_type_embeddings(token_type_ids)

        embeddings = words_embeddings + position_embeddings + token_type_embeddings
        embeddings = self.LayerNorm(embeddings)
        embeddings = self.dropout(embeddings)

        # embeddings = embeddings.cpu().numpy()
        # import numpy as np
        # noise = np.random.normal(0, 0.01, embeddings.shape).astype(dtype=np.float32)  # add guass noise
        # embeddings += noise
        # embeddings = self.dropout(torch.from_numpy(embeddings).cuda())

        return embeddings


class BertSelfAttention(nn.Module):
    def __init__(self, config):
        super(BertSelfAttention, self).__init__()
        if config.hidden_size % config.num_attention_heads != 0:
            raise ValueError(
                "The hidden size (%d) is not a multiple of the number of attention "
                "heads (%d)" %
                (config.hidden_size, config.num_attention_heads))
        self.num_attention_heads = config.num_attention_heads
        self.attention_head_size = int(
            config.hidden_size / config.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size
        self.quantize_act = config.quantize_act
        self.query = QuantizeLinear(config.hidden_size,
                                    self.all_head_size,
                                    config=config)
        self.key = QuantizeLinear(config.hidden_size,
                                  self.all_head_size,
                                  config=config)
        self.value = QuantizeLinear(config.hidden_size,
                                    self.all_head_size,
                                    config=config)
        prob_mean = None
        if self.quantize_act:
            self.input_bits = config.input_bits
            if self.input_bits == 1:
                self.act_quantizer = BinaryQuantizer
            else:
                self.act_quantizer = SymQuantizer
            self.register_buffer(
                'clip_query', torch.Tensor([-config.clip_val,
                                            config.clip_val]))
            self.register_buffer(
                'clip_key', torch.Tensor([-config.clip_val, config.clip_val]))
            self.register_buffer(
                'clip_value', torch.Tensor([-config.clip_val,
                                            config.clip_val]))
            self.register_buffer(
                'clip_attn', torch.Tensor([-config.clip_val, config.clip_val]))

        self.dropout = nn.Dropout(config.attention_probs_dropout_prob)

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.num_attention_heads,
                                       self.attention_head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(self,
                hidden_states,
                attention_mask,
                output_att=False,
                layer_num=-1):

        mixed_query_layer = self.query(hidden_states)
        mixed_key_layer = self.key(hidden_states)
        mixed_value_layer = self.value(hidden_states)

        query_layer = self.transpose_for_scores(mixed_query_layer)
        key_layer = self.transpose_for_scores(mixed_key_layer)
        value_layer = self.transpose_for_scores(mixed_value_layer)

        query_scores = torch.matmul(query_layer, query_layer.transpose(-1, -2))
        query_scores = query_scores / math.sqrt(self.attention_head_size)

        key_scores = torch.matmul(key_layer, key_layer.transpose(-1, -2))
        key_scores = key_scores / math.sqrt(self.attention_head_size)

        if self.quantize_act and self.input_bits == 1:
            query_layer = self.act_quantizer.apply(query_layer)
            key_layer = self.act_quantizer.apply(key_layer)

        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        attention_scores = attention_scores / math.sqrt(
            self.attention_head_size)
        attention_scores = attention_scores + attention_mask
        attention_probs = self.dropout(attention_scores)

        value_scores = torch.matmul(value_layer, value_layer.transpose(-1, -2))
        value_scores = value_scores / math.sqrt(self.attention_head_size)

        if self.quantize_act: 
            if self.input_bits == 1:
                attention_probs = ZMeanBinaryQuantizer.apply(attention_probs)
                value_layer = self.act_quantizer.apply(value_layer)

        context_layer = torch.matmul(attention_probs, value_layer)
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (
            self.all_head_size, )
        context_layer = context_layer.view(*new_context_layer_shape)
        return context_layer, attention_scores, value_scores, 0, query_scores, key_scores


class BertAttention(nn.Module):
    def __init__(self, config):
        super(BertAttention, self).__init__()
        self.self = BertSelfAttention(config)
        self.output = BertSelfOutput(config)

    def forward(self, input_tensor, attention_mask, layer_num=0):
        self_output, layer_att, value_att, context_score, query_scores, key_scores = self.self(
            input_tensor, attention_mask, layer_num=layer_num)
        attention_output = self.output(self_output,
                                       input_tensor,
                                       layer_num=layer_num)
        return attention_output, layer_att, value_att, context_score, query_scores, key_scores


class BertSelfOutput(nn.Module):
    def __init__(self, config):
        super(BertSelfOutput, self).__init__()
        self.dense = QuantizeLinear(config.hidden_size,
                                    config.hidden_size,
                                    config=config)
        self.LayerNorm = nn.LayerNorm(config.hidden_size,
                                      eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.register_parameter('gate', Parameter(torch.ones(1).squeeze()))

    def forward(self, hidden_states, input_tensor, layer_num=0):
        hidden_states = self.dense(hidden_states,
                                   type="layer" + str(layer_num) + "_dense")
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states


class BertIntermediate(nn.Module):
    def __init__(self, config):
        super(BertIntermediate, self).__init__()
        self.dense = QuantizeLinear(config.hidden_size,
                                    config.intermediate_size,
                                    config=config)

    def forward(self, hidden_states, layer_num=0):
        hidden_states = self.dense(hidden_states,
                                   type="layer" + str(layer_num) + "_dense")
        hidden_states = gelu(hidden_states)
        return hidden_states


class BertOutput(nn.Module):
    def __init__(self, config):
        super(BertOutput, self).__init__()
        self.dense = QuantizeLinear(config.intermediate_size,
                                    config.hidden_size,
                                    config=config)
        self.LayerNorm = nn.LayerNorm(config.hidden_size,
                                      eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.register_parameter('gate', Parameter(torch.ones(1).squeeze()))

    def forward(self, hidden_states, input_tensor, layer_num=0):
        hidden_states = self.dense(hidden_states,
                                   type="layer" + str(layer_num) + "_dense")
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states


class BertLayer(nn.Module):
    def __init__(self, config):
        super(BertLayer, self).__init__()
        self.attention = BertAttention(config)
        self.intermediate = BertIntermediate(config)
        self.output = BertOutput(config)

    def forward(self, hidden_states, attention_mask, layer_num=0):
        attention_output, layer_att, value_att, context_score, query_score, key_score = self.attention(
            hidden_states, attention_mask, layer_num=layer_num)
        intermediate_output = self.intermediate(attention_output,
                                                layer_num=layer_num)
        layer_output = self.output(intermediate_output,
                                   attention_output,
                                   layer_num=layer_num)

        return layer_output, layer_att, value_att, context_score, query_score, key_score


class BertEncoder(nn.Module):
    def __init__(self, config):
        super(BertEncoder, self).__init__()
        self.layer = nn.ModuleList(
            [BertLayer(config) for _ in range(config.num_hidden_layers)])

    def forward(self, hidden_states, attention_mask):
        all_encoder_layers = [hidden_states]
        all_encoder_atts = []
        all_value_atts = []
        all_context_scores = []
        all_query_scores = []
        all_key_scores = []

        for _, layer_module in enumerate(self.layer):
            hidden_states, layer_att, value_att, context_score, query_score, key_score = layer_module(
                hidden_states, attention_mask, layer_num=_)
            all_encoder_layers.append(hidden_states)
            all_encoder_atts.append(layer_att)
            all_value_atts.append(value_att)
            all_context_scores.append(context_score)
            all_query_scores.append(query_score)
            all_key_scores.append(key_score)

        return all_encoder_layers, all_encoder_atts, all_value_atts, all_context_scores, all_query_scores, all_key_scores


class BertPooler(nn.Module):
    def __init__(self, config, recurs=None):
        super(BertPooler, self).__init__()
        self.dense = QuantizeLinear(config.hidden_size,
                                    config.hidden_size,
                                    config=config)
        self.activation = nn.Tanh()
        self.config = config

    def forward(self, hidden_states):
        pooled_output = hidden_states[-1][:, 0]
        pooled_output = self.dense(pooled_output, type="pooler")
        pooled_output = self.activation(pooled_output)

        return pooled_output


class BertPreTrainedModel(nn.Module):
    """ An abstract class to handle weights initialization and
        a simple interface for dowloading and loading pretrained models.
    """
    def __init__(self, config, *inputs, **kwargs):
        super(BertPreTrainedModel, self).__init__()
        self.config = config

    def init_bert_weights(self, module):
        """ Initialize the weights.
        """
        if isinstance(module, (nn.Linear, nn.Embedding)):
            # Slightly different from the TF version which uses truncated_normal for initialization
            # cf https://github.com/pytorch/pytorch/pull/5617
            module.weight.data.normal_(mean=0.0,
                                       std=self.config.initializer_range)
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, *inputs, **kwargs):
        """
        Instantiate a BertPreTrainedModel from a pre-trained model file or a pytorch state dict.
        Params:
            pretrained_model_name_or_path:
                - a path or url to a pretrained model archive containing:
                    . `bert_config.json` a configuration file for the model
                    . `pytorch_model.bin` a PyTorch dump of a BertForPreTraining instance
            state_dict: an optional state dictionnary (collections.OrderedDict object) to use instead of Google pre-trained models
            config: BertConfig instance
            *inputs, **kwargs: additional input for the specific Bert class
                (ex: num_labels for BertForSequenceClassification)
        """
        state_dict = kwargs.get('state_dict', None)
        kwargs.pop('state_dict', None)
        config = kwargs.get('config', None)
        kwargs.pop('config', None)
        if config is None:
            # Load config
            config_file = os.path.join(pretrained_model_name_or_path,
                                       CONFIG_NAME)
            config = BertConfig.from_json_file(config_file)

        logger.info("Model config {}".format(config))
        # Instantiate model.

        model = cls(config, *inputs, **kwargs)
        if state_dict is None:
            weights_path = os.path.join(pretrained_model_name_or_path,
                                        WEIGHTS_NAME)
            logger.info("Loading model {}".format(weights_path))
            state_dict = torch.load(weights_path, map_location='cpu')

        # Load from a PyTorch state_dict
        old_keys = []
        new_keys = []
        for key in state_dict.keys():
            new_key = None
            if 'gamma' in key:
                new_key = key.replace('gamma', 'weight')
            if 'beta' in key:
                new_key = key.replace('beta', 'bias')
            if new_key:
                old_keys.append(key)
                new_keys.append(new_key)
        for old_key, new_key in zip(old_keys, new_keys):
            state_dict[new_key] = state_dict.pop(old_key)

        missing_keys = []
        unexpected_keys = [
'bert.encoder.layer.0.attention.self.query.weight',
'bert.encoder.layer.0.attention.self.query.bias',
'bert.encoder.layer.0.attention.self.query.scale',
'bert.encoder.layer.0.attention.self.key.weight',
'bert.encoder.layer.0.attention.self.key.bias',
'bert.encoder.layer.0.attention.self.key.scale',
'bert.encoder.layer.0.attention.self.value.weight',
'bert.encoder.layer.0.attention.self.value.bias',
'bert.encoder.layer.0.attention.self.value.scale',
'bert.encoder.layer.0.attention.output.gate',
'bert.encoder.layer.0.attention.output.dense.weight',
'bert.encoder.layer.0.attention.output.dense.bias',
'bert.encoder.layer.0.attention.output.dense.scale',
'bert.encoder.layer.0.attention.output.LayerNorm.weight',
'bert.encoder.layer.0.attention.output.LayerNorm.bias',
'bert.encoder.layer.0.intermediate.dense.weight',
'bert.encoder.layer.0.intermediate.dense.bias',
'bert.encoder.layer.0.intermediate.dense.scale',
'bert.encoder.layer.0.output.gate',
'bert.encoder.layer.0.output.dense.weight',
'bert.encoder.layer.0.output.dense.bias',
'bert.encoder.layer.0.output.dense.scale',
'bert.encoder.layer.0.output.LayerNorm.weight',
'bert.encoder.layer.0.output.LayerNorm.bias',
'bert.encoder.layer.1.attention.self.query.weight',
'bert.encoder.layer.1.attention.self.query.bias',
'bert.encoder.layer.1.attention.self.query.scale',
'bert.encoder.layer.1.attention.self.key.weight',
'bert.encoder.layer.1.attention.self.key.bias',
'bert.encoder.layer.1.attention.self.key.scale',
'bert.encoder.layer.1.attention.self.value.weight',
'bert.encoder.layer.1.attention.self.value.bias',
'bert.encoder.layer.1.attention.self.value.scale',
'bert.encoder.layer.1.attention.output.gate',
'bert.encoder.layer.1.attention.output.dense.weight',
'bert.encoder.layer.1.attention.output.dense.bias',
'bert.encoder.layer.1.attention.output.dense.scale',
'bert.encoder.layer.1.attention.output.LayerNorm.weight',
'bert.encoder.layer.1.attention.output.LayerNorm.bias',
'bert.encoder.layer.1.intermediate.dense.weight',
'bert.encoder.layer.1.intermediate.dense.bias',
'bert.encoder.layer.1.intermediate.dense.scale',
'bert.encoder.layer.1.output.gate',
'bert.encoder.layer.1.output.dense.weight',
'bert.encoder.layer.1.output.dense.bias',
'bert.encoder.layer.1.output.dense.scale',
'bert.encoder.layer.1.output.LayerNorm.weight',
'bert.encoder.layer.1.output.LayerNorm.bias',
'bert.encoder.layer.2.attention.self.query.weight',
'bert.encoder.layer.2.attention.self.query.bias',
'bert.encoder.layer.2.attention.self.query.scale',
'bert.encoder.layer.2.attention.self.key.weight',
'bert.encoder.layer.2.attention.self.key.bias',
'bert.encoder.layer.2.attention.self.key.scale',
'bert.encoder.layer.2.attention.self.value.weight',
'bert.encoder.layer.2.attention.self.value.bias',
'bert.encoder.layer.2.attention.self.value.scale',
'bert.encoder.layer.2.attention.output.gate',
'bert.encoder.layer.2.attention.output.dense.weight',
'bert.encoder.layer.2.attention.output.dense.bias',
'bert.encoder.layer.2.attention.output.dense.scale',
'bert.encoder.layer.2.attention.output.LayerNorm.weight',
'bert.encoder.layer.2.attention.output.LayerNorm.bias',
'bert.encoder.layer.2.intermediate.dense.weight',
'bert.encoder.layer.2.intermediate.dense.bias',
'bert.encoder.layer.2.intermediate.dense.scale',
'bert.encoder.layer.2.output.gate',
'bert.encoder.layer.2.output.dense.weight',
'bert.encoder.layer.2.output.dense.bias',
'bert.encoder.layer.2.output.dense.scale',
'bert.encoder.layer.2.output.LayerNorm.weight',
'bert.encoder.layer.2.output.LayerNorm.bias',
'bert.encoder.layer.3.attention.self.query.weight',
'bert.encoder.layer.3.attention.self.query.bias',
'bert.encoder.layer.3.attention.self.query.scale',
'bert.encoder.layer.3.attention.self.key.weight',
'bert.encoder.layer.3.attention.self.key.bias',
'bert.encoder.layer.3.attention.self.key.scale',
'bert.encoder.layer.3.attention.self.value.weight',
'bert.encoder.layer.3.attention.self.value.bias',
'bert.encoder.layer.3.attention.self.value.scale',
'bert.encoder.layer.3.attention.output.gate',
'bert.encoder.layer.3.attention.output.dense.weight',
'bert.encoder.layer.3.attention.output.dense.bias',
'bert.encoder.layer.3.attention.output.dense.scale',
'bert.encoder.layer.3.attention.output.LayerNorm.weight',
'bert.encoder.layer.3.attention.output.LayerNorm.bias',
'bert.encoder.layer.3.intermediate.dense.weight',
'bert.encoder.layer.3.intermediate.dense.bias',
'bert.encoder.layer.3.intermediate.dense.scale',
'bert.encoder.layer.3.output.gate',
'bert.encoder.layer.3.output.dense.weight',
'bert.encoder.layer.3.output.dense.bias',
'bert.encoder.layer.3.output.dense.scale',
'bert.encoder.layer.3.output.LayerNorm.weight',
'bert.encoder.layer.3.output.LayerNorm.bias',
'bert.encoder.layer.4.attention.self.query.weight',
'bert.encoder.layer.4.attention.self.query.bias',
'bert.encoder.layer.4.attention.self.query.scale',
'bert.encoder.layer.4.attention.self.key.weight',
'bert.encoder.layer.4.attention.self.key.bias',
'bert.encoder.layer.4.attention.self.key.scale',
'bert.encoder.layer.4.attention.self.value.weight',
'bert.encoder.layer.4.attention.self.value.bias',
'bert.encoder.layer.4.attention.self.value.scale',
'bert.encoder.layer.4.attention.output.gate',
'bert.encoder.layer.4.attention.output.dense.weight',
'bert.encoder.layer.4.attention.output.dense.bias',
'bert.encoder.layer.4.attention.output.dense.scale',
'bert.encoder.layer.4.attention.output.LayerNorm.weight',
'bert.encoder.layer.4.attention.output.LayerNorm.bias',
'bert.encoder.layer.4.intermediate.dense.weight',
'bert.encoder.layer.4.intermediate.dense.bias',
'bert.encoder.layer.4.intermediate.dense.scale',
'bert.encoder.layer.4.output.gate',
'bert.encoder.layer.4.output.dense.weight',
'bert.encoder.layer.4.output.dense.bias',
'bert.encoder.layer.4.output.dense.scale',
'bert.encoder.layer.4.output.LayerNorm.weight',
'bert.encoder.layer.4.output.LayerNorm.bias',
'bert.encoder.layer.5.attention.self.query.weight',
'bert.encoder.layer.5.attention.self.query.bias',
'bert.encoder.layer.5.attention.self.query.scale',
'bert.encoder.layer.5.attention.self.key.weight',
'bert.encoder.layer.5.attention.self.key.bias',
'bert.encoder.layer.5.attention.self.key.scale',
'bert.encoder.layer.5.attention.self.value.weight',
'bert.encoder.layer.5.attention.self.value.bias',
'bert.encoder.layer.5.attention.self.value.scale',
'bert.encoder.layer.5.attention.output.gate',
'bert.encoder.layer.5.attention.output.dense.weight',
'bert.encoder.layer.5.attention.output.dense.bias',
'bert.encoder.layer.5.attention.output.dense.scale',
'bert.encoder.layer.5.attention.output.LayerNorm.weight',
'bert.encoder.layer.5.attention.output.LayerNorm.bias',
'bert.encoder.layer.5.intermediate.dense.weight',
'bert.encoder.layer.5.intermediate.dense.bias',
'bert.encoder.layer.5.intermediate.dense.scale',
'bert.encoder.layer.5.output.gate',
'bert.encoder.layer.5.output.dense.weight',
'bert.encoder.layer.5.output.dense.bias',
'bert.encoder.layer.5.output.dense.scale',
'bert.encoder.layer.5.output.LayerNorm.weight',
'bert.encoder.layer.5.output.LayerNorm.bias',
'bert.encoder.layer.6.attention.self.query.weight',
'bert.encoder.layer.6.attention.self.query.bias',
'bert.encoder.layer.6.attention.self.query.scale',
'bert.encoder.layer.6.attention.self.key.weight',
'bert.encoder.layer.6.attention.self.key.bias',
'bert.encoder.layer.6.attention.self.key.scale',
'bert.encoder.layer.6.attention.self.value.weight',
'bert.encoder.layer.6.attention.self.value.bias',
'bert.encoder.layer.6.attention.self.value.scale',
'bert.encoder.layer.6.attention.output.gate',
'bert.encoder.layer.6.attention.output.dense.weight',
'bert.encoder.layer.6.attention.output.dense.bias',
'bert.encoder.layer.6.attention.output.dense.scale',
'bert.encoder.layer.6.attention.output.LayerNorm.weight',
'bert.encoder.layer.6.attention.output.LayerNorm.bias',
'bert.encoder.layer.6.intermediate.dense.weight',
'bert.encoder.layer.6.intermediate.dense.bias',
'bert.encoder.layer.6.intermediate.dense.scale',
'bert.encoder.layer.6.output.gate',
'bert.encoder.layer.6.output.dense.weight',
'bert.encoder.layer.6.output.dense.bias',
'bert.encoder.layer.6.output.dense.scale',
'bert.encoder.layer.6.output.LayerNorm.weight',
'bert.encoder.layer.6.output.LayerNorm.bias',
'bert.encoder.layer.7.attention.self.query.weight',
'bert.encoder.layer.7.attention.self.query.bias',
'bert.encoder.layer.7.attention.self.query.scale',
'bert.encoder.layer.7.attention.self.key.weight',
'bert.encoder.layer.7.attention.self.key.bias',
'bert.encoder.layer.7.attention.self.key.scale',
'bert.encoder.layer.7.attention.self.value.weight',
'bert.encoder.layer.7.attention.self.value.bias',
'bert.encoder.layer.7.attention.self.value.scale',
'bert.encoder.layer.7.attention.output.gate',
'bert.encoder.layer.7.attention.output.dense.weight',
'bert.encoder.layer.7.attention.output.dense.bias',
'bert.encoder.layer.7.attention.output.dense.scale',
'bert.encoder.layer.7.attention.output.LayerNorm.weight',
'bert.encoder.layer.7.attention.output.LayerNorm.bias',
'bert.encoder.layer.7.intermediate.dense.weight',
'bert.encoder.layer.7.intermediate.dense.bias',
'bert.encoder.layer.7.intermediate.dense.scale',
'bert.encoder.layer.7.output.gate',
'bert.encoder.layer.7.output.dense.weight',
'bert.encoder.layer.7.output.dense.bias',
'bert.encoder.layer.7.output.dense.scale',
'bert.encoder.layer.7.output.LayerNorm.weight',
'bert.encoder.layer.7.output.LayerNorm.bias',
'bert.encoder.layer.8.attention.self.query.weight',
'bert.encoder.layer.8.attention.self.query.bias',
'bert.encoder.layer.8.attention.self.query.scale',
'bert.encoder.layer.8.attention.self.key.weight',
'bert.encoder.layer.8.attention.self.key.bias',
'bert.encoder.layer.8.attention.self.key.scale',
'bert.encoder.layer.8.attention.self.value.weight',
'bert.encoder.layer.8.attention.self.value.bias',
'bert.encoder.layer.8.attention.self.value.scale',
'bert.encoder.layer.8.attention.output.gate',
'bert.encoder.layer.8.attention.output.dense.weight',
'bert.encoder.layer.8.attention.output.dense.bias',
'bert.encoder.layer.8.attention.output.dense.scale',
'bert.encoder.layer.8.attention.output.LayerNorm.weight',
'bert.encoder.layer.8.attention.output.LayerNorm.bias',
'bert.encoder.layer.8.intermediate.dense.weight',
'bert.encoder.layer.8.intermediate.dense.bias',
'bert.encoder.layer.8.intermediate.dense.scale',
'bert.encoder.layer.8.output.gate',
'bert.encoder.layer.8.output.dense.weight',
'bert.encoder.layer.8.output.dense.bias',
'bert.encoder.layer.8.output.dense.scale',
'bert.encoder.layer.8.output.LayerNorm.weight',
'bert.encoder.layer.8.output.LayerNorm.bias',
'bert.encoder.layer.9.attention.self.query.weight',
'bert.encoder.layer.9.attention.self.query.bias',
'bert.encoder.layer.9.attention.self.query.scale',
'bert.encoder.layer.9.attention.self.key.weight',
'bert.encoder.layer.9.attention.self.key.bias',
'bert.encoder.layer.9.attention.self.key.scale',
'bert.encoder.layer.9.attention.self.value.weight',
'bert.encoder.layer.9.attention.self.value.bias',
'bert.encoder.layer.9.attention.self.value.scale',
'bert.encoder.layer.9.attention.output.gate',
'bert.encoder.layer.9.attention.output.dense.weight',
'bert.encoder.layer.9.attention.output.dense.bias',
'bert.encoder.layer.9.attention.output.dense.scale',
'bert.encoder.layer.9.attention.output.LayerNorm.weight',
'bert.encoder.layer.9.attention.output.LayerNorm.bias',
'bert.encoder.layer.9.intermediate.dense.weight',
'bert.encoder.layer.9.intermediate.dense.bias',
'bert.encoder.layer.9.intermediate.dense.scale',
'bert.encoder.layer.9.output.gate',
'bert.encoder.layer.9.output.dense.weight',
'bert.encoder.layer.9.output.dense.bias',
'bert.encoder.layer.9.output.dense.scale',
'bert.encoder.layer.9.output.LayerNorm.weight',
'bert.encoder.layer.9.output.LayerNorm.bias',
'bert.encoder.layer.10.attention.self.query.weight',
'bert.encoder.layer.10.attention.self.query.bias',
'bert.encoder.layer.10.attention.self.query.scale',
'bert.encoder.layer.10.attention.self.key.weight',
'bert.encoder.layer.10.attention.self.key.bias',
'bert.encoder.layer.10.attention.self.key.scale',
'bert.encoder.layer.10.attention.self.value.weight',
'bert.encoder.layer.10.attention.self.value.bias',
'bert.encoder.layer.10.attention.self.value.scale',
'bert.encoder.layer.10.attention.output.gate',
'bert.encoder.layer.10.attention.output.dense.weight',
'bert.encoder.layer.10.attention.output.dense.bias',
'bert.encoder.layer.10.attention.output.dense.scale',
'bert.encoder.layer.10.attention.output.LayerNorm.weight',
'bert.encoder.layer.10.attention.output.LayerNorm.bias',
'bert.encoder.layer.10.intermediate.dense.weight',
'bert.encoder.layer.10.intermediate.dense.bias',
'bert.encoder.layer.10.intermediate.dense.scale',
'bert.encoder.layer.10.output.gate',
'bert.encoder.layer.10.output.dense.weight',
'bert.encoder.layer.10.output.dense.bias',
'bert.encoder.layer.10.output.dense.scale',
'bert.encoder.layer.10.output.LayerNorm.weight',
'bert.encoder.layer.10.output.LayerNorm.bias',
'bert.encoder.layer.11.attention.self.query.weight',
'bert.encoder.layer.11.attention.self.query.bias',
'bert.encoder.layer.11.attention.self.query.scale',
'bert.encoder.layer.11.attention.self.key.weight',
'bert.encoder.layer.11.attention.self.key.bias',
'bert.encoder.layer.11.attention.self.key.scale',
'bert.encoder.layer.11.attention.self.value.weight',
'bert.encoder.layer.11.attention.self.value.bias',
'bert.encoder.layer.11.attention.self.value.scale',
'bert.encoder.layer.11.attention.output.gate',
'bert.encoder.layer.11.attention.output.dense.weight',
'bert.encoder.layer.11.attention.output.dense.bias',
'bert.encoder.layer.11.attention.output.dense.scale',
'bert.encoder.layer.11.attention.output.LayerNorm.weight',
'bert.encoder.layer.11.attention.output.LayerNorm.bias',
'bert.encoder.layer.11.intermediate.dense.weight',
'bert.encoder.layer.11.intermediate.dense.bias',
'bert.encoder.layer.11.intermediate.dense.scale',
'bert.encoder.layer.11.output.gate',
'bert.encoder.layer.11.output.dense.weight',
'bert.encoder.layer.11.output.dense.bias',
'bert.encoder.layer.11.output.dense.scale',
'bert.encoder.layer.11.output.LayerNorm.weight',
'bert.encoder.layer.11.output.LayerNorm.bias',
'bert.pooler.dense.weight',
'bert.pooler.dense.bias',
'bert.pooler.dense.scale']
        error_msgs = []
        metadata = getattr(state_dict, '_metadata', None)
        state_dict = state_dict.copy()
        if metadata is not None:
            state_dict._metadata = metadata

        def load(module, prefix=''):
            local_metadata = {} if metadata is None else metadata.get(
                prefix[:-1], {})
            module._load_from_state_dict(state_dict, prefix, local_metadata,
                                         True, missing_keys, unexpected_keys,
                                         error_msgs)
            for name, child in module._modules.items():
                if child is not None:
                    load(child, prefix + name + '.')

        start_prefix = ''
        if not hasattr(model, 'bert') and any(
                s.startswith('bert.') for s in state_dict.keys()):
            start_prefix = 'bert.'

        logger.info('loading model...')
        load(model, prefix=start_prefix)
        logger.info('done!')
        if len(missing_keys) > 0:
            logger.info(
                "Weights of {} not initialized from pretrained model: {}".
                format(model.__class__.__name__, missing_keys))
        if len(unexpected_keys) > 0:
            logger.info(
                "Weights from pretrained model not used in {}: {}".format(
                    model.__class__.__name__, unexpected_keys))
        if len(error_msgs) > 0:
            raise RuntimeError(
                'Error(s) in loading state_dict for {}:\n\t{}'.format(
                    model.__class__.__name__, "\n\t".join(error_msgs)))
                    
        return model


class BertModel(BertPreTrainedModel):
    def __init__(self, config):
        super(BertModel, self).__init__(config)
        self.embeddings = BertEmbeddings(config)
        self.encoder = BertEncoder(config)
        self.pooler = BertPooler(config)
        self.apply(self.init_bert_weights)

    def forward(self, input_ids, token_type_ids=None, attention_mask=None):
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids)
        if token_type_ids is None:
            token_type_ids = torch.zeros_like(input_ids)

        extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)
        # print(next(self.parameters()).dtype)
        # print(list(self.parameters()))
        extended_attention_mask = extended_attention_mask.to(
            dtype=next(self.parameters()).dtype)  # fp16 compatibility
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0

        embedding_output = self.embeddings(input_ids, token_type_ids)
        encoded_layers, attention_scores, value_scores, context_scores, query_scores, key_scores = self.encoder(
            embedding_output, extended_attention_mask)

        pooled_output = self.pooler(encoded_layers)
        return encoded_layers, attention_scores, pooled_output, value_scores, context_scores, query_scores, key_scores


class BertForSequenceClassification(BertPreTrainedModel):
    def __init__(self, config, num_labels=2, fit_size=768):
        super(BertForSequenceClassification, self).__init__(config)
        self.num_labels = num_labels
        self.bert = BertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, num_labels)
        # self.classifier = QuantizeLinear(config.hidden_size, num_labels, config=config)
        # self.fit_dense = nn.Linear(config.hidden_size, fit_size)
        # self.fit_dense = QuantizeLinear(config.hidden_size, fit_size, config=config)
        self.apply(self.init_bert_weights)

    def forward(self,
                input_ids,
                token_type_ids=None,
                attention_mask=None,
                labels=None,
                is_student=False,
                epoch=-1):
        global cnt_epoch
        cnt_epoch = epoch
        # import pdb
        # pdb.set_trace()
        # print(list(self.parameters()))
        encoded_layers, attention_scores, pooled_output, value_scores, context_scores, query_scores, key_scores = self.bert(
            input_ids, token_type_ids, attention_mask)
        pooled_output = self.dropout(pooled_output)  # tinybert 是 torch.relu()
        logits = self.classifier(pooled_output)

        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            return loss, attention_scores, encoded_layers, value_scores, context_scores, query_scores, key_scores
        else:
            # tmp = []
            # if is_student:
            #     for s_id, encoded_layer in enumerate(encoded_layers):
            #         tmp.append(self.fit_dense(encoded_layer))
            #     encoded_layers = tmp
            return logits, attention_scores, encoded_layers, value_scores, context_scores, query_scores, key_scores  # student_logits, student_atts, student_reps, student_value_atts


class BertForQuestionAnswering(BertPreTrainedModel):
    def __init__(self, config):
        super(BertForQuestionAnswering, self).__init__(config)
        self.bert = BertModel(config)
        self.qa_outputs = nn.Linear(config.hidden_size, 2)
        self.apply(self.init_bert_weights)

    def forward(self,
                input_ids,
                token_type_ids=None,
                attention_mask=None,
                start_positions=None,
                end_positions=None):
        sequence_output, att_output, pooled_output, value_scores = self.bert(
            input_ids, token_type_ids, attention_mask)

        last_sequence_output = sequence_output[-1]

        logits = self.qa_outputs(last_sequence_output)
        start_logits, end_logits = logits.split(1, dim=-1)
        start_logits = start_logits.squeeze(-1)
        end_logits = end_logits.squeeze(-1)

        logits = (start_logits, end_logits)
        if start_positions is not None and end_positions is not None:
            # If we are on multi-GPU, split add a dimension
            if len(start_positions.size()) > 1:
                start_positions = start_positions.squeeze(-1)
            if len(end_positions.size()) > 1:
                end_positions = end_positions.squeeze(-1)
            # sometimes the start/end positions are outside our model inputs, we ignore these terms
            ignored_index = start_logits.size(1)
            start_positions.clamp_(0, ignored_index)
            end_positions.clamp_(0, ignored_index)

            loss_fct = CrossEntropyLoss(ignore_index=ignored_index)
            start_loss = loss_fct(start_logits, start_positions)
            end_loss = loss_fct(end_logits, end_positions)
            total_loss = (start_loss + end_loss) / 2
            return total_loss, att_output, sequence_output, value_scores

        return logits, att_output, sequence_output, value_scores
