import os
import sys

import torch
from torch import nn

from transformerxl.pytorch.mem_transformer import (MemTransformerLM, RelPartialLearnableDecoderLayer,
                                                          RelLearnableDecoderLayer, DecoderLayer, AdaptiveEmbedding)
from transformerxl.pytorch.utils.proj_adaptive_softmax import ProjectedAdaptiveLogSoftmax
from transformerxl.pytorch.utils.log_uniform_sampler import LogUniformSampler


class TransformerXL(MemTransformerLM):
    def __init__(self, config):
        super(TransformerXL, self).__init__()

        self.n_token = config['num_tokens']

        self.n_layer = config['n_layer']
        self.n_head = config['n_head']
        self.d_model = config['d_model']
        self.d_head = config['d_head']
        self.d_inner = config['d_inner']
        self.dropout = config['dropout']
        self.dropatt = config['dropatt']
        self.tie_weight = config['tie_weight']
        self.d_embed = config['d_embed']
        self.div_val = config['div_val']
        self.tie_projs = config['tie_projs']
        self.pre_lnorm = config['pre_lnorm']
        self.tgt_len = config['tgt_len']
        self.ext_len = config['ext_len']
        self.mem_len = config['mem_len']
        self.cutoffs = config['cutoffs']
        self.adapt_inp = config['adapt_inp']
        self.same_length = config['same_length']
        self.attn_type = config['attn_type']
        self.clamp_len = config['clamp_len']
        self.sample_softmax = config['sample_softmax']

        self.layers = nn.ModuleList()
        if self.attn_type == 0:  # the default attention
            for i in range(self.n_layer):
                self.layers.append(
                    RelPartialLearnableDecoderLayer(
                        self.n_head, self.d_model, self.d_head, self.d_inner, self.dropout,
                        tgt_len=self.tgt_len, ext_len=self.ext_len, mem_len=self.mem_len,
                        dropatt=self.dropatt, pre_lnorm=self.pre_lnorm)
                )
        elif self.attn_type == 1:  # learnable embeddings
            for i in range(self.n_layer):
                self.layers.append(
                    RelLearnableDecoderLayer(
                        self.n_head, self.d_model, self.d_head, self.d_inner, self.dropout,
                        tgt_len=self.tgt_len, ext_len=self.ext_len, mem_len=self.mem_len,
                        dropatt=self.dropatt, pre_lnorm=self.pre_lnorm)
                )
        elif self.attn_type in [2, 3]:  # absolute embeddings
            for i in range(self.n_layer):
                self.layers.append(
                    DecoderLayer(
                        self.n_head, self.d_model, self.d_head, self.d_inner, self.dropout,
                        dropatt=self.dropatt, pre_lnorm=self.pre_lnorm)
                )

        self.sample_softmax = self.sample_softmax
        # use sampled softmax
        if self.sample_softmax > 0:
            self.out_layer = nn.Linear(self.d_model, self.n_token)
            if self.tie_weight:
                self.out_layer.weight = self.word_emb.weight
            self.tie_weight = self.tie_weight
            self.sampler = LogUniformSampler(self.n_token, self.sample_softmax)

        # use adaptive softmax (including standard softmax)
        else:
            self.crit = ProjectedAdaptiveLogSoftmax(self.n_token, self.d_embed, self.d_model,
                                                    self.cutoffs, div_val=self.div_val)

            if self.tie_weight:
                for i in range(len(self.crit.out_layers)):
                    self.crit.out_layers[i].weight = self.word_emb.emb_layers[i].weight

            if self.tie_projs:
                for i, tie_proj in enumerate(self.tie_projs):
                    if tie_proj and self.div_val == 1 and self.d_model != self.d_embed:
                        self.crit.out_projs[i] = self.word_emb.emb_projs[0]
                    elif tie_proj and self.div_val != 1:
                        self.crit.out_projs[i] = self.word_emb.emb_projs[i]

        self.same_length = self.same_length
        self.clamp_len = self.clamp_len

        self._create_params()
        self.mems = tuple()

    def reset_mems(self):
        self.mems = tuple()

    def forward(self, data, target, *mems):
        # nn.DataParallel does not allow size(0) tensors to be broadcasted.
        # So, have to initialize size(0) mems inside the model forward.
        # Moreover, have to return new_mems to allow nn.DataParallel to piece
        # them together.
        if not mems: mems = self.init_mems()

        tgt_len = target.size(0)
        hidden, new_mems = self._forward(data, mems=mems)

        pred_hid = hidden[-tgt_len:]

        if new_mems is not None:
            self.mems = new_mems

        return pred_hid[:, -1, :]
