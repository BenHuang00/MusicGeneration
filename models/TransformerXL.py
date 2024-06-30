import os
import sys

import torch
from torch import nn

from .transformerxl.pytorch.mem_transformer import (MemTransformerLM, RelPartialLearnableDecoderLayer,
                                                          RelLearnableDecoderLayer, DecoderLayer, AdaptiveEmbedding)
from .transformerxl.pytorch.utils.proj_adaptive_softmax import ProjectedAdaptiveLogSoftmax
from .transformerxl.pytorch.utils.log_uniform_sampler import LogUniformSampler


class TransformerXL(MemTransformerLM):
    def __init__(self, config):
        super().__init__(
            n_token=config['num_tokens'],
            n_layer=config['n_layer'],
            n_head=config['n_head'],
            d_model=config['d_model'],
            d_head=config['d_head'] // config['n_head'],
            d_inner=config['d_inner'],
            dropout=config['dropout'],
            dropatt=config['dropatt'],
            tie_weight=config['tie_weight'],
            d_embed=config['d_embed'],
            div_val=config['div_val'],
            tie_projs=config['tie_projs'],
            pre_lnorm=config['pre_lnorm'],
            tgt_len=config['tgt_len'],
            ext_len=config['ext_len'],
            mem_len=config['mem_len'],
            cutoffs=[],
            same_length=config['same_length'],
            attn_type=0,
            clamp_len=-1,
            sample_softmax=-1,
        )

        self.mems = tuple()

    def reset_mems(self):
        self.mems = tuple()

    def forward(self, data, *mems):
        # nn.DataParallel does not allow size(0) tensors to be broadcasted.
        # So, have to initialize size(0) mems inside the model forward.
        # Moreover, have to return new_mems to allow nn.DataParallel to piece
        # them together.
        if not mems: mems = self.init_mems()

        tgt_len = data.size(0)
        hidden, new_mems = self._forward(data, mems=mems)

        pred_hid = hidden[-tgt_len:]

        if new_mems is not None:
            self.mems = new_mems

        return pred_hid[:, -1, :]
