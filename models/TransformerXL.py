import os
import sys
import math

import torch
from torch import nn
from typing import TypeVar

from .transformerxl.pytorch.mem_transformer import (MemTransformerLM, RelPartialLearnableDecoderLayer,
                                                          RelLearnableDecoderLayer, DecoderLayer, AdaptiveEmbedding)
from .transformerxl.pytorch.utils.proj_adaptive_softmax import ProjectedAdaptiveLogSoftmax
from .transformerxl.pytorch.utils.log_uniform_sampler import LogUniformSampler

T = TypeVar('T', bound='Module')


class Embeddings(nn.Module):
    def __init__(self, n_token, d_model):
        super(Embeddings, self).__init__()
        self.lut = nn.Embedding(n_token, d_model)
        self.d_model = d_model

    def forward(self, x):
        return self.lut(x) * math.sqrt(self.d_model)


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

        self.word_emb = Embeddings(self.n_token, self.d_model)
        self.linear_proj = nn.Linear(self.d_model, self.n_token)

    def train(self: T, mode: bool = True) -> T:
        r"""Set the module in training mode.

        This has any effect only on certain modules. See documentations of
        particular modules for details of their behaviors in training/evaluation
        mode, if they are affected, e.g. :class:`Dropout`, :class:`BatchNorm`,
        etc.

        Args:
            mode (bool): whether to set training mode (``True``) or evaluation
                         mode (``False``). Default: ``True``.

        Returns:
            Module: self
        """
        if not isinstance(mode, bool):
            raise ValueError("training mode is expected to be boolean")
        self.training = mode
        for module in self.children():
            module.train(mode)
        self.reset_mems()   # Clear memory
        return self

    def eval(self: T) -> T:
        r"""Set the module in evaluation mode.

        This has any effect only on certain modules. See documentations of
        particular modules for details of their behaviors in training/evaluation
        mode, if they are affected, e.g. :class:`Dropout`, :class:`BatchNorm`,
        etc.

        This is equivalent with :meth:`self.train(False) <torch.nn.Module.train>`.

        See :ref:`locally-disable-grad-doc` for a comparison between
        `.eval()` and several similar mechanisms that may be confused with it.

        Returns:
            Module: self
        """
        self.reset_mems()  # Clear memory
        return self.train(False)

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
        predict = self.linear_proj(pred_hid)
        predict = predict[:, -1, :]

        if new_mems is not None:
            self.mems = new_mems

        return predict
