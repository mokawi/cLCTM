# coding: utf-8

import numpy as np
import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModel

class cLCTM:

    def __init__(
            self,
            n_topics=10,
            n_concepts=None,
            tokenizer=None,
            vecmodel=None,

            ):
        self.n_topics = n_topics
        self.n_concepts = n_concepts if n_concepts is not None else n_topics*50

        self.tokenizer = AutoTokenizer.from_pretrained("bert-base-multilingual-cased") if tokenizer is None else tokenizer
        self.cvmodel = AutoModel.from_pretrained("bert-base-multilingual-cased") if vecmodel is None else vecmodel

        self.cmu, self.stdev = self.cvec_init()

    def 

