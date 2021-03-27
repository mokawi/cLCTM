# coding: utf-8

import numpy as np
from scipy.spatial.distance import cdist
from scipy.special import softmax
from scipy.stats import rankdata
import tqdm.auto as tqdm
import datetime as dt
from operator import sub as substract
import time

torchtf_avail = True
try:
    import torch
    import torch.nn as nn
    from transformers import AutoTokenizer, AutoModel, PreTrainedModel
except ImportError:
    torchtf_avail = False

faiss_avail = True
try:
    import faiss
except ImportError:
    faiss_avail = False

gensim_avail = True
try:
    import gensim
except ImportError:
    gensim_avail = False

try:
    from fastrand import pcg32bounded
    fastrand_avail=True
except:
    fastrand_avail=False

from random import randrange, shuffle, sample
from itertools import chain
from collections import Counter

import logging

profenabled = True
proffile = "clctm.prof.log"
def profprint(msg):
    with open(proffile, "a") as f:
        f.write(msg + "\n")

def timefn(func):
    def wrapper(*arg, **kw):
        t1 = time.time()
        res = func(*arg, **kw)
        t2 = time.time()
        profprint(f"{func.__name__}:\t{t2-t1}")
        return res
    return wrapper

if fastrand_avail:
    def pcgchoice(data, size=1):
        if size == 1:
            if isinstance(data, int):
                return pcg32bounded(data)
            return data[pcg32bounded(len(data))]
        else:
            d = list(range(len(data))) if not isinstance(data, int) else list(range(data))
            idx = [ d.pop(pcg32bounded(len(d))) for i in range(size) ]
            if isinstance(data, (np.ndarray, np.matrix)):
                return data[idx]
            elif isinstance(data, int):
                return idx
            else:
                return [ data[i] for i in idx ]

class Corpus:
    
    def __init__(
                self,
                data=None,
                langmodel="bert-base-multilingual-cased",
                store_corpus=None,                         # Not yet implemented
                softmax=False,
                tokenizer=None,
                vecmodel=None,
                use_cuda=None,
                cuda_device=0
            ):
        """
        langmodel: if it's a str, then the model is loaded using transformers. If it's a gensim KeyedVector
            or a dict, then we're using static word embeddings. Default is Multilingual BERT.

        store_corpus: if True, then the corpus is converted to vectors, which are stored on disk, in a temporary
            folder.

        softmax: Currently not in use. But maybe softmaxing vectors from BERT is useful? Dunno

        For the moment, though, static word embeddings are not implemented yet.
        """
        
        self.tokenizer = tokenizer
        self.cvmodel = vecmodel

        if torchtf_avail and isinstance(langmodel, str):
            self.tokenizer = AutoTokenizer.from_pretrained(langmodel) if tokenizer is None else tokenizer
            self.cvmodel = AutoModel.from_pretrained(langmodel) if vecmodel is None else vecmodel
            self.we_type = "cwe"
            self.n_dims = self.cvmodel.config.hidden_size
        elif gensim_avail and isinstance(langmodel, gensim.models.keyedvectors.KeyedVectors):
            self.we_type = "gensim"
            self.wv = langmodel
            self.n_dims = self.wv.vector_size
        elif isinstance(self.cvmodel, PreTrainedModel):
            self.n_dims = self.cvmodel.config.hidden_size

        self.store_corpus=store_corpus
        self.softmax = softmax
        self.input_ids = []
        self.doc_ids = []
        self.token_vectors = None
        self.doc_rng = {}
        self.n_docs = 0

        if use_cuda is None:
            self.use_cuda = torch.cuda.is_available()
            self.cvmodel = self.cvmodel.to(f'cuda:{cuda_device}')
        else:
            self.use_cuda = use_cuda
        self.cuda_device = cuda_device

        if data is not None:
            self.add_docs(data)

    def _restrict_length(self, tokens, max_seqlen=None):
        if max_seqlen is None:
            max_seqlen = self.cvmodel.config.max_position_embeddings
        if isinstance(tokens[0], int):
            return tokens[:max_seqlen]
        else:
            return [ t[:max_seqlen] for t in tokens ]

    def add_docs(
                self,
                data,
                vectorize = True
            ):
        """
        Appends documents to the corpus.
        
        data can be:
            - a str: then it simply adds it as a single document
            - a list of str: each item is added as a single document
            - a list of int: assume that ints are index_ids; added as a single document
            - a list of lists of int: each list of int is added as a single document
        """

        old_n_docs = self.n_docs

        def do_single(iids):
            self.doc_rng[self.n_docs] = (len(self.input_ids), len(self.input_ids) + len(iids))
            self.input_ids.extend(iids)
            self.doc_ids.extend([self.n_docs]*len(iids))
            self.n_docs += 1

        def do_multiple(iids):
            ncs = np.cumsum([0] + [ len(i) for i in iids ])
            self.doc_rng.update(dict(zip(range(self.n_docs, self.n_docs+len(iids)), zip(ncs[:-1], ncs[1:]))))
            self.input_ids.extend(chain(*iids))
            self.doc_ids.extend(chain(*([self.n_docs+i]*l for i, l in enumerate((len(j) for j in iids)))))
            self.n_docs += len(iids)

        if isinstance(data, str):
            typ = "str"
            iids = self._restrict_length(self.tokenizer(data)["input_ids"])
            do_single(iids)

        elif isinstance(data, list) and len(data)>0:
            if isinstance(data[0],str):
                typ = "lostr"
                iids = self._restrict_length(self.tokenizer(data)["input_ids"])
                do_multiple(iids)

            elif isinstance(data[0], int):
                typ = "loint"
                do_single(data)

            elif isinstance(data[0], list) and len(data[0])>0:
                # Obv, first sentence shouldn't be empty cause no sentence should be empty
                # if ever we need to test this (but I suspect it may take time):
                # assert min(len(i) for i in data)>0
                assert isinstance(data[0][0], int)
                typ = "loloint"
                do_multiple(data)

        self.unifs = set(self.input_ids)
        
        if vectorize:
            self._vectorize_docs(old_n_docs)

    def _get_indices(self, doc_idx):
        if isinstance(doc_idx, int):
            return list(range(*self.doc_rng[doc_idx]))
        else:
            return list(chain(*(range(*self.doc_rng[d]) for d in doc_idx)))

    def _get_mask(self, doc_idx):
        r = np.full(len(self.input_ids), False)
        r[self._get_indices(doc_idx)] = True
        return r

    def get_doc_iids(self, doc_idx):
        return np.array(self.input_ids)[self._get_indices(doc_idx)]

    def _get_single_doc(self, doc_idx):
        if self.token_vectors is not None:
            if self.token_vectors.shape[0] <= doc_idx:
                return self.token_vectors[doc_idx]

        indices = self._get_indices(doc_idx)
        
        t_iids = torch.tensor([np.array(self.input_ids)[indices]])
        t_dids = torch.tensor([np.array(self.doc_ids)[indices]])

        if self.use_cuda:
            t_iids = t_iids.to(f'cuda:{self.cuda_device}')
            t_dids = t_dids.to(f'cuda:{self.cuda_device}')

        r = self.cvmodel(t_iids, t_dids)[0][0].detach()
        
        if self.use_cuda:
            return r.cpu().numpy()
        else:
            return r.numpy()

    def _vectorize_docs(self, start=0):
        tvo = [self.token_vectors] if self.token_vectors else []
        self.token_vectors = np.concatenate(tvo + [
            self._get_single_doc(d) for d in tqdm.trange(start, self.n_docs, desc="Infering token vectors")
        ])

    def vectorize_new(self):
        self._vectorize_docs(len(self.token_vectors))

    def get_doc(self, doc_idx):
        """
        Returns the vector for document at position <doc_idx>. If `doc_idx` is a list, tuple or
        set of int, then it return the vectors for all those positions. Output is always a
        numpy array.
        """

        if doc_idx is None: doc_idx = tuple(range(self.n_docs))

        if isinstance(doc_idx, int):
            return self._get_single_doc(doc_idx)
        else:
            if len(doc_idx)>50 and self.token_vectors is None:
                return [
                    self._get_single_doc(d)
                    for d in tqdm.tqdm(doc_idx, desc="Retrieving vectors")
                ]
            elif self.token_vectors is None:
                return [
                    self._get_single_doc(d)
                    for d in doc_idx
                ]
            else:
                return self.token_vectors[self._get_indices(doc_idx)]
    
    def get_docs(self, doc_ids=None):
        """
        Kinda like `get_doc`, but if no parameter is specified, it returns the vectors
        for the whole corpus.
        """
        if doc_ids is None:
            return self.get_doc(range(len(self.n_docs)))

        else:
            return self.get_doc(doc_ids)


