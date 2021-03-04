# coding: utf-8

import numpy as np
from tqdm.auto import tqdm

torchtf_avail = True
try:
    import torch
    import torch.nn as nn
    from transformers import AutoTokenizer, AutoModel
except ImportError:
    torchtf_avail = False

gensim_avail = True
try:
    import gensim
except ImportError:
    gensim_avail = False

from random import randrange, shuffle, sample
from itertools import chain
from collections import Counter


class Corpus:
    
    def __init__(
                self,
                langmodel="bert-base-multilingual-cased",   # TODO: maybe change this to reg BERT?
                store_corpus=False,                         # Not yet implemented
                softmax=False

            ):
        """
        langmodel: if it's a str, then the model is loaded using transformers. If it's a gensim KeyedVector
            or a dict, then we're using static word embeddings.

        store_corpus: if True, then the corpus is converted to vectors, which are stored on disk, in a temporary
            folder.

        For the moment, though, static word embeddings are not implemented yet.
        """

        if torchtf_avail and isinstance(langmodel, str):
            self.tokenizer = AutoTokenizer.from_pretrained(langmodel) if tokenizer is None else tokenizer
            self.cvmodel = AutoModel.from_pretrained(langmodel) if vecmodel is None else vecmodel
            self.we_type = "cwe"
        elif gensim_avail and isinstance(langmodel, gensim.models.keyedvectors.KeyedVectors):
            self.we_type = "gensim"
            self.wv = langmodel
        elif isinstance(langmodel, dict):
            self.we_type = "dict"
            self.wv = langmodel

        self.store_corpus=store_corpus
        self.softmax = softmax
        self.input_ids = []

    def __call__(
                self,
                data
            ):
        """
        Appends documents to the corpus.
        
        data can be:
            - a str: then it simply adds it as a single document
            - a list of str: each item is added as a single document
            - a list of int: assume that ints are index_ids; added as a single document
            - a list of lists of int: each list of int is added as a single document
        """

        if isinstance(data, str):
            typ = "str"
            self.input_ids.append(self.tokenizer(data)["input_ids"])
        elif isinstance(data, list) and len(data)>0:
            if isinstance(data[0],str):
                typ = "lostr"
                self.input_ids += self.tokenizer(data)["input_ids"]
            elif isinstance(data[0], int):
                typ = "loint"
                self.input_ids += [ data ]
            elif isinstance(data[0], list) and len(data[0])>0:
                # Obv, first sentence shouldn't be empty cause no sentence should be empty
                # if ever we need to test this (but I suspect it may take time):
                # assert min(len(i) for i in data)>0
                assert isinstance(data[0][0], int)
                typ = "loloint"
                self.input_ids += data

        self.unifs = set(chain(*self.input_ids))

    def n_docs(self):
        return len(self.input_ids)

    def get_doc(self, doc_idx):
        if doc_idx is None: doc_idx = tuple(range(self.n_docs()))

        if isinstance(doc_idx, int):
            return self.cvmodel(torch.tensor([doc]))[0][0].detach().numpy()
        elif isinstance(doc_idx, [list, tuple, set]):
            assert isinstance(doc_idx[0], int)
            r = self.cvmodel(torch.tensor(
                [list(chain(*self.input_ids))],
                [list(chain(*([i]*len(doc) for i, doc in enumerate(self.input_ids))))]
            )[0][0].detach().numpy()

            i0 = 0
            for d in doc_idx:
                yield r[i0:i0+len(self.input_ids[d])]
                i0 += len(self.input_ids[d])
    
    def get_docs(self):
        for r in self.get_doc(range(self.n_docs())):
            yield r

class cLCTM:

    def __init__(
            self,
            n_topics=10,
            concept_vectors=None,
            n_concepts=None,
            n_dims=768,                 # That's the number of dimensions in BERT
            conceptinitmethod="random",
            alpha=0.1,
            beta=0.01,
            faster_heuristic=False      # Not implemented yet
            ):
        self.n_topics = n_topics
        self.n_dims = n_dims
        self.alpha = alpha
        self.beta = beta
        self.n_docs = 0

        # set count variables according to preset concept vectors, if that's what we have
        if concept_vectors is not None and n_concepts is None:
            n_concepts = len(concept_vectors)
        if concept_vectors is not None:
            self.n_dims = len(concept_vectors[0])
        self.n_concepts = n_concepts if n_concepts is not None else self.n_topics*50

        self.init_concepts = concept_vectors
        # if concept_vectors is None, then I'll have to init it when I'm fed the data.
        # Worth mentioning that I do need the data, because word vectors are not bound

    def fit(self, corpus):
        assert isinstance(corpus, Corpus)
        assert corpus.n_docs() > 0

        if self.init_concepts is None:
            self.mk_init_concepts(
        self.init_values(corpus)
    
    def mk_init_concepts(self, corpus, sample_size=0.1, method="random"):
        self.init_wordconcept = dict(zip(sample(unifs, len(unifs)), np.random.randint(0, self.n_concepts)))
        if corpus.softmax:
            self.conceptvecs = np.random.random(self.n_concepts)
        else:
            if isinstance(sample_size, float):
                sample_size = int(corpus.n_docs()*sample_size)
            assert isinstance(sample_size, int)
            assert sample_size>0

            sampled_docids = sample(range(corpus.n_docs()), sample_size)
            sampled_vecs = np.concatenate(tuple(corpus.get_doc(sampled_docids)))

            mu = sampled_vecs.flatten().mean()
            sigma = sampled_vecs.flatten().std()

            self.conceptvecs = np.random.normal(mu, sigma, self.n_concepts)

    def init_values(self, corpus):
        self.n_docs = corpus.n_docs()
        self.topics = []
        self.concepts = []
        self.n_z = Counter()
        self.n_c = Counter()
        self.n_dz = np.zeros((self.n_docs, self.n_topics), dtype=int)
        self.n_zc = np.zeros((self.n_topics, self.n_concepts), dtype=int)
        self.alpha_vec = self.alpha * np.ones(self.n_topics)
        self.n_w = Counter(chain(*corpus.input_ids))
        self.wordconcept = dict(zip(self.n_w.keys(), 
        self.sum_mu_c = np.zeros((self.n_concepts, self.n_dims))

        for d, doc in enumerate(corpus.input_ids):
            # topics = [randrange(self.n_topics) for w in doc]
            topics = np.random.randint(0, self.n_topics, len(doc))
            # concepts = [init_concepts[w] for w in doc]
            concepts = np.vectorize(init_wordconcept.__getitem__)(doc)
            vectors = corpus.get_doc(d)
            self.topics.append(topics)
            self.concepts.append(concepts)

            self.n_z.update(topics)
            self.n_c.update(concepts)

            for z, count in Counter(topics).items():
                self.n_dz[d, z] += count
            for (z, c), count in Counter(zip(topics,concepts)).items():
                self.n_zc[z,c] += count
            for c in set(concepts):
               self.sum_mu_c[c] += vectors[concepts==c,:].sum(0)
        
        self.sum_words = sum(self.n_w.values())

        # Init concepts
        self.mu_c = np.zeros((self.n_dims, self.n_concepts))
        self.sigma_c = np.zeros(self.n_concepts)
        self.mu_c_dot_mu_c = np.zeros(self.n_concepts)

        for c in range(self.n_concepts):
            self.mu_c[c], self.sigma_c[c] = self.calc_mu_sigma(c)
            self.mu_c_dot_mu_c = np.inner(self.mu_c[c], self.mu_c[c])

    def calc_mu_sigma(self, concept_idx):
        z = concept_idx
        var_inverse = self.noise/self.n_z[z] + 1/self.sigma_prior
        sigma = noise + 1/var_inverse

        c1 = self.n_z[z] + self.noise/self.sigma_prior
        c2 = 1 + self.n_z[z] * (self.sigma_prior/self.noise)
        mu = self.sum_mu_c[concept_id]/c1 + self.mu_prior/c2

        return mu, sigma

    def set_wv_priors(self):
        

