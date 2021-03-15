# coding: utf-8

import numpy as np
import tqdm.auto as tqdm

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

import logging

class Corpus:
    
    def __init__(
                self,
                langmodel="bert-base-multilingual-cased",
                store_corpus=False,                         # Not yet implemented
                softmax=False,
                tokenizer=None,
                vecmodel=None
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
        "Returns the number of documents in the corpus"

        return len(self.input_ids)

    def get_doc(self, doc_idx):
        """
        Returns the vector for document at position <doc_idx>. If `doc_idx` is a list, tuple or
        set of int, then it return the vectors for all those positions. Output is always a
        numpy array.
        """

        if doc_idx is None: doc_idx = tuple(range(self.n_docs()))

        if isinstance(doc_idx, int):
            return self.cvmodel(torch.tensor([doc]))[0][0].detach().numpy()
        elif isinstance(doc_idx, [list, tuple, set]):
            assert isinstance(doc_idx[0], int)
            r = self.cvmodel(torch.tensor(
                [list(chain(*self.input_ids))],
                [list(chain(*([i]*len(doc) for i, doc in enumerate(self.input_ids))))]
            )[0][0].detach().numpy())

            i0 = 0
            for d in doc_idx:
                yield r[i0:i0+len(self.input_ids[d])]
                i0 += len(self.input_ids[d])
    
    def get_docs(self, doc_ids=None):
        """
        Kinda like `get_doc`, but if no parameter is specified, it returns the vectors
        for the whole corpus.
        """
        if doc_ids is None:
            for r in self.get_doc(range(self.n_docs())):
                yield r
        else:
            return self.get_doc(doc_ids)

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
            noise=0.5,
            n_iter=1500,
            faster_heuristic=False,     # Not implemented yet
            max_consec=100,
            sampling_neighbors=300      # For sampling concepts
            ):
        self.n_topics = n_topics
        self.n_dims = n_dims
        self.alpha = alpha
        self.beta = beta
        self.n_docs = 0
        self.noise = noise
        self.nneighbors = sampling_neighbors
        self.faster = faster_heuristic
        self.max_consec = max_consec

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

        self._init_values(corpus)
        self._infer(corpus)

    def _init_values(self, corpus):

        # Initialize a bunch of variables
        self.n_docs = corpus.n_docs()
        self.topics = []
        self.concepts = []
        self.n_z = np.zeros(self.n_topics, dtype=int)
        self.n_c = np.zeros(self.n_concepts, dtype=int)
        self.n_dz = np.zeros((self.n_docs, self.n_topics), dtype=int)
        self.n_zc = np.zeros((self.n_topics, self.n_concepts), dtype=int)
        self.alpha_vec = self.alpha * np.ones(self.n_topics)
        self.n_w = Counter(chain(*corpus.input_ids))
        self.init_wordconcept = dict(zip(self.n_w.keys(), np.random.randint(0, self.n_concepts, len(self.n_w.keys())))) 
        self.sum_mu_c = np.zeros((self.n_concepts, self.n_dims))

        pb = tqdm.tqdm(desc="Initialize assignments", total=corpus.n_docs())

        # Make assignements, and counts that will be used in inference
        for d, doc in enumerate(corpus.input_ids):
            # Assign topics and concepts for this doc
            topics = np.random.randint(0, self.n_topics, len(doc))
            concepts = np.vectorize(self.init_wordconcept.__getitem__)(doc)
            # NB: Would be interesting to have other ways of initializing vectors. e.g. kmeans++
            
            vectors = corpus.get_doc(d)
            self.topics.append(topics)
            self.concepts.append(concepts)
            
            # Counts
            nzc = Counter(topics)
            ncc = Counter(concepts)
            self.n_z[list(nzc.keys())] = list(nzc.values())
            self.n_c[list(ncc.keys())] = list(ncc.values())

            for z, count in nzc.items():
                self.n_dz[d, z] += count
            for (z, c), count in Counter(zip(topics,concepts)).items():
                self.n_zc[z,c] += count
            for c in ncc.keys():
               self.sum_mu_c[c] += vectors[concepts==c,:].sum(0)
            
            pb.update(1)
        
        self.sum_words = sum(self.n_w.values())
        
        # only useful for set_wv_priors, and at that point, sum_mu_c == sum_vectors
        # so using sum_mu_c instead
        # self.sum_vectors = self.sum_mu_c.copy()

        self.set_wv_priors()

        # Init concepts
        self.mu_c = np.zeros((self.n_dims, self.n_concepts))
        self.sigma_c = np.zeros(self.n_concepts)
        self.mu_c_dot_mu_c = np.zeros(self.n_concepts)

        # From concept assignments, produce concept vectors
        for c in tqdm.trange(self.n_concepts, desc="Initialize concept vectors"):
            self.mu_c[c], self.sigma_c[c] = self._calc_mu_sigma(c)
            self.mu_c_dot_mu_c = np.inner(self.mu_c[c], self.mu_c[c])

        # Stuff for "faster" heuristic
        if self.faster:
            self.consec_sampled_num = [
                    [0] * len(d)
                    for d in corpus.input_ids
                ]

    def _calc_mu_sigma(self, concept_idx):
        z = concept_idx
        var_inverse = self.noise/self.n_z[z] + 1/self.sigma_prior
        sigma = self.noise + 1/var_inverse

        c1 = self.n_z[z] + self.noise/self.sigma_prior
        c2 = 1 + self.n_z[z] * (self.sigma_prior/self.noise)
        mu = self.sum_mu_c[concept_id]/c1 + self.mu_prior/c2

        return mu, sigma

    def theta(self):
        return (self.n_dz.T + self.alpha_vec)/(self.n_dz.sum(0) + self.alpha_vec.sum())
    
    def phi(self):
        return (self.n_zc + self.beta)/(self.n_zc.sum(0) + self.beta*self.n_concepts)

    def _set_wv_priors(self):
        # Double check this. Shouldn't it be a different denominator? E.g. number of words per concept.
        #self.mu_prior = self.sum_mu_c / self.sum_words
        # Alternatively, a per-concept count.
        self.mu_prior = self.sum_mu_c / self.n_c

        self.sigma_prior = 1.0 # Yes, this is kinda weird as well.

        #TODO: double check this weird function

    def _infer(self, corpus):

        def ghost_topic(d, z, c):
            self.n_dz[d, z] -= 1
            self.n_zc[z, c] -= 1
            self.n_z[z] -= 1
        
        def update_topic(d,z,c):
            self.n_dz[d,z] += 1
            self.n_zc[z,c] += 1
            self.n_z[z] +=1

        def ghost_concept(w, wvec, c, z):
            self.sum_mu_c[c] -= wvec
            self.n_c[c] -= 1
            self.n_zc[z,c] -=1

            self.mu_c[c], self.sigma_c[c] = self._calc_mu_sigma(c)
            self.mu_c_dot_mu_c = np.inner(self.mu_c[c], self.mu_c[c])

        def update_concept(w, wvec, c, z):
            self.sum_mu_c[c] += wvec
            self.n_c[c] += 1
            self.n_zc[z,c] +=1

            self.mu_c[c], self.sigma_c[c] = self._calc_mu_sigma(c)
            self.mu_c_dot_mu_c = np.inner(self.mu_c[c], self.mu_c[c])

        def sample_z(d, c):
            c1 = (self. n_zc[:,c] + self.beta)/(self.n_z + self.beta * self.n_concepts)
            c2 = self.n_dz[d] + self.alpha_vec
            p = c1*c2
            p = p/p.sum()

            return np.random.choice(list(range(self.n_topics)), p=p)

        def softmax(v):
            r = np.exp(v-v.max())
            # .maxCoeff() effectively seems to be .max()
            return r/r.sum()

        def sample_c(w, wvec, z):
            # NB: orig implementation reduced time here by creating "neighbor lists" for each
            # token. Obviously, cwe change even when tokens are the same, so we can't do that
            # here.
            # TODO: check if it's still faster to pick closest concepts for each word token
            # TODO: Make sure not using softmax is ok.
            # TODO: (Maybe) check if derivation checks out? Pretty weird to me.
            t1 = -0.5 * self.n_dims * np.log(self.sigma_c)
            t2 = -(0.5 / self.sigma_c) * (self.mu_c_dot_mu_c - 2 * self.mu_c.T @ wvec)

            prob = softmax(np.log(self.n_zc[:,c] + self.beta) + t1 + t2)

            return np.random.choice(list(range(self.n_concepts)), p=prob)

        pbg = tqdm.tqdm(total=self.n_iter, desc="Iterations")
        pbdoc = tqdm.tqdm(total=self.n_docs, desc="Documents")
       
        for it in range(self.n_iter):
            num_z_changed = 0
            num_c_changed = 0
            num_omit = 0

            pbdoc.reset()
            
            for doc in range(self.n_docs):
                
                for w, wvec, z, c, i in zip(self.input_ids[doc], corpus.get_doc(doc), self.topics[doc], self.concepts[doc], range(len(self.input_ids[doc]))):
                    assert c>=0 and c<self.n_concepts
                    assert z>=0 and z<self.n_topics
                    
                    # Draw new topic
                    ghost_topic(doc, z, c)
                    z_new = sample_z(doc, c)
                    self.topics[doc, i] = z_new
                    update_topic(doc, z, c)

                    if z_new != z: num_z_changed += 1
                    z = z_new

                    if faster and self.consec_sampled_num[doc][i] > self.max_consec:
                        num_omit +=1
                        continue

                    # Draw new concept
                    ghost_concept(w, wvec, c, z)
                    c_new = sample_c(w, wvec, z)
                    self.concepts[doc, i] = c_new
                    update_concept(w, wvec, c_new, z)

                    if c != c_new:
                        num_c_changed += 1

                        if self.faster:
                            self.consec_sampled_num[doc][i] = 0
                    elif self.faster:
                        self.consec_sampled_num[doc][i] += 1

                pbdoc.update(1) 

            pb.update(1)
