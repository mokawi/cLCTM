# coding: utf-8

import numpy as np
from scipy.spatial.distance import cdist
from scipy.special import softmax
from scipy.stats import rankdata
import tqdm.auto as tqdm
import datetime as dt
from operator import sub as substract
import time
import random
from gibbssampler import *

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

try:
    from numba import jit
    from numba.typed import List
    numba_avail=True
except:
    numba_avail = False

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

    def pcgweightedchoice(weights):
        N = len(weights)
        avg = sum(weights)/N
        aliases = [(1, None)]*N
        smalls = ((i, w/avg) for i,w in enumerate(weights) if w < avg)
        bigs = ((i, w/avg) for i,w in enumerate(weights) if w >= avg)
        small, big = next(smalls, None), next(bigs, None)
        while big and small:
            aliases[small[0]] = (small[1], big[0])
            big = (big[0], big[1] - (1-small[1]))
            if big[1] < 1:
                small = big
                big = next(bigs, None)
            else:
                small = next(smalls, None)

        def weighted_random():
            i = pcg32bounded(N)
            odds, alias = aliases[i]
            return alias if (r-i) > odds else i

        return weighted_random()

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
        else:
            self.use_cuda = use_cuda
        self.cuda_device = cuda_device if cuda_device is not None else 0
        if self.use_cuda:
            self.cvmodel = self.cvmodel.to(f'cuda:{self.cuda_device}')

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
            sigma_prior=1.0,
            n_iter=1500,
            faster_heuristic=True,
            max_consec=100,
            sampling_neighbors=300,     # For sampling concepts
            profiling=True
            ):
        self.n_topics = n_topics
        self.n_dims = n_dims
        self.alpha = alpha
        self.beta = beta
        self.sigma_prior = sigma_prior
        self.n_docs = 0
        self.noise = noise
        self.nneighbors = sampling_neighbors
        self.faster = faster_heuristic
        self.max_consec = max_consec
        self.n_iter = n_iter

        self.profiling=profiling

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
        global profenabled

        assert isinstance(corpus, Corpus)
        assert corpus.n_docs > 0

        assert self.n_dims == corpus.n_dims

        profenabled = self.profiling

        self._init_values(corpus)
        self._infer(corpus)

    def _init_concept_vectors(self, corpus, sample_size=0.01, method="kmeans++", metric="cosine"):
        """
        Departs from the original algorithm, which assigned words to a concept and deduced concept vectors from its assignments.
        Doing the other way round enables using the kmeans++ heuristic. Maybe faster too.
        """
        #choicefn = pcgchoice if fastrand_avail else np.random.choice
        sampsize = int(len(corpus.input_ids)*sample_size) if isinstance(sample_size, float) and sample_size<1 else sample_size
        samp = corpus.token_vectors[np.random.randint(len(corpus.input_ids), size=sampsize)]

        # Init mu prior, as we already have a sample (boosts time)
        self.mu_prior = samp.mean(0)
        
        if corpus.token_vectors is None:
            # TODO: make it possible to init concepts w/o vectorizing
            raise Exception("Corpus needs to be vectorized (Corpus.vectorize function).")
        assert len(corpus.input_ids) == len(corpus.token_vectors)

        if method == "kmeans++":
            
            if faiss_avail:
                cfg = faiss.GpuIndexFlatConfig()
                cfg.device = 0

                # step 1
                self.concept_index = faiss.GpuIndexFlatL2(faiss.StandardGpuResources(), self.n_dims, cfg)
                cvs = [random.randint(0,sampsize)]
                self.concept_index.add(samp[cvs[0]:cvs[0]+1])

                for i in tqdm.trange(1, self.n_concepts, desc="Kmeans++ initialization (with faiss)"):
                    #step 2
                    D, _ = self.concept_index.search(samp, 1)

                    #step 3
                    cvs += random.choices(range(sampsize), weights=D.T[0]/D.sum())
                    self.concept_index.add(samp[cvs[-1]:cvs[-1]+1])

                self.concept_vectors = samp[cvs]

            else:
                # step 1
                self.concept_vectors = [random.choice(samp)]
                distances = cdist(self.concept_vectors, samp, metric=metric)

                for i in tqdm.trange(1, self.n_concepts, desc="Kmeans++ initialization"):
                    #step 2 & 3 - note that random.multinomial is 3x faster than random.choice
                    self.concept_vectors = np.concatenate((self.concept_vectors, random.choices(samp, weights=softmax(distances.min(0)**2)).argmax()))
                    distances = np.concatenate((distances, cdist([self.concept_vectors[-1]], samp, metric=metric)))

        else:
            self.concept_vectors = np.random.choice(samp, size=self.n_concepts)

    def _init_values(self, corpus):
        def kv2array(keys, values, size=None, dtype=None):
            if size is None: size=max(keys)+1
            if dtype is None:
                if isinstance(values, np.ndarray):
                    dtype = values.dtype
                else:
                    dtype = type(values[0])

            r = np.zeros(size, dtype=dtype)
            r[keys] = values
            return r

        self.n_docs = corpus.n_docs
        self.max_doclen = corpus.cvmodel.config.max_position_embeddings
        self.alpha_vec = self.alpha * np.ones(self.n_topics)

        self.topics = np.random.randint(0, self.n_topics, len(corpus.input_ids))
        self._init_concept_vectors(corpus, sample_size=int(min(len(corpus.input_ids),max(100, min(1000000, 0.01*len(corpus.input_ids))))))

        #NB: faiss is actually much faster (x18!) than cdist here
        t0 = dt.datetime.now()
        if faiss_avail:
            _, c = self.concept_index.search(corpus.token_vectors, 1)
            self.concepts = c.T[0]
        else:
            self.concepts = cdist(corpus.token_vectors, self.concept_vectors).argmin(axis=1)
        t1 = dt.datetime.now()
        print(f"Concept assignments calculations took {t1-t0}")

        self.n_z = kv2array(*np.unique(self.topics, return_counts=True), size=self.n_topics)
        self.n_c = kv2array(*np.unique(self.concepts, return_counts=True), size=self.n_concepts)

        self.n_w = dict(zip(*np.unique(corpus.input_ids, return_counts=True)))

        t2 = dt.datetime.now()
        print(f"Counted topics, concepts and word types ({t2-t1})")
        
        self.n_dz = np.concatenate([
            [kv2array(*np.unique(np.array(corpus.doc_ids)[self.topics==z], return_counts=True), size=corpus.n_docs)]
            for z in range(self.n_topics)
        ]).T
        t3 =dt.datetime.now()
        print(f"Counted concepts per document ({t3-t2}).")
        self.n_zc = np.concatenate([
            [kv2array(*np.unique(self.concepts[self.topics==z], return_counts=True), size=self.n_concepts)]
            for z in range(self.n_topics)
        ])
        t4 =dt.datetime.now()
        print(f"Counted concepts per topic ({t4-t3}).")

        self.sum_mu_c = np.concatenate([
            [corpus.token_vectors[self.concepts==c].sum(0)]
            for c in range(self.n_concepts)
        ])
        t5 = dt.datetime.now()
        print(f"Computed sum_mu_c ({t5-t4})")

        # Init concepts
        self.mu_c = self.concept_vectors
        self.sigma_c = np.zeros(self.n_concepts)
        self.mu_c_dot_mu_c = np.zeros(self.n_concepts)

        # From concept assignments, produce concept vectors
        #TODO: Optimize this process, or replace with sigma calc
        for c in range(self.n_concepts):
            self.mu_c[c], self.sigma_c[c] = self._calc_mu_sigma(c)
            self.mu_c_dot_mu_c = np.inner(self.mu_c[c], self.mu_c[c])

        t6 = dt.datetime.now()
        print(f"Computed new vectors from assignments ({t6-t5}). Initialization is done.")

        # Stuff for "faster" heuristic
        if self.faster:
            self.consec_sampled_num = np.zeros(len(corpus.input_ids), dtype=np.uint32)
        else:
            self.consec_sampled_num = None

    def _calc_mu_sigma(self, concept_idx):
        c = concept_idx
        var_inverse = self.noise/self.n_c[c] + 1/self.sigma_prior
        sigma = self.noise + 1/var_inverse

        c1 = self.n_c[c] + self.noise/self.sigma_prior
        c2 = 1 + self.n_c[c] * (self.sigma_prior/self.noise)
        mu = self.sum_mu_c[c]/c1 + self.mu_prior/c2

        return mu, sigma

    def theta(self):
        return (self.n_dz.T + self.alpha_vec)/(self.n_dz.sum(0) + self.alpha_vec.sum())
    
    def phi(self):
        return (self.n_zc + self.beta)/(self.n_zc.sum(0) + self.beta*self.n_concepts)

    def _infer(self, corpus):
        #choicefn = pcgchoice if fastrand_avail else np.random.choice
        
        def softmax_lctm(z):
            num = np.exp(z - z.sum())
            s = num / sum(z)
            return s

        #@timefn
        def ghost_topic(d, z, c):
            self.n_dz[d, z] -= 1
            self.n_zc[z, c] -= 1
            self.n_z[z] -= 1
        
        #@timefn
        def update_topic(d,z,c):
            self.n_dz[d,z] += 1
            self.n_zc[z,c] += 1
            self.n_z[z] +=1

        #@timefn
        def ghost_concept(wvec, c, z):
            self.sum_mu_c[c] -= wvec
            self.n_c[c] -= 1
            self.n_zc[z,c] -=1

            self.mu_c[c], self.sigma_c[c] = self._calc_mu_sigma(c)
            self.mu_c_dot_mu_c = np.inner(self.mu_c[c], self.mu_c[c])

        #@timefn
        def update_concept(wvec, c, z):
            self.sum_mu_c[c] += wvec
            self.n_c[c] += 1
            self.n_zc[z,c] +=1

            self.mu_c[c], self.sigma_c[c] = self._calc_mu_sigma(c)
            self.mu_c_dot_mu_c = np.inner(self.mu_c[c], self.mu_c[c])

        #@timefn
        def sample_z(d, c):
            c1 = (self. n_zc[:,c] + self.beta)/(self.n_z + self.beta * self.n_concepts)
            c2 = self.n_dz[d] + self.alpha_vec
            p = c1*c2
            p = p/p.sum()

            return random.choices(list(range(self.n_topics)), weights=p)[0]

        #@timefn
        def sample_c(i, wvec, z):
            nbindices = self.token_neighbors[i]
            t1 = -0.5 * self.n_dims * np.log(self.sigma_c[nbindices])
            t2 = -(0.5 / self.sigma_c[nbindices]) * (self.mu_c_dot_mu_c - 2 * self.mu_c[nbindices] @ wvec)

            prob = softmax_lctm(np.log(self.n_zc[z, nbindices] + self.beta) + t1 + t2)

            return random.choices(nbindices, weights=prob)[0]

        #@timefn
        def create_neighbor_list():
            if faiss_avail:
                _, self.token_neighbors = self.concept_index.search(corpus.token_vectors, self.nneighbors)
            else:
                self.token_neigbors = rankdata(
                    cdict(corpus.token_vectors, self.concept_vectors),
                    axis=1,
                    method="dense"
                )[:,:self.nneighbors]

        if numba_avail:

            pbg = tqdm.tqdm(total=self.n_iter, desc="Iterations")

            for i in range(0, self.n_iter, 5):
                create_neighbor_list()

                gibbslctm(
                    List(corpus.doc_ids),
                    self.topics,
                    self.concepts,
                    corpus.token_vectors,
                    self.n_z, self.n_c,
                    self.n_dz, self.n_zc,
                    self.sum_mu_c,
                    self.mu_c, self.sigma_c,
                    self.mu_prior, self.sigma_prior, self.noise,
                    self.alpha_vec, self.beta,
                    self.token_neighbors,
                    self.consec_sampled_num,
                    max_consec=self.max_consec,
                    faster=self.faster,
                    n_iter=5
                )

                pbg.update(5)


        else: # no numba

            pbg = tqdm.tqdm(total=self.n_iter, desc="Iterations")
            pbw = tqdm.tqdm(total=len(corpus.input_ids), desc="Words")
       
            for it in range(self.n_iter):
                if it % 5 == 0:
                    create_neighbor_list()
                    
                num_z_changed = 0
                num_c_changed = 0
                num_omit = 0

                pbw.reset()
                
                for w in range(len(corpus.input_ids)):
                    
                    #profprint(f"\n# word {w}")

                    doc = corpus.doc_ids[w]
                    z = self.topics[w]
                    c = self.concepts[w]
                    wvec = corpus.token_vectors[w]
                    
                    # Draw new topic
                    ghost_topic(doc, z, c)
                    z_new = sample_z(doc, c)
                    self.topics[w] = z_new
                    update_topic(doc, z, c)

                    if z_new != z: num_z_changed += 1
                    z = z_new

                    if self.faster and self.consec_sampled_num[w] > self.max_consec:
                        num_omit +=1
                        continue

                    # Draw new concept
                    ghost_concept(wvec, c, z)
                    c_new = sample_c(w, wvec, z)
                    self.concepts[w] = c_new
                    update_concept(wvec, c_new, z)

                    if c != c_new:
                        num_c_changed += 1

                        if self.faster:
                            self.consec_sampled_num[w] = 0
                    elif self.faster:
                        self.consec_sampled_num[w] += 1

                    pbw.update(1) 

            pb.update(1)

