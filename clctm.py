# coding: utf-8

import numpy as np
from scipy.spatial.distance import cdist
from scipy.special import softmax
import tqdm.auto as tqdm

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

from random import randrange, shuffle, sample
from itertools import chain
from collections import Counter

import logging

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
        if self.token_vectors:
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
            yield self._get_single_doc(doc_idx)
        else:
            if len(doc_idx)>50 and self.token_vectors is None:
                for d in tqdm.tqdm(doc_idx, desc="Retrieving vectors"):
                    yield self._get_single_doc(d)
            else:
                for d in doc_idx:
                    yield self._get_single_doc(d)
    
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
        assert corpus.n_docs > 0

        assert self.n_dims == corpus.n_dims

        self._init_values(corpus)
        self._infer(corpus)

    def _init_concept_vectors(self, corpus, sample_size=0.01, method="kmeans++", metric="cosine"):
        """
        Departs from the original algorithm, which assigned words to a concept and deduced concept vectors from its assignments.
        Doing the other way round enables using the kmeans++ heuristic. Maybe faster too.
        """
        sampsize = int(len(corpus.input_ids)*sample_size) if isinstance(sample_size, float) and sample_size<1 else sample_size
        
        if corpus.token_vectors is None:
            # TODO: make it possible to init concepts w/o vectorizing
            raise Exception("Corpus needs to be vectorized (Corpus.vectorize function).")
        assert len(corpus.input_ids) == len(corpus.token_vectors)

        if method == "kmeans++":
            if faiss_avail:
                samp = corpus.token_vectors[np.random.choice(len(corpus.input_ids), sampsize)]
                
                # step 1
                distidx = faiss.IndexFlatL2(self.n_dims)
                cvs = [np.random.choice(sampsize)]
                distidx.add(self.concept_vectors[cvs[0]:cvs[0]+1)

                for i in tqdm.trange(1, self.n_concepts, desc="Kmeans++ initialization (with faiss)"):
                    #step 2
                    D, _ = distidx.search(samp, 1)

                    #step 3
                    cvs.append(p.random.choice(sampsize, p=D.T[0]/D.sum()))
                    distidx.add(self.concept_vectors[cvs[-1]:cvs[-1]+1])

                selv.concept_vectors = samp[cvs]

            else:
                samp = corpus.token_vectors[np.random.choice(len(corpus.input_ids), sampsize)]
                
                # step 1
                self.concept_vectors = [samp[np.random.choice(sampsize)]]
                distances = cdist(self.concept_vectors, samp, metric=metric)

                for i in tqdm.trange(1, self.n_concepts, desc="Kmeans++ initialization"):
                    #step 2 & 3
                    self.concept_vectors = np.concatenate((self.concept_vectors, [samp[np.random.choice(sampsize, p=softmax(distances.min(0)**2))]]))
                    distances = np.concatenate((distances, cdist([self.concept_vectors[-1]], samp, metric=metric)))

        else:
            self.concept_vectors = np.random.choice(samp, size=self.n_concepts)

    def _init_values(self, corpus):
        def kv2array(keys, values, size=None, dtype=None):
            if size is None: size=max(keys)
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
        if faiss_avail:
            cidx = faiss.IndexFlatL2(self.n_dims)
            cidx.add(self.concept_vectors)
            c, _ = cidx.search(corpus.token_vectors)
            self.concepts = c.T[0]
        else:
            self.concepts = cdist(corpus.token_vectors, self.concept_vectors).argmin(axis=1)

        self.n_z = kv2array(*np.unique(self.topics, return_counts=True), size=self.n_topics)
        self.n_c = kv2array(*np.unique(self.concepts, return_counts=True), size=self.n_concepts)

        self.n_w = dict(zip(*np.unique(corpus.input_ids, return_counts=True)))

        self.n_dz = np.concatenate((
            [kv2array(*np.unique(np.array(corpus.doc_ids)[self.topics==z], return_counts=True), size=corpus.n_docs)]
            for z in range(self.n_topics)
        )).T
        self.n_zc = np.concatenate((
            [kv2array(*np.unique(self.concepts[self.topics==z]), size=self.n_concepts)]
            for z in range(self.n_topics)
        ))

        self.sum_mu_c = np.concatenate((
            [corpus.token_vectors[self.concept==c].sum(0)]
            for c in range(self.n_concepts)
        ))

    def _init_values_old(self, corpus):
        #TODO: remove this function if the new one works

        # Initialize a bunch of variables
        self.n_docs = corpus.n_docs
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

        pb = tqdm.tqdm(desc="Initialize assignments", total=corpus.n_docs)

        # Make assignements, and counts that will be used in inference
        for d, doc in enumerate(corpus.input_ids):
            # Assign topics and concepts for this doc
            topics = np.random.randint(0, self.n_topics, len(doc))
            concepts = np.vectorize(self.init_wordconcept.__getitem__)(doc)
            # NB: Would be interesting to have other ways of initializing vectors. e.g. kmeans++
            
            vectors = list(corpus.get_doc(d))[0]
            self.topics.append(topics)
            self.concepts.append(concepts)
            
            if len(vectors) != len(concepts):
                print(concepts)
                print(vectors)

                raise Exception(f"Length concept list: {len(concepts)}, vectors: {len(vectors)}")
            
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
               #self.sum_mu_c[c] += sum( vectors[i] for i in np.where(concepts==c)[0] )
            
            pb.update(1)
        
        self.sum_words = sum(self.n_w.values())
        
        # only useful for set_wv_priors, and at that point, sum_mu_c == sum_vectors
        # so using sum_mu_c instead
        # self.sum_vectors = self.sum_mu_c.copy()

        self._set_wv_priors()

        # Init concepts
        self.mu_c = self.concept_vectors
        self.sigma_c = np.zeros(self.n_concepts)
        self.mu_c_dot_mu_c = np.zeros(self.n_concepts)

        # From concept assignments, produce concept vectors
        for c in tqdm.trange(self.n_concepts, desc="Recompute concept vectors from assignments"):
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
        self.mu_prior = self.sum_mu_c / self.n_c.T

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
