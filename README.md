# cLCTM

Latent concept topic model (LCTM), but with contextualized word embeddings. Implementation in Python. Token embeddings are learned with Transformers, the Gibbs sampler is optimized for numba (there is also a pure python Gibbs sampler, but it's slow). Uses Faiss to speed up inference and initialization.

This is a very bares-bones implementation.

To do:

- More details in how it works
- Functions to retrieve topic top tokens, most similar word/concepts
- pyLDAvis
- ...
