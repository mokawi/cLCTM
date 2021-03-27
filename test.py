import clctm

import sqlite3
from random import sample

cxn = sqlite3.connect("/home/mok/corpus/philpapers/philpapers.sqlite")
cs = cxn.cursor()

corpus = dict(sample(cs.execute("select ppid, abstract from records where abstract is not null").fetchall(), 10000))

c = clctm.Corpus()
c(list(corpus.values()))
print("n docs:", c.n_docs())

model = clctm.cLCTM(n_topics=10, n_concepts=300)

model.fit(c)
