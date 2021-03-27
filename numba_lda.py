import re
import sys
import numpy as np
import numba

file_name = sys.argv[1]  # one document per line
num_topics = int(sys.argv[2])
num_iterations = int(sys.argv[3])
alpha = float(sys.argv[4])
alpha0 = float(sys.argv[5])
beta = float(sys.argv[6])

def tokenize(line):
    line = line.lower()
    line = re.sub('\W+', ' ', line)
    return line.split()

@numba.jit(nopython=True)
def sample(word_array, doc_array, topic_array, word_topic_counts, 
           doc_topic_counts, topic_counts, rands, probs, beta, alpha, alpha0):
    for i in range(len(topic_array)):
        w = word_array[i]
        d = doc_array[i]
        old_t = topic_array[i]

        word_topic_counts[w, old_t] -= 1
        doc_topic_counts[d, old_t] -= 1
        topic_counts[old_t] -= 1

        for t in range(len(topic_counts)):
            if t == 0:
                a = alpha0
            else:
                a = alpha
            top = ((word_topic_counts[w, t] + beta) 
                   * (doc_topic_counts[d, t] + a))
            bottom = topic_counts[t] + word_topic_counts.shape[0] * beta
            probs[t] = top / bottom

        r = rands[i] * np.sum(probs)
        for t in range(len(topic_counts)):
            r = r - probs[t]
            if r < 0:
                new_t = t
                break

        topic_array[i] = new_t

        word_topic_counts[w, new_t] += 1
        doc_topic_counts[d, new_t] += 1
        topic_counts[new_t] += 1

word2id = {}
id2word = []
word_array = []
doc_array = []

for d, doc in enumerate(open(file_name)):
    words = tokenize(doc)
    for word in words:
        doc_array.append(d)
        if word not in word2id:
            word2id[word] = len(word2id)
            id2word.append(word)
        word_array.append(word2id[word])
    if d > 5e4:
        break

num_words = len(id2word)
num_docs = len(set(doc_array))

print len(word_array), "total words"
print num_words, "unique words"
print num_docs, "documents"

word_array = np.array(word_array)
doc_array = np.array(doc_array)

topic_array = np.random.randint(num_topics, size=len(word_array))

word_topic_counts = np.zeros((num_words, num_topics), dtype='int')
doc_topic_counts = np.zeros((num_docs, num_topics), dtype='int')
topic_counts = np.zeros(num_topics, dtype='int')

for w, d, t in zip(word_array, doc_array, topic_array):
    word_topic_counts[w, t] += 1
    doc_topic_counts[d, t] += 1
    topic_counts[t] += 1

probs = np.zeros(num_topics)
for i in range(num_iterations):
    print "iteration", i
    rands = np.random.random(len(word_array))
    sample(word_array, doc_array, topic_array, word_topic_counts, 
           doc_topic_counts, topic_counts, rands, probs, beta, alpha, alpha0)

for t in range(num_topics):
    counts = word_topic_counts[:, t]
    word_counts = [(id2word[i], c) for (i, c) in enumerate(counts)]
    word_counts.sort(key=lambda (w, c): c, reverse=True)
    print t, ': ', ' '.join('%s(%s)' % (w,c) for (w, c) in word_counts[:15])