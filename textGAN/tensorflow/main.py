# coding: utf-8


from textGAN import *

from HTMLParser import HTMLParser

from utils import *
from collections import Counter

import string
import re


h = HTMLParser()

x = []
with open('../../data/reddit/train/dialogues.txt') as f:
    for l in f:
        l = l.decode(u'utf-8')
        [x.append(h.unescape(ll.strip())) for ll in l.split('__eou__') if len(ll.strip()) > 0]

        if len(x) >= 500000:
            break

x = list(set(x))

x = [re.sub('[%s]' % string.punctuation, '', s.lower()) for s in x]

vocab = Counter()

for s in x:
    for w in s.split(' '):
        vocab[w] += 1

vocab = [w for w, c in vocab.most_common(5000)]

vocab.append('<unk>')


len(vocab)


ixtoword = {i: x for i, x in enumerate(vocab)}
wordtoix = {w: i for i, w in ixtoword.items()}
len(ixtoword)


x = [[wordtoix[w if w in wordtoix else '<unk>'] for w in s.split(' ')] for s in x if len(s.split(' ')) > 1]


train, val = x[:(len(x) - 1000)], x[-1000:]


print "Train size: %d, Test Size: %d" % (len(train), len(val))

opt = Options()

opt.valid_freq = 100
opt.print_freq = 100
opt.lr = 8e-5
opt.max_epochs = 500
opt.batch_size = 256

opt.n_words = len(vocab)

print dict(opt)
print('Total words: %d' % opt.n_words)

run_model(opt, train, val, ixtoword)
