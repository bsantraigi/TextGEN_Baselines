# coding: utf-8

# In[1]:


from textGAN import *

from HTMLParser import HTMLParser

from utils import *
from collections import Counter

# In[2]:


import string
import re

# In[3]:


h = HTMLParser()

# In[4]:


x = []
with open('../../data/reddit/train/dialogues.txt') as f:
    for l in f:
        l = l.decode(u'utf-8')
        [x.append(h.unescape(ll.strip())) for ll in l.split('__eou__') if len(ll.strip()) > 0]

        if len(x) >= 50000:
            break

x = list(set(x))

# In[5]:


x = [re.sub('[%s]' % string.punctuation, '', s.lower()) for s in x]

# In[6]:


vocab = Counter()

for s in x:
    for w in s.split(' '):
        vocab[w] += 1

# In[7]:


vocab = [w for w, c in vocab.most_common(5000)]

# In[8]:


vocab.append('<unk>')

# In[9]:


len(vocab)

# In[10]:


ixtoword = {i: x for i, x in enumerate(vocab)}
wordtoix = {w: i for i, w in ixtoword.items()}
len(ixtoword)

# In[11]:


x = [[wordtoix[w if w in wordtoix else '<unk>'] for w in s.split(' ')] for s in x if len(s.split(' ')) > 1]

# In[12]:


train, val = x[:(len(x) - 1000)], x[-1000:]

# In[13]:


print len(train), len(val)

opt = Options()

opt.valid_freq = 10
opt.print_freq = 10

# In[14]:


opt.n_words = len(vocab)

# In[ ]:


print dict(opt)
print('Total words: %d' % opt.n_words)

run_model(opt, train, val, ixtoword)

# model_fn = auto_encoder
# ae = learn.Estimator(model_fn=model_fn)
# ae.fit(train, opt , steps=opt.max_epochs)
