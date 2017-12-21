'''
Miscellaneous functions.
'''

import numpy as np
import cPickle as pkl
from nltk.tokenize import wordpunct_tokenize
import parameters as prm
from random import randint
import math
import re
from collections import OrderedDict
from sklearn.decomposition import PCA
from theano import config
from time import time
# only print four decimals on float arrays.
np.set_printoptions(linewidth=150, formatter={'float': lambda x: "{0:0.4f}".format(x)})

# Set the random number generators' seeds for consistency
SEED = 123
np.random.seed(SEED)

def clean(txt):
    '''
    #remove most of Wikipedia and AQUAINT markups, such as '[[', and ']]'.
    '''
    txt = re.sub(r'\|.*?\]\]', '', txt) # remove link anchor

    txt = txt.replace('&amp;', ' ').replace('&lt;',' ').replace('&gt;',' ').replace('&quot;', ' ').replace('\'', ' ').replace('(', ' ').replace(')', ' ').replace('.', ' ').replace('"',' ').replace(',',' ').replace(';',' ').replace(':',' ').replace('<93>', ' ').replace('<98>', ' ').replace('<99>',' ').replace('<9f>',' ').replace('<80>',' ').replace('<82>',' ').replace('<83>', ' ').replace('<84>', ' ').replace('<85>', ' ').replace('<89>', ' ').replace('=', ' ').replace('*', ' ').replace('\n', ' ').replace('!', ' ').replace('-',' ').replace('[[', ' ').replace(']]', ' ')

    return txt


def BOW(words, vocab):
    '''
    Convert a list of words to the BoW representation.
    '''
    bow = {} # BoW densely represented as <vocab word idx: quantity>
    for word in words:
        if word in vocab:
            if vocab[word] not in bow:
                bow[vocab[word]] = 0.
            bow[vocab[word]] += 1.

    bow_v = np.asarray(bow.values())
    sumw = float(bow_v.sum())
    if sumw == 0.:
        sumw = 1.
    bow_v /= sumw

    return [bow.keys(), bow_v]


def BOW2(texts, vocab, dim):
    '''
    Convert a list of texts to the BoW dense representation.
    '''
    out = np.zeros((len(texts), dim), dtype=np.int32)
    mask = np.zeros((len(texts), dim), dtype=np.float32)
    for i, text in enumerate(texts):
        bow = BOW(wordpunct_tokenize(text), vocab)
        out[i,:len(bow[0])] = bow[0]
        mask[i,:len(bow[1])] = bow[1]

    return out, mask


def Word2Vec_encode(texts, wemb):
    
    out = np.zeros((len(texts), prm.dim_emb), dtype=np.float32)
    for i, text in enumerate(texts):
        words = wordpunct_tokenize(text)
        n = 0.
        for word in words:
            if word in wemb:
                out[i,:] += wemb[word]
                n += 1.
        out[i,:] /= max(1.,n)

    return out


def text2idx(texts, vocab, dim, use_mask=False):
    '''
    Convert a list of texts to their corresponding vocabulary indexes.
    '''
    if use_mask:
        out = -np.ones((len(texts), dim), dtype=np.int32)
        mask = np.zeros((len(texts), dim), dtype=np.float32)
    else:
        out = -2 * np.ones((len(texts), dim), dtype=np.int32)

    for i, text in enumerate(texts):
        for j, symbol in enumerate(text[:dim]):
            if symbol in vocab:
                out[i,j] = vocab[symbol]
            else:
                out[i,j] = -1 # for UNKnown symbols

        if use_mask:
            mask[i,:j] = 1.

    if use_mask:
        return out, mask
    else:
        return out



def text2idx2(texts, vocab, dim, use_mask=False):
    '''
    Convert a list of texts to their corresponding vocabulary indexes.
    '''
    
    if use_mask:
        out = -np.ones((len(texts), dim), dtype=np.int32)
        mask = np.zeros((len(texts), dim), dtype=np.float32)
        # this is both padding and unk as -1
    else:
        out = -2 * np.ones((len(texts), dim), dtype=np.int32)
        # this is padding with -2

    out_lst = []
    for i, text in enumerate(texts):
        words = wordpunct_tokenize(text)[:dim]
        print i, text, words

        for j, word in enumerate(words):
            if word in vocab:
                out[i,j] = vocab[word]
            else:
                out[i,j] = -1 # Unknown words

        out_lst.append(words)

        if use_mask:
            mask[i,:j] = 1.

    if use_mask:
        return out, mask, out_lst
    else:
        return out, out_lst


def idx2text(idxs, vocabinv, max_words=-1, char=False, output_unk=True):
    '''
    Convert list of vocabulary indexes to text.
    '''
    out = []
   
    for i in idxs:
      
        if i >= 0:
            out.append(vocabinv[i])
        elif i == -1:
            if output_unk:
                out.append('<UNK>')
        else:
            break

        if max_words > -1:
            if len(out) >= max_words:
                break

    if char:
        return ''.join(out)
    else:
        return ' '.join(out)
    
def idx2text2(idxs, vocabinv, char=False, output_unk=True):
    '''
    Convert list of vocabulary indexes to text.
    '''
    
    out = []
    for i,vocab_indexes in enumerate(idxs):
        text_veci=[]
        for j,index in enumerate(vocab_indexes):
            if index >= 0:
                text_veci.append(vocabinv[index])
            elif i == -1:
                if output_unk:
                    text_veci.append('<UNK>')
                else:
                    text_veci.append('')
            else:
                text_veci.append('')
        out.append(text_veci)
    return out
    
   
            

def n_words(words, vocab):
    '''
    Counts the number of words that have an entry in the vocabulary.
    '''
    c = 0
    for word in words:
        if word in vocab:
            c += 1
    return c


def load_vocab(path, n_words=None):
    t0 = time()
    dic = pkl.load(open(path, "rb"))
    print("Loading pickled vobac in {}".format(time()-t0))
    vocab = {}

    if not n_words:
        n_words = len(dic.keys())

    for i, word in enumerate(dic.keys()[:n_words]):
        vocab[word] = i
    return vocab


def numpy_floatX(data):
    return np.asarray(data, dtype=config.floatX)


def np_floatX(data):
    return np.asarray(data, dtype=config.floatX)


def _slice(_x, n, dim):
    if _x.ndim == 3:
        return _x[:, :, n * dim:(n + 1) * dim]
    return _x[:, n * dim:(n + 1) * dim]


def get_one_random_batch_idx(n, batch_size, shuffle=True):
    idx_list = np.arange(n, dtype="int32")
    
    if batch_size > n:
      batch_size = n
      
    if shuffle:
        np.random.shuffle(idx_list)
        
    idx_list = idx_list[:batch_size]
    return idx_list
          

def get_minibatches_idx(n, minibatch_size, shuffle=False, max_samples=None):
    """
    Used to shuffle the dataset at each iteration.
    """
    idx_list = np.arange(n, dtype="int32")

    if shuffle:
        np.random.shuffle(idx_list)

    if max_samples:
        idx_list = idx_list[:max_samples]
        n = max_samples

    minibatches = []
    minibatch_start = 0
    for i in range(n // minibatch_size):
        minibatches.append(idx_list[minibatch_start:
        minibatch_start + minibatch_size])
        minibatch_start += minibatch_size

    if (minibatch_start != n):
        # Make a minibatch out of what is left
        minibatches.append(idx_list[minibatch_start:])

    return zip(range(len(minibatches)), minibatches)


def zipp(params, tparams):
    """
    When we reload the model. Needed for the GPU stuff.
    """
    for kk, vv in params.iteritems():
        tparams[kk].set_value(vv)


def unzip(zipped):
    """
    When we pickle the model. Needed for the GPU stuff.
    """
    new_params = OrderedDict()
    for kk, vv in zipped.iteritems():
        new_params[kk] = vv.get_value()
    return new_params

def lst2matrix(lst):
    maxdim = len(max(lst, key=len))
    out = -np.ones((len(lst), maxdim), dtype=np.int32)
    for i, item in enumerate(lst):
        out[i, :min(len(item), maxdim)] = item[:maxdim]
    return out

def load_params(path, params):
    pp = np.load(path)
    for kk, vv in params.iteritems():
        if kk in pp:
            if params[kk].shape == pp[kk].shape:
                params[kk] = pp[kk]
            else:
                print 'The shape of layer', kk, params[kk].shape, 'is different from shape of the stored layer with the same name', pp[kk].shape, '.'
        else:
            print '%s is not in the archive' % kk

    return params


def load_wemb(params, vocab):
    wemb = pkl.load(open(prm.wordemb_path, 'rb'))
    dim_emb_orig = wemb.values()[0].shape[0]

    W = 0.01 * np.random.randn(prm.n_words, dim_emb_orig).astype(config.floatX)
    for word, pos in vocab.items():
        if word in wemb:
            W[pos, :] = wemb[word]

    if prm.dim_emb < dim_emb_orig:
        pca = PCA(n_components=prm.dim_emb, copy=False, whiten=True)
        W = pca.fit_transform(W)

    params['W'] = W

    return params


def itemlist(tparams):
    return [vv for kk, vv in tparams.iteritems()]


def init_tparams(params):
    tparams = OrderedDict()
    for kk, pp in params.iteritems():
        tparams[kk] = params[kk]
    return tparams


def ortho_weight(ndim):
    W = np.random.randn(ndim, ndim)
    u, s, v = np.linalg.svd(W)
    return u.astype(config.floatX)


def matrix(dim):
    return np.concatenate([ortho_weight(dim),
                           ortho_weight(dim),
                           ortho_weight(dim),
                           ortho_weight(dim)], axis=1)


def init_params(options):
    params = OrderedDict()
    exclude_params = {}

    params['W'] = 0.01 * np.random.randn(prm.n_words, prm.dim_emb).astype(config.floatX)  # vocab to word embeddings
    params['UNK'] = 0.01 * np.random.randn(1, prm.dim_emb).astype(config.floatX)  # vector for unknown words.

    n_features = [prm.dim_emb, ] + prm.filters_query
    for i in range(len(prm.filters_query)):
        params['Ww_att_q' + str(i)] = 0.01 * np.random.randn(n_features[i + 1], n_features[i], 1,
                                                             prm.window_query[i]).astype(config.floatX)
        params['bw_att_q' + str(i)] = np.zeros((n_features[i + 1],)).astype(config.floatX)  # bias score

    params['Aq'] = 0.01 * np.random.randn(n_features[-1], prm.dim_proj).astype(config.floatX)  # score

    n_hidden_actor = [prm.dim_proj] + prm.n_hidden_actor + [2]
    for i in range(len(n_hidden_actor) - 1):
        params['V' + str(i)] = 0.01 * np.random.randn(n_hidden_actor[i], n_hidden_actor[i + 1]).astype(
            config.floatX)  # score
        params['bV' + str(i)] = np.zeros((n_hidden_actor[i + 1],)).astype(config.floatX)  # bias score

    # set initial bias towards not selecting words.
    params['bV' + str(i)] = np.array([10., 0.]).astype(config.floatX)  # bias score

    n_hidden_critic = [prm.dim_proj] + prm.n_hidden_critic + [1]
    for i in range(len(n_hidden_critic) - 1):
        params['C' + str(i)] = 0.01 * np.random.randn(n_hidden_critic[i], n_hidden_critic[i + 1]).astype(
            config.floatX)  # score
        params['bC' + str(i)] = np.zeros((n_hidden_critic[i + 1],)).astype(config.floatX)  # bias score

    n_features = [prm.dim_emb, ] + prm.filters_cand
    for i in range(len(prm.filters_cand)):
        params['Ww_att_c_0_' + str(i)] = 0.01 * np.random.randn(n_features[i + 1], n_features[i], 1,
                                                                prm.window_cand[i]).astype(config.floatX)
        params['bw_att_c_0_' + str(i)] = np.zeros((n_features[i + 1],)).astype(config.floatX)  # bias score

    params['Ad'] = 0.01 * np.random.randn(n_features[-1], prm.dim_proj).astype(config.floatX)  # score
    params['bAd'] = np.zeros((prm.dim_proj,)).astype(config.floatX)  # bias score

    if prm.fixed_wemb:
        exclude_params['W'] = True

    return params, exclude_params
