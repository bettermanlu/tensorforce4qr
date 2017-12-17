import random
#from gym.utils import seeding
from nltk.tokenize import RegexpTokenizer
import nltk
import corpus_hdf5
import dataset_hdf5
import utils
from time import time
from search import Search
import lucene_search
import os
import numpy as np
import parameters as prm
from tensorforce.environments import Environment
from tensorforce import TensorForceError

# Set the random number generators' seeds for consistency
SEED = 123
np.random.seed(SEED)

MAX_COUNT_STEPS = 20 

class QueryReformulatorEnv(Environment):
    def __init__(self, DATA_DIR, dset, is_train, verbose, reward = 'RECALL'):
        # this method returns simulator, state/action vocabularies, and the maximum number of actions
        n_words = 374000 # 100 words for the vocabulary
        vocab_path = os.path.join(DATA_DIR,'data/D_cbow_pdw_8B.pkl')   # Path to the python dictionary containing the vocabulary.
        wordemb_path = os.path.join(DATA_DIR,'data/D_cbow_pdw_8B.pkl')  # Path to the python dictionary containing the word embeddings.
        dataset_path = os.path.join(DATA_DIR, 'data/msa_dataset.hdf5')  # path to load the hdf5 dataset containing queries and ground-truth documents.
        docs_path = os.path.join(DATA_DIR, 'data/msa_corpus.hdf5')  # Path to load the articles and links.
        docs_path_term = os.path.join(DATA_DIR, 'data/msa_corpus.hdf5')  # Path to load the articles and links.
        ############################
        # Search Engine Parameters #
        ############################
        n_threads = 1  # 20 # number of parallel process that will execute the queries on the search engine.
        index_name = 'index'  # index name for the search engine. Used when engine is 'lucene'.
        index_name_term = 'index_terms'  # index name for the search engine. Used when engine is 'lucene'.
        use_cache = False  # If True, cache (query-retrieved docs) pairs. Watch for memory usage.
        max_terms_per_doc = 15  # Maximum number of candidate terms from each feedback doc. Must be always less than max_words_input .
        #self.batch_size_train=2 # The batch size during training.
        self.vocab = utils.load_vocab(vocab_path, n_words)
        vocabinv = {}
        for k, v in self.vocab.items():
            vocabinv[v] = k
        self.reward = reward
        self.is_train = is_train
        self.search = Search(engine=lucene_search.LuceneSearch(DATA_DIR, self.vocab, n_threads, max_terms_per_doc, index_name, index_name_term, docs_path, docs_path_term, use_cache))

        t0 = time()
        dh5 = dataset_hdf5.DatasetHDF5(dataset_path)
        self.qi = dh5.get_queries(dset)
        self.dt = dh5.get_doc_ids(dset)
        print("Loading queries and docs {}".format(time() - t0))
        self.reset()
        

    def get_samples(self, sample_num = 1, max_words_input = 200):
        if sample_num <= 0:
          sample_num = 1          
        train_index = utils.get_one_random_batch_idx(len(self.qi),sample_num)
         
        input_queries = self.qi
        target_docs = self.dt
        vocab = self.vocab
        engine = self.search.engine
        qi = [utils.clean(input_queries[t].lower()) for t in train_index]
        D_gt_title = [target_docs[t] for t in train_index]

        D_gt_id_lst = []
        for j, t in enumerate(train_index):
            #print("j",j)
            D_gt_id_lst.append([])
            for title in D_gt_title[j]:
                #print("title", title)
                if title in engine.title_id_map:
                    D_gt_id_lst[-1].append(engine.title_id_map[title])
                #else:
                #    print 'ground-truth doc not in train_index:', title

        D_gt_id = utils.lst2matrix(D_gt_id_lst)

        qi_i, qi_lst_ = utils.text2idx2(qi, vocab, max_words_input)
        #print("qi_i", qi_i)
        #print("qi_lst_", qi_lst_)

        qi_lst = []
        for qii_lst in qi_lst_:
            # append empty strings, so the list size becomes <dim>.
            qi_lst.append(qii_lst + max(0, max_words_input - len(qii_lst)) * [''])
        return qi, qi_i, qi_lst, D_gt_id, D_gt_title

    def _seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def __str__(self):
        return "Query Reformulator Env"
        
    def execute(self, actions):
        done = False
        self.counsteps+= 1
        
        qi_i, current_queries, D_gt_id = actions
        print 'current_queries (before calling search.perform) = ',current_queries
        metrics, D_i_, D_id_, D_gt_m_ = self.search.perform(qi_i, D_gt_id, self.is_train, current_queries)
        #print "D_id_", D_id_
        print 'current_queries (after calling search.perform) = ',current_queries 
        i = 0
        #print "ALALALA ", [self.search.engine.id_title_map[d_id] for d_id in D_id_[i]]
        text =  [[self.search.engine.id_title_map[d_id] for d_id in D_id_[i]] for i in range(D_id_.shape[0])]
        expanded_queries = current_queries
        metric_idx = self.search.metrics_map[self.reward.upper()]
        reward = metrics[:,metric_idx]
        if self.counsteps > MAX_COUNT_STEPS:
            done = True
        return text, done, reward

    def train_samples(self,sample_num):
        qi, qi_i, qi_lst, D_gt_id, D_gt_title = self.get_samples(sample_num, max_words_input=self.search.max_words_input)

        print 'qi = ', qi
        print 'qi_i = ', qi_i
        print 'qi_lst = ', qi_lst
        print 'D_gt_id = ', D_gt_id
        #print 'D_gt_title = ', D_gt_title
                 
        current_queries = qi_lst
        actions = qi_i, current_queries, D_gt_id
        [text, expanded_queries], reward, done, found = self.execute(actions)
        #print "text = ", text
        print "expanded_queries = ", expanded_queries
        print "reward = ", reward
        return [expanded_queries, reward]        
        
    def reset(self):
        """
        Resets the state of the environment, returning an initial observation.
        Outputs
        -------
        observation : the initial observation of the space. (Initial reward is assumed to be 0.)
        """
        #t0 = time()
        #for now lets get one sample with all.
        #kf = utils.get_minibatches_idx(len(self.qi), len(self.qi), shuffle=True)
        #kf = utils.get_minibatches_idx(len(self.qi),prm.batch_size_train, shuffle=True)        
        #_, train_index = kf[0] #iterate if len(kf)>1 --> for _, train_index in kf:
        #train_index = self.get_samples(sample_num = 1)
        #print "train_index = ", train_index 
        #print "len(self.qi) =", len(self.qi)
        #print("Got minibatch index {}, cost:".format(time() - t0))

        #qi, qi_i, qi_lst, D_gt_id, D_gt_title = self.get_samples(sample_num = 2, max_words_input=self.search.max_words_input)

        #print 'qi = ', qi
        #print 'qi_i = ', qi_i
        #print 'qi_lst = ', qi_lst
        #print 'D_gt_id = ', D_gt_id
        #print 'D_gt_title = ', D_gt_title
                 
        #current_queries = qi_lst
        #n_iterations = 1  # number of query reformulation iterations.
        #if n_iterations < self.search.q_0_fixed_until:
        #    ones = np.ones((len(current_queries), self.search.max_words_input))
        #    reformulated_query = ones
        #    if n_iterations > 0:
                # select everything from the original query in the first iteration.
        #        reformulated_query = np.concatenate([ones, ones], axis=1)

        #print 'reformulated_query', reformulated_query.shape
        # reformulated_query is our action!!!

        #actions = qi_i, current_queries, D_gt_id
        #[text, actions], reward, done, found =  self.step(actions)
        #print "text = ", text
        #print "actions = ", actions
        #print "reward = ", reward
        #return [text, actions]
        self.counsteps = 0
        return []
        

    def __del__(self):
        pass

    def get_tokenizers(self):
        state_tokenizer = nltk.word_tokenize
        action_tokenizer = nltk.word_tokenize
        return state_tokenizer, action_tokenizer

    @property
    def states(self):
        """
        Return the state space. Might include subdicts if multiple states are available simultaneously.

        Returns: dict of state properties (shape and type).

        """
        return dict(shape=1, type='float')
        
    @property
    def actions(self):
        """
        Return the action space. Might include subdicts if multiple actions are available simultaneously.

        Returns: dict of action properties (continuous, number of actions)

        """
        return dict(shape=1, type='float')
        #raise NotImplementedError    

    
#if __name__ == '__main__':
#  DATA_DIR = '/srv/local/work/sixilu2/sixilu2/github/queryreformulator/QueryReformulator'
#  env = QueryReformulatorEnv(DATA_DIR,dset='train',is_train=True,verbose=True)
#  env.train_samples(2)
