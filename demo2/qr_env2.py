import random
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
import cPickle as pkl
from tensorflow.contrib.learn.python.learn.preprocessing import categorical_vocabulary
from tensorflow.contrib import learn
import unicodedata
from tensorforce.environments import Environment
import sys
reload(sys)
sys.setdefaultencoding('utf8')

import parameters as prm
# Set the random number generators' seeds for consistency
#SEED = 123
SEED=245
np.random.seed(SEED)

class QR_ENV(Environment):
    def __init__(self, cfg, dset, is_train, verbose, rewardtype = 'RECALL'):
        # this method returns simulator, state/action vocabularies, and the maximum number of actions
        n_words = 374000 # 100 words for the vocabulary
        DATA_DIR=cfg['data']['base_path']
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
        self.cfg=cfg
        t0=time()
        

        self.vocab = utils.load_vocab(vocab_path, n_words)
        vocabinv = {}
        for k, v in self.vocab.items():
            vocabinv[v] = k
        self.vocabinv=vocabinv

        self.rewardtype = rewardtype
        self.is_train = is_train
        self.search = Search(engine=lucene_search.LuceneSearch(DATA_DIR, self.vocab, n_threads, max_terms_per_doc, index_name, index_name_term, docs_path, docs_path_term, use_cache))

        self.batch_size=cfg['agent']['batch_size']
        t0 = time()
        dh5 = dataset_hdf5.DatasetHDF5(dataset_path)
        self.qi = dh5.get_queries(dset)
        cfg['data']['querydb_size']=len(self.qi)
        self.dt = dh5.get_doc_ids(dset)
        print("Loading queries and docs {}".format(time() - t0))
        self.counsteps = 0
        #self.reset()

    

    def process_expanded_words(self,expanded_queries):
        expand_text=[]
        for i in range(len(expanded_queries)):
            expand_text_i=[]
            for j, word in enumerate(expanded_queries[i]):
                if word!='':
                    if isinstance(word,str):
                        #expand_text_i.append(word.encode('ascii', 'ignore'))
                        expand_text_i.append(word)
                    else:
                        expandedcode = unicodedata.normalize('NFKD', word).encode('ascii', 'ignore')
                        expand_text_i.append(expandedcode)
            expand_text.append(' '.join(expand_text_i))
        return expand_text
    
    


    def execute(self, actions):
        """
        Executes action, observes next state(s) and reward.

        Args:
            actions: Actions to execute.

        Returns:
            (Dict of) next state(s), boolean indicating terminal, and reward signal.
        """
        done = False
        #reformulated_query, current_queries, D_gt_id = actions
        

        self.counsteps += 1
        print "************execute query reformulation iteration:", self.counsteps
        #print 'current states........', self.state
        #query_text = utils.idx2text2(self.state, self.vocabinv)
        
        n,m=np.shape(actions)
        query_index=-2*np.ones((n,m),dtype='int')
        for i in range(len(actions)):
            actionids=actions[i]
            for j in range(len(actionids)):
                query_index[i,j] =self.state[i,actionids[j]]
            
        print 'current_actions = ', np.shape(query_index),query_index

        '''action_status=utils.is_emptyaction(query_index)
        if sum(action_status)<self.batch_size:
            self.state=self.reset()
            query_index=self.action
            
        print 'current_actions = ', np.shape(query_index), query_index'''
        query_text=utils.idx2text2(query_index, self.vocabinv)
        print 'current actions = ', query_text
        D_gt_id=self.D_gt_id
        metrics, D_i_, D_id_, D_gt_m_ = self.search.perform(query_index, D_gt_id, self.is_train, query_text)
        
        #self.D_gt_id=D_id_
        
        #print 'current_queries (after calling search.perform) = ',query_text
        i = 0
        expanded_query_text = query_text
        metric_idx = self.search.metrics_map[self.rewardtype.upper()]
        reward = metrics[:,metric_idx]
       
        if  self.counsteps > self.cfg['reformulation']['max_steps']:
            done = True
        print 'expanded query text,:', expanded_query_text
        expanded_query_text = self.process_expanded_words(expanded_query_text)
        #print ' process_expanded_words:', expanded_query_text
        expanded_i, expandes_lst_ = utils.text2idx2(expanded_query_text, self.vocab, self.cfg['search']['max_terms'])
        #self.state= expanded_i
        #print 'execution__new state id after expansion',expanded_i
        terminal=done
        return self.state,terminal, sum(reward)
    
    

    def __str__(self):
        return "Query Reformulator Env"

    def get_samples(self, sample_num = 1, max_words_input = 200):
        print 'get_samples.................'
        if sample_num <= 0:
          sample_num = 1
        train_index = utils.get_one_random_batch_idx(len(self.qi),sample_num)
        print 'train_index',train_index

        input_queries = self.qi
        target_docs = self.dt
        vocab = self.vocab
        engine = self.search.engine
        qi = [utils.clean(input_queries[t].lower()) for t in train_index]
        D_gt_title = [target_docs[t] for t in train_index]
        print 'qi',qi
        print 'D_gt_title',D_gt_title

        D_gt_id_lst = []
        for j, t in enumerate(train_index):
            print("j",j)
            D_gt_id_lst.append([])
            for title in D_gt_title[j]:
                #print("title", title)
                if title in engine.title_id_map:
                    D_gt_id_lst[-1].append(engine.title_id_map[title])
                #else:
                #    print 'ground-truth doc not in train_index:', title

        print 'D_gt_id_lst',D_gt_id_lst
        D_gt_id = utils.lst2matrix(D_gt_id_lst)
        print 'D_gt_id',D_gt_id

        qi_i, qi_lst_ = utils.text2idx2(qi, vocab, max_words_input)
        #print("qi_i", qi_i)
        #print("qi_lst_", qi_lst_)

        qi_lst = []
        for qii_lst in qi_lst_:
            # append empty strings, so the list size becomes <dim>.
            qi_lst.append(qii_lst + max(0, max_words_input - len(qii_lst)) * [''])
        return qi, qi_i, qi_lst, D_gt_id, D_gt_title
        
    def reset(self):
        print '********************reset**********'
        """
        Reset environment and setup for new episode.

        Returns:
            initial state of resetted environment.
        """
        t0 = time()
        qi, qi_i, qi_lst, D_gt_id, D_gt_title = self.get_samples(sample_num = self.batch_size, max_words_input=self.search.max_words_input)
        current_query_text = qi_lst
        current_query_code=qi_i
        self.action=current_query_code
        print 'reset current query text:',current_query_text
        print 'reset current query code:',current_query_code
        print 'reset action code:',  self.action
        self.D_gt_id=D_gt_id
        self.D_gt_title=D_gt_title
        metrics, D_i_, D_id_, D_gt_m_ = self.search.perform(current_query_code, D_gt_id, self.is_train, current_query_text)
        expanded_query_text = current_query_text
        metric_idx = self.search.metrics_map[self.rewardtype.upper()]
        reward = metrics[:, metric_idx]
        expanded_query_text = self.process_expanded_words(expanded_query_text)
        print ' reset expanded text:', expanded_query_text
        expanded_query_code, expandes_lst_ = utils.text2idx2(expanded_query_text, self.vocab, self.cfg['search']['max_terms'])
        #expanded_query_code, terminal,reward =  self.execute(self.action)
        print 'reward',reward
        self.state=expanded_query_code
        self.counsteps =0
        return self.state



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

        states_num=self.cfg['search']['max_terms']#30
        return dict(shape=(states_num), type='int')

    @property
    def actions(self):
        """
        Return the action space. Might include subdicts if multiple actions are available simultaneously.

        Returns: dict of action properties (continuous, number of actions)

        """
     
        return dict(type='int',shape=(15),num_actions=self.cfg['search']['max_terms'])
        
        #return dict(shape=1, type='float')
        #raise NotImplementedError



'''if __name__ == '__main__':
  DATA_DIR = '/srv/local/work/sixilu2/sixilu2/github/queryreformulator/QueryReformulator'
  env = QueryReofrmulatorEnv(DATA_DIR,dset='train',is_train=True,verbose=True)
  env.train_samples(2)
'''
