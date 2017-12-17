# -*- coding: utf-8 -*-
'''
Custom theano class to query the search engine.
'''

import numpy as np
import average_precision
from collections import OrderedDict

class Search():

    def __init__(self, engine):
        self.engine = engine
        self.n_iterations = 2 # number of query reformulation iterations.
        self.q_0_fixed_until = 2
        self.reformulated_queries = {}
        self.max_terms_per_doc = 15 # Maximum number of candidate terms from each feedback doc. Must be always less than max_words_input .
        self.max_candidates = 40 # maximum number of candidate documents that will be returned by the search engine.
        self.max_feedback_docs = 7  # maximum number of feedback documents whose words be used to reformulate the query.
        self.max_feedback_docs_train = 1  # maximum number of feedback documents whose words be used to reformulate the query. Only used during training.
        self.use_cache = False # If True, cache (query-retrieved docs) pairs. Watch for memory usage.
        self.max_words_input = 15 # Maximum number of words from the input text.
        self.reward = 'RECALL'  # metric that will be optimized. Valid values are 'RECALL', 'F1', 'MAP', and 'gMAP'.
        self.metrics_map = OrderedDict([('RECALL', 0), ('PRECISION', 1), ('F1', 2), ('MAP', 3), ('LOG-GMAP', 4)])

    def perform(self, q_i, D_truth, is_train, current_queries):
        print 'q_i = ', q_i        
        q_m = (q_i > -2).astype('float32')
        print 'q_m = ', q_m 
        n_iter = len(self.reformulated_queries)
        print 'n_iter = ', n_iter
        # outputs
        metrics = np.zeros((len(q_m), len(self.metrics_map)), np.float32)

        if is_train:
            max_feedback_docs = self.max_feedback_docs_train
        else:
            max_feedback_docs = self.max_feedback_docs

        #D_i: word indices ids for all selected documents returned for each query
        D_i = -2 * np.ones((len(q_m), max_feedback_docs, self.max_words_input), np.int32)
        D_gt_m = np.zeros((len(q_m), self.max_candidates), np.float32) #a cell (i,j) is 1 if the jth document returned is ground truth for query i
        D_id = np.zeros((len(q_m), self.max_candidates), np.int32) #document ids for all documents returned for each query


        # no need to retrieve extra terms in the last iteration
        if n_iter == self.n_iterations - 1:
            extra_terms = False
        else:
            extra_terms = True

        # allow the search engine to cache queries only in the first iteration.
        if n_iter == 0:
            save_cache = self.use_cache
        else:
            save_cache = False

        max_cand = self.max_candidates

        qs = []
        for i, q_lst in enumerate(current_queries):
            q = []
            #print " i, q_lst", i, q_lst
            for j, word in enumerate(q_lst):
                if q_m[i, j] == 1 and len(word)>0:
                    #print "j, word ", j, word
                    q.append(str(word))
            q = ' '.join(q)

            if len(q) == 0:
                q = 'dummy'
            #print "q", q
            qs.append(q)

        print "qs = ", qs
        # only used to print the reformulated queries.
        self.reformulated_queries[n_iter] = qs
        print "AAA", n_iter, qs, self.reformulated_queries

        # always return one more candidate because one of them might be the input doc.
        candss = self.engine.get_candidates(qs, max_cand, self.max_feedback_docs, save_cache, extra_terms)
        # for every query, returns a list of documents (their words and the indices of each word)

        for i, cands in enumerate(candss):
            D_truth_dic = {}
            for d_truth in D_truth[i]:
                if d_truth > -1:
                    D_truth_dic[d_truth] = 0

            D_id[i, :len(cands.keys())] = cands.keys()

            j = 0
            m = 0
            cand_ids = []

            selected_docs = np.arange(self.max_feedback_docs)

            if is_train:
                selected_docs = np.random.choice(selected_docs, size=self.max_feedback_docs_train, replace=False)
            #SOOOOO... metrics are calculcated for all returned documents, but candidate words for query reformulation are coming
            #only from max_feedback_docs_train number of documents

            for k, (cand_id, (words_idx, words)) in enumerate(cands.items()):
                cand_ids.append(cand_id)
                # no need to add candidate words in the last iteration.
                if n_iter < self.n_iterations - 1:
                    # only add docs selected by sampling (if training).
                    if k in selected_docs:
                        words = words[:self.max_terms_per_doc]
                        words_idx = words_idx[:self.max_terms_per_doc]

                        D_i[i, m, :len(words_idx)] = words_idx

                        # append empty strings, so the list size becomes <dim>.
                        words = words + max(0, self.max_words_input - len(words)) * ['']

                        # append new words to the list of current queries.
                        current_queries[i] += words

                        m += 1

                if cand_id in D_truth_dic:
                    D_gt_m[i, j] = 1.

                j += 1

            cands_set = set(cands.keys())

            if qs[i].lower() in self.engine.title_id_map:
                input_doc_id = self.engine.title_id_map[qs[i].lower()]
                # Remove input doc from returned docs.
                # This operation does not raise an error if the element is not there.
                cands_set.discard(input_doc_id)

            intersec = len(set(D_truth_dic.keys()) & cands_set)
            recall = intersec / max(1., float(len(D_truth_dic)))
            precision = intersec / max(1., float(self.max_candidates))
            metrics[i, self.metrics_map['RECALL']] = recall
            metrics[i, self.metrics_map['PRECISION']] = precision
            metrics[i, self.metrics_map['F1']] = 2 * recall * precision / max(0.01, recall + precision)
            avg_precision = average_precision.compute(D_truth_dic.keys(), cand_ids)
            metrics[i, self.metrics_map['MAP']] = avg_precision
            metrics[i, self.metrics_map['LOG-GMAP']] = np.log(avg_precision + 1e-5)
            #print 'current_queries = ', current_queries
        return metrics, D_i, D_id, D_gt_m

