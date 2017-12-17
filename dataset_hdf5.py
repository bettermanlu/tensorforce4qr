'''
Class to access queries and references stored in the hdf5 file.
'''
import h5py
import ast

class DatasetHDF5():

    def __init__(self, path):
        self.f = h5py.File(path, 'r')


    def get_queries(self, dset='train'):
        '''
        Return the queries.
        'dset':  dataset to be returned ('train', 'valid' and/or 'test').
        '''
        #return list(self.f['queries_'+dset][0:10])
        return list(self.f['queries_'+dset])


    def get_doc_ids(self, dset='train'):
        '''
        Return the <queries, references> pairs.
        'dset': dataset to be returned ('train', 'valid' and/or 'test').
        '''
        #return map(ast.literal_eval, list(self.f['doc_ids_'+dset][0:10]))
        return map(ast.literal_eval, list(self.f['doc_ids_'+dset]))

