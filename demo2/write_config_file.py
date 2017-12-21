import yaml
import os

data = dict(
    data = dict(
	    base_path = '/home/demo/RL/QueryReformulator',
	    pretrained_embedding_path = '/home/demo/RL/QueryReformulator/data/D_cbow_pdw_8B.pkl',
	    #query_dataset_path = '/home/demo/RL/QueryReformulator/data/msa_dataset.hdf5',
		#candidate_terms_corpus_path= '/home/demo/RL/QueryReformulator/data/msa_corpus.hdf5',
		max_words_input=15,
		vocab_size=374557,
	    embedding_dim=500,
	    querydb_size=10
    ),
	model=dict(
		cnn=dict(
			num_filters_query=[256,256],
			filter_sizes_query=[3,3],
			num_dense_nodes_query=[256,30],
			num_filters_terms=[256,256],
			filter_sizes_terms=[9,3],
            num_dense_nodes_terms=[256,1]
		),
		loss=dict(
			neg_entropy_lambda=1e-3,
			alpha=0.1
		)
	),
	search=dict(
		max_terms=15+15
	),
	agent=dict(
		batch_size=10
	),
	reformulation=dict(
		max_steps=3
	)

)

with open('config.yml', 'w') as outfile:
    yaml.dump(data, outfile, default_flow_style=False)