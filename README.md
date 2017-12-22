# A Query Reformulator based on Tensorforce
tensorforce4qr is an implementation of a Query Reformulator(https://arxiv.org/abs/1704.04572) .
It is based on a  Deep Reinforcement Learning framework tensorforce (https://github.com/reinforceio/tensorforce)

Originally it is the project for UIUC(MCS-DS) CS410 course. As a team, this repository is just one implementation of our project, another implementation is located at https://github.com/YifanTian/tensorforce_QR.

The lastest design document can be found at https://docs.google.com/document/d/10HiHWQJoQcBINGntWygAGJAYGHSVVygZz8eyetWnyHE/

Our project video presentaiton can be found at https://mediaspace.illinois.edu/media/t/1_edr4fpp3


## Dependencies
To run the code, you will need:
* Python 2.7
* [NumPy](http://www.numpy.org/)
* [NLTK](http://www.nltk.org/)
* [h5py](http://www.h5py.org/)
* [PyLucene 6.2 or higher](http://lucene.apache.org/pylucene/)
* [tensorforce](https://github.com/reinforceio/tensorforce)
* [tensorflow](https://www.tensorflow.org/)

## Notes on installation(Linux Ubuntu)
* If your account does not have the sudo right, try to use **pip --user** command to install .

* pylucene installtion

I mainly followed the instruction: http://bendemott.blogspot.com/2013/11/installing-pylucene-4-451.html

But did some modifications as below.

**1. pylucene-6.5.0/jcc/setup.py**
```
JDK = {
    'darwin': JAVAHOME or JAVAFRAMEWORKS,
    'ipod': '/usr/include/gcc',
     #'linux': '/usr/lib/jvm/java-8-oracle',  
    'linux': '/usr/lib/jvm/java-8-openjdk-amd64',#!!!!change to your local java openjdk
    'sunos5': '/usr/jdk/instances/jdk1.6.0',
    'win32': JAVAHOME,
    'mingw32': JAVAHOME,
    'freebsd7': '/usr/local/diablo-jdk1.6.0'
}
```
**2.pylucene-6.5.0/Makefile**

Change below sections:

**From:**
```
#Linux     (Ubuntu 6.06, Python 2.4, Java 1.5, no setuptools)
#PREFIX_PYTHON=/usr
#ANT=ant
#PYTHON=$(PREFIX_PYTHON)/bin/python
#JCC=$(PYTHON) $(PREFIX_PYTHON)/lib/python2.4/site-packages/jcc/__init__.py
#NUM_FILES=8
```
**to**:
```
#Linux     (Ubuntu 6.06, Python 2.4, Java 1.5, no setuptools)
PREFIX_PYTHON=/usr
ANT=ant
PYTHON=$(PREFIX_PYTHON)/bin/python
JCC=$(PYTHON) -m jcc --shared
NUM_FILES=8
```

In addtion, due to limited user account access right, install pylucene into current user local lib folder, modify below commands in Makefile:

```
INSTALL_DEST:=/home/sixilu2/.local/lib/python2.7/site-packages/
install: jars
	$(GENERATE) --install $(DEBUG_OPT) $(INSTALL_OPT) --install-dir $(INSTALL_DEST)
```

## How to run?

1.Download datset from [downloaded here](https://drive.google.com/drive/folders/0BwmD_VLjROrfLWk3QmctMXpWRkE?usp=sharing):

Due to large size of dataset files, this project only requires below smaller ones.

 **msa_dataset.hdf5**: MS Academic dataset: a query is the title of a paper and the ground-truth documents are the papers cited within.
 
 **msa_corpus.hdf5**: MS Academic corpus: each document consists of a paper title and abstract.
 
 **D_cbow_pdw_8B.pkl**: A python dictionary pretrained word embeddings from the [Word2Vec tool](https://code.google.com/archive/p/word2vec/).

Save them under /srv/local/work/sixilu2/sixilu2/github/queryreformulator/QueryReformulator/data/ folder.

2. modify below configuration of **qr_runner.py**:

```
DATA_DIR = '/srv/local/work/sixilu2/sixilu2/github/queryreformulator/QueryReformulator'
```

3. run  **python qr_runner.py**

## Design details
For the latest design document,please refer to 
https://docs.google.com/document/d/10HiHWQJoQcBINGntWygAGJAYGHSVVygZz8eyetWnyHE/
