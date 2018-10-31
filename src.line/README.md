# LINE implementation of NeRank

Here's the structure of NeRank.

* `data_loader.py`: loading data to memory.
* `embed.py`: contains the R and A embeddings and the LSTM for Q's.
* `generate_walk.py`: the tool to generate random walks.
* `main.py`, `pder.py`: the major logics of the model. _NeRank_ was used to be called "Personal Domain Expert Recommendation" (PDER). Later, we rename it as NeRank. However, the name of "PDER" leaves unchanged in the source files.
* `preprocessing.py`: preprocessing from raw data to a formatted data.
* `recsys.py`: the recommender system part of NeRank.
* `skipgram.py`: a PyTorch implementation of *metapath2vec* and *Deepwalk*.
* `utils.py`: some utility funcs that used during training.