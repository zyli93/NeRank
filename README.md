# NeRank

@Author: Zeyu Li, Jyun-Yu Jiang, Yizhou Sun, Wei Wang


## Introduction
This is the repo of "Personalized Question Routing via Heterogeneous Network Embedding".
Bibtex citation format:
```
@inproceedings{li2019personalized,
  title={Personlaized Question Routing via Heterogeneous Network Embedding},
  author={Li, Zeyu and Jiang, Jyun-Yu and Sun, Yizhou and Wang, Wei},
  year={2019},
  organization={AAAI Conference on Artificial Intelligence (AAAI)}
}
```

## Install Dependencies

NeRank is prototyped in PyTorch 0.4.0a0. Other package dependencies are listed in "requirements.txt". 
You can run
```
$ pip install -r requirements.txt
```
 to install the packages that are needed. 


## Preparing Data
The archive of the dataset is available [here](https://archive.org/details/stackexchange). Download the dataset and unzip the `7z` files into `./raw/`. Then run
```
$ python src/preprocessing.py [name of dataset] [threshold] [prop of test] [test sample size]
```
to preprocess the `XML` to `json`.

_Parameters_:
* `name of dataset`: name of the dataset, such as "Biology", "English", and "3dpringting".
* `threshold`: #. of entities to be selected as for the training.
* `prop of test`: float. proportion of questions for testing.
* `test sample size`: number of candidate questions in each test samples.
The produced files will appear in `./data/`. 


## Generate Metapath and Input Pairs
Generate the input pairs by `generate_walk.py` using the following command.
```
$ python src/generate_walk.py [name of dataset] [length] [num_walk] [window_size]
```

_Parameters_:
* `name of dataset`: Similar to above.
* `length`: the length of the generated walks.
* `num_walk`: the number of times each node is covered by walks.
* `window_size`: the size of Skip-gram sampling window.


## Run
Run NeRank by the following command:
```
$ ./run.sh 
```
But before running `run.sh`, please look into it and tweak the settings.

_Parameter_:
* `ID`: the identifier of a certain training/testing, will be used in output file name.
You would see the performance in `./performance/`.

