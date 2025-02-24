# Accurate Link Prediction for Edge-Incomplete Graphs via PU Learning

This repository is the official implementation of Accurate Link Prediction for Edge-Incomplete Graphs via PU Learning (AAAI '25).

## Abstract
Given an edge-incomplete graph, how can we accurately find its missing links?
The problem aims to discover the missing relations between entities when their relationships are represented as a graph.
Edge-incomplete graphs are prevalent in real-world due to practical limitations, such as not checking all users when adding friends in a social network.
Addressing the problem is crucial for various tasks, including recommending friends in social networks and finding references in citation networks.
However, previous approaches rely heavily on the given edge-incomplete (observed) graph, making it challenging to consider the missing (unobserved) links.
In this paper, we propose PULL, an accurate link prediction method based on the positive-unlabeled (PU) learning.
PULL treats the observed edges in the training graph as positive examples, and the unconnected node pairs as unlabeled ones.
PULL prevents the model from blindly trusting the observed graph by proposing latent variables for unconnected node pairs, and leveraging the expected graph structure with respect to these variables.
Extensive experiments on real-world datasets show that PULL consistently outperforms the baselines for predicting links in edge-incomplete graphs.


## Requirements
We recommend using the following versions of packages:
- `python==3.7.13`
- `cuda==11.6`
- `cudnn==8.5.0`
- `pytorch==1.13.1`
- `torch-geometric==2.3.1`

## Code Description
- `src/models.py` implements for the GCN Link Predictor.
- `src/train.py` contains functions for training the link predictor via PU-learning.
- `main.py` is the main script for training our link predictor for a graph dataset.

## Data Overview
| **Dataset**    |           **Path or Package**            | 
|:--------------:|:----------------------------------------:| 
|   **PubMed**     | `torch_geometric.datasets.CitationFull`     | 
| **Cora-full**   | `torch_geometric.datasets.CitationFull`     | 
| **Chameleon**     | `torch_goemetric.datasets.WikipediaNetwork` | 
| **Crocodile**     | `torch_goemetric.datasets.WikipediaNetwork` | 
| **Facebook**     | `torch_goemetric.datasets.FacebookPagePage` | 

We load public datasets from the Torch Geometric package. 

## How to Run
You can run the demo script in the directory by `bash run.sh`.

You can reproduce the experimental results in the paper with the following commands:
```shell
python main.py --data PubMed --epoch 10 --val-ratio 0.1 --test-ratio 0.1
python main.py --data Cora_full --epoch 10 --val-ratio 0.1 --test-ratio 0.1
python main.py --data chameleon --epoch 10 --val-ratio 0.1 --test-ratio 0.1
python main.py --data crocodile --epoch 10 --val-ratio 0.1 --test-ratio 0.1
python main.py --data FacebookPagePage --epoch 10 --val-ratio 0.1 --test-ratio 0.1
```

Hyperparameters for the main script are summarized as follows:
- `gpu`: index of a GPU to use.
- `seed`: a random seed (any integer).
- `data`: name of a dataset.
- `epochs`: number of iterations to train.
- `val-ratio`: ratio of edges to use in validation.
- `test-ratio`: ratio of edges to use in test.
- `verbose`: print details while running the experiment if set to 'y'.
- `layer`: number of layers in GCN link predictor.
- `units`: number of units in GCN link predictor.
