# XAI611_mid-term-project
autism spectrum disorder classification with multi site dataset


## 1. Introduction

project goal

- Developing a deep learning model for Autism Classification with multi-site fMRI dataset  
 => ABIDE: large-scale multi-site dataset*
- Improving the ASD classification performance of the NYU site by adding data from other multi-site sources to the training dataset  

data structure

node: region of interest
edge: functional connectivity
node features: a row of functional connectivity

## 2. Code description

- **data_preprocessing.py** 
  - data_loader: define DataLoader
- **graph_preprocessing.py**
  - t_test: calculate p_value for global topology masking
  - pValueMasking, define_node_edge: determine graph edge sparsity
- **main_NYU.py**
  - train, test, trainer, main: only NYU site data
- **main_NYU_with_multi_site.py**
  - with multi site data
- **metric.py**
  - ACC, SENS, SPEC, F1
- **model.py**
  - ChebyNet, GCN, GAT
- **seed.py**

## 3. Dataset
ABIDE benchmark multi-site fMRI dataset

## 4. Experiments

## 5. Reference
#### Dataset & code
: Improving Multi-Site Autism Classification via Site-Dependence Minimization and Second-Order Functional Connectivit, TMI 2023  
#### Framework
: https://pytorch-geometric.readthedocs.io/en/latest/  
#### Baseline:  
[“ChebyNet”] Convolutional Neural Networks on Graphs with Fast Localized Spectral Filtering, NIPS 2016  
[“GCN”] Semi-Supervised Classification with Graph Convolutional Networks, ICLR 2017  
[“GAT”] Graph Attention Networks, ICLR 2018  

