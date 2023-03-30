# XAI611: mid-term-project
autism spectrum disorder classification with multi site dataset


## 1. Introduction

### project goal

- Developing a deep learning model for Autism Classification with multi-site fMRI dataset  
  ABIDE: large-scale multi-site dataset
- Improving the ASD classification performance of the NYU site by adding data from other multi-site sources to the training dataset  

### data structure

node: region of interest  
edge: functional connectivity  
node features: a row of functional connectivity  

## 2. Code description
```shell
.
├── README.md
├── Data
│   ├── Data_folder
│   └── results
├── data_preprocessing.py
│   └── data_loader()
├── graph_preprocessing.py
│   ├── flatten_fc()
│   ├── flatten2dense()
│   ├── t_test()
│   ├── pValueMasking()
│   └── define_node_edge()
├── main_NYU.py
│   ├── main()
│   ├── trainer()
│   ├── train()
│   └── test()
├── main_NYU_with_multi_site.py
│   ├── main()
│   ├── trainer()
│   ├── train()
│   └── test()
├── metric.py
│   ├── accuracy()
│   ├── sensitivity()
│   ├── specificity()
│   └── get_clf_eval()
├── model.py
│   ├── ChebyNet()
│   ├── GCN()
│   └── GAT()
└── seed.py
    └── set_seed()

```
### hyperparameter
```shell

# hyperparameter
parser.add_argument('--lr', default=1e-5, type=float, help='learning rate')
parser.add_argument('--weight_decay', default=1e-2, type=float, help='weight decay')
parser.add_argument('--batch_size', default=16, type=int, help='batch size')
parser.add_argument('--num_epochs', default=1000, type=int, help='num_epochs')
parser.add_argument('--optim', default="Adam", type=str, help='optimizer')
parser.add_argument('--betas', default=(0.5, 0.9), type=tuple, help='adam betas')
parser.add_argument("--momentum", default=0.9, type=float, help="optimizer momentum - SGD, MADGRAD")
parser.add_argument("--gamma", default=0.995, type=float, help="gamma for lr learning")

# GNN parameter
parser.add_argument("--embCh", default="[200, 128, 64]", type=str, help="")  # gnn channel
parser.add_argument("--numROI", default=200, type=int, help="cc200")  # number of RoI & intial node features dimension
parser.add_argument("--p_value", default=0.1, type=float, help="topology ratio")  # p_value
parser.add_argument("--dropout_ratio", default=0.5, type=float)
# directory path
parser.add_argument("--timestamp", default=timestamp, type=str, help="")
parser.add_argument('--root_dir', default="Data/", type=str, help='Download dir')  # Data, results root directory 
```

### Environment-수정필요

- Python == 3.7.10
- PyTorch == 1.9.0
- torchgeometric == 1.9.0
- CUDA == 


## 3. Dataset
ABIDE benchmark multi-site fMRI dataset

Site | Scanner | ASD/TC 
---- | ---- | ---- |
NYU | SIEMENS Allegra | 75/100
PITT | SIEMENS Allegra | 30/26
UCLA_1 | SIEMENS Trio | 41/31
UM_1 | GE Signa | 53/53
USM | SIEMENS Trio | 46/25
YALE | SIEMENS Trio | 28/28

https://drive.google.com/drive/folders/1fDH3ULunE0tSVefErfpaZr7LX9vkEAjS?usp=share_link

## 4. Experiments

Site | ACC | SENS | SPEC | F1
---- | ---- | ---- | ---- | ---- |
(1) NYU | 55.12 | 56.34 | 54.22 | 51.22
(2) w multi site | 61.52 | 46.60 | 72.70 | 50.16

## 5. Reference
#### Dataset & code
: [TMI 2023]Improving Multi-Site Autism Classification via Site-Dependence Minimization and Second-Order Functional Connectivit[[paper]][1.1]
#### Framework
: https://pytorch-geometric.readthedocs.io/en/latest/  
#### Baseline:  
[“ChebyNet”, NIPS 2016] Convolutional Neural Networks on Graphs with Fast Localized Spectral Filtering[[paper]][1.2]  
[“GCN”, ICLR 2017] Semi-Supervised Classification with Graph Convolutional Networks[[paper]][1.3]  
[“GAT”, ICLR 2018] Graph Attention Networks[[paper]][1.4]  

[1.1]: https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=9874890
[1.2]: https://arxiv.org/pdf/1606.09375.pdf
[1.3]: https://arxiv.org/pdf/1609.02907.pdf
[1.4]: https://arxiv.org/pdf/1710.10903.pdf
