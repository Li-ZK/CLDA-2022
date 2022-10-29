## Confident Learning-Based Domain Adaptation for Hyperspectral Image Classification
January 2022 IEEE Transactions on Geoscience and Remote Sensing 60:1-1 Follow journal DOI: 10.1109/TGRS.2022.3166817
This is a code demo for the paper "Confident Learning-Based Domain Adaptation for Hyperspectral Image Classification"

Some of our code references the projects
* [Learning to Compare: Bi-Classifier Determinacy Maximization for Unsupervised Domain Adaptation](https://github.com/BIT-DA/BCDM)


## Requirements
CUDA = 10.2

Python = 3.7 

Pytorch = 1.5 

sklearn = 0.23.2

numpy = 1.19.2

cleanlab = 1.0

## dataset

You can download the hyperspectral datasets in mat format at: https://pan.baidu.com/s/14pqanFPK3JQhDIxrjzSn3g?pwd=l3wf, and move the files to `./datasets` folder.

An example dataset folder has the following structure:

```
datasets
├──Indiana
│   ├── DataCube.mat
├── Houston
│   ├── Houston13.mat
│   └── Houston13_7gt.mat
│   ├── Houston18.mat
│   └── Houston18_7gt.mat
├── Pavia
│   ├── paviaU.mat
│   └── paviaU_gt_7.mat
│   ├── pavia.mat
│   └── pavia_gt_7.mat
│── Shanghai-Hangzhou
│   └── DataCube.mat
```

## Usage:
Take CLDA method on the UP2PC dataset as an example: 
1. Open a terminal or put it into a pycharm project. 
2. Put the dataset into the correct path. 
3. Run CLDA_UP2PC.py. `

