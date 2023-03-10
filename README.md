# Pruning Quantum Neural Network 

* This repository includes all the scripts for reproducing qnn pruning project ([link](https://j0807s.github.io/projects/3_project/)).

## Installation

### Prerequisites
* python == 3.8
* pennylane == 0.15.1
* tensorflow == 2.5.0
* scikit-learn==1.2.0
* scikit-image=0.19.3

### Setup
```bash
conda env create -f pennylane.yaml
```

## Reproduce
```bash
sh run_org_all.sh
sh run_pruning_all.sh
```

## Results
* The above shell scripts save the best performance and the training time of the model

* output_org.txt 
* output_pruning.txt