# Benchmarks

This directory contains a number of benchmarks, that we used to evaluate HO-GDB.


## Datasets

We used three different types of datasets in our evaluation of HO-GDB.

We used parts of the [ZINC20](https://zinc20.docking.org/) dataset, specifically the first 400 molecules (downloaded with the torch dataset loader).
The resulting HO graph has 9,137 nodes, 19,622 edges, 5,692 subgraphs, 10,584 subgraph-edges; the lowered graph contains 45,035 nodes and 159,936 edges.

We also used the [MAG-10](https://www.cs.cornell.edu/~arb/data/cat-edge-MAG-10/) dataset.
The HO graph contains 80,198 nodes and 51,889 hyperedges of 10 different types, and the lowered representation of this graph contains 132,087 nodes and 180,725 edges.

Finally, we use random graphs for some of the strong scaling experiments.
The hypergraph consists of 60017 nodes and 57000 hyperedges.
The size of the hyperedges varies between 2 and 200 with an average of 18.7.
The lowered graph consists of 117017 nodes and 2.14 million edges.
The HO graph with node-tuples has 20000 nodes, 997500 edges, and 4000 tuples.
The size of the node-tuples varies between 2 and 984, with an average of 69.
The tuple sizes are generated with a lognormal distribution and higher degree nodes are more likely to occur in more tuples.
The heterogeneous graph consists of 1.02 million nodes and 2.27 million edges.

The source code to generate the input CSV files for the benchmarks can be found in the file [generate_csvs.py](generate_csvs.py).
Its command line interface is as follows:
```
usage: generate_csvs.py [-h] [--mode MODE] [--root ROOT] [--dir DIR]
                        [--data DATA] [--weak]

options:
  -h, --help   show this help message and exit
  --mode MODE  Mode of operation: 'debug' or 'full'.
  --root ROOT  Root directory containing raw datasets.
  --dir DIR    Output directory for generated CSV files.
  --data DATA  Dataset to process: 'zinc', 'mag10', 'random_t', or 'random_h'.
  --weak       Generate datasets for weak scaling (exclusive to ZINC dataset).
```

## Read OLTP Workloads

### Strong Scaling

<p align="center">
  <img src="/paper/pics/eval_oltp_read_strong_subgraphs.svg" width="60%">
</p>

We use the ZINC20 dataset for the evaluation illustrated in the Figure above.
The benchmark for our strong-scaling read OLTP workload can be found in the file [read_benchmark.py](read_benchmark.py).
Its command line interface is as follows:
```
usage: read_benchmark.py [-h] [--mode MODE] [--subgraph] [--hypergraph]
                         [--import_dir IMPORT_DIR] [--online]
                         [--csv_url CSV_URL]

options:
  -h, --help            show this help message and exit
  --mode MODE           Mode of operation: 'debug' or 'full'.
  --subgraph            Run benchmark for graphs with subgraph collections.
  --hypergraph          Run benchmark for hypergraphs.
  --import_dir IMPORT_DIR
                        Directory containing CSV files for import.
  --online              Use online CSV URLs for data import.
  --csv_url CSV_URL     Base URL for online CSV directory.
```

Additional strong scaling results for other types of higher-order graphs can be found in the paper as well as plots for the latency distribution of specific operations.


### Weak Scaling

<p align="center">
  <img src="/paper/pics/eval_oltp_read_weak_subgraphs.svg" width="60%">
</p>

We use the ZINC20 dataset for the evaluation illustrated in the Figure above.
The benchmark for our weak-scaling read OLTP workload for subgraphs can be found in the file [read_weak_benchmark.py](read_weak_benchmark.py).
Its command line interface is as follows:
```
usage: read_weak_benchmark.py [-h] [--mode MODE] [--results_to_csv]
                              [--import_dir IMPORT_DIR] [--online]
                              [--csv_url CSV_URL]

options:
  -h, --help            show this help message and exit
  --mode MODE           Mode of operation: 'debug' or 'full'.
  --results_to_csv      Save results to CSV.
  --import_dir IMPORT_DIR
                        Directory containing CSV files for import.
  --online              Use online CSV URLs for data import.
  --csv_url CSV_URL     Base URL for online CSV directory.
```


## Mixed OLTP Workloads

<p align="center">
  <img src="/paper/pics/eval_oltp_mixed_strong_node-tuples.svg" width="60%">
</p>

Breakdown of OLTP query workloads used in the node-tuple strong scaling experiment:
| Query Type          | mostly-reads  | mixed      | write-heavy |
| :------------------ | :-----------: | :--------: | :---------: |
| <b>Reads:</b>       | <b>95%</b>    | <b>50%</b> | <b>25%</b>  |
| Retrieve node       | 31.67%        | 16.67%     | 8.33%       |
| Retrieve edge       | 31.67%        | 16.67%     | 8.33%       |
| Retrieve node-tuple | 31.67%        | 16.66%     | 8.33%       |
| <b>Writes:</b>      | <b>5%</b>     | <b>50%</b> | <b>75%</b>  |
| Add edge            | 4.16%         | 41.67%     | 62.5%       |
| Add node-tuple      | 0.84%         | 8.33%      | 12.5%       |

<p align="center">
  <img src="/paper/pics/eval_oltp_mixed_strong_hypergraphs.svg" width="60%">
</p>

Breakdown of OLTP query workloads used in the hypergraph strong scaling experiment:
| Query Type         | mostly-reads  | mixed      | write-heavy |
| :----------------- | :-----------: | :--------: | :---------: |
| <b>Reads:</b>      | <b>93.75%</b> | <b>50%</b> | <b>25%</b>  |
| Retrieve node      | 46.87%        | 25%        | 12.5%       |
| Retrieve hyperedge | 46.88%        | 25%        | 12.5%       |
| <b>Writes:</b>     | <b>6.25%</b>  | <b>50%</b> | <b>75%</b>  |
| Update node        | 3.13%         | 42.19%     | 57.81%      |
| Add hyperedge      | 3.12%         | 7.81%      | 17.19%      |

We use the random graphs (as described above) as datasets for the evaluation illustrated in the Figures above.
The benchmark implementation of the mixed OLTP workloads can be found in the file [scale_benchmark.py](scale_benchmark.py).
Its command line interface is as follows:
```
usage: scale_benchmark.py [-h] [--mode MODE] [--tuples] [--hypergraph]
                          [--split SPLIT] [--no_import]
                          [--import_dir IMPORT_DIR] [--online]
                          [--csv_url CSV_URL]

options:
  -h, --help            show this help message and exit
  --mode MODE           Mode of operation: 'debug' or 'full'.
  --tuples              Run benchmark for graphs with node-tuple collections.
  --hypergraph          Run benchmark for hypergraphs.
  --split SPLIT         Query split type: 'mostly-reads', 'mixed', or 'write-
                        heavy'.
  --no_import           Skip data import step.
  --import_dir IMPORT_DIR
                        Directory containing CSV files for import.
  --online              Use online CSV URLs for data import.
  --csv_url CSV_URL     Base URL for online CSV files.
```


## OLAP Workload: HO GNN

<p align="center">
  <img src="/paper/pics/eval_ho_gnn.svg" width="100%">
</p>

We evaluated HO-GDB with a Higher-Order Graph Neural Network (HO GNN), implemented as a Inter-Message Passing-enabled GNN architecture described by [Fey et al](https://grlplus.github.io/papers/45.pdf).
It is a dual-message-passing architecture with two GNNs (molecular graph and its junction tree). These networks exchange features between layers, allowing cross-level interaction.
The base GNN uses standard Graph Isomorphism Operator layer (GINEConv).
The model is implemented in the file [model.py](model.py).

Our pipeline begins by populating the graph database with 400 molecules from the ZINC20 dataset's test suite; these graphs are then queried and used in training the model.
We compare the performance of the resulting HO GNN against a traditional non-HO GINEConv architecture with three layers.
The benchmark is implemented in the file [gnn_benchmark.py](gnn_benchmark.py).
Its command line interface is as follows:
```
usage: gnn_benchmark.py [-h] [--no_train] [--device DEVICE] [--mode MODE]
                        [--root ROOT] [--hidden_channels HIDDEN_CHANNELS]
                        [--num_layers NUM_LAYERS] [--dropout DROPOUT]
                        [--epochs EPOCHS] [--no_inter_message_passing]
                        [--online] [--csv_url CSV_URL]

options:
  -h, --help            show this help message and exit
  --no_train            Skip training and exit after data import.
  --device DEVICE       CUDA device ID to use for training.
  --mode MODE           Mode of operation: 'debug' or 'full'.
  --root ROOT           Root directory containing raw ZINC dataset.
  --hidden_channels HIDDEN_CHANNELS
                        Number of hidden channels in the GNN.
  --num_layers NUM_LAYERS
                        Number of layers in the GNN.
  --dropout DROPOUT     Dropout rate for the GNN.
  --epochs EPOCHS       Number of training epochs.
  --no_inter_message_passing
                        Disable inter-message passing in the GNN.
  --online              Use online CSV URLs for data import.
  --csv_url CSV_URL     Base URL for online CSV directory.
```
