# LogHGAD: A hypergraph-based log anomaly detection method for AIoT systems

The official PyTorch implementation for LogHGAD: A hypergraph-based log anomaly detection method for AIoT systems.

If you have any questions, please feel free to issue or contact me by email.

# Citation

If you use our codes in your research, please cite:

```bibtex
@article{HE2025103630,
    title = {LogHGAD: A hypergraph-based log anomaly detection method for AIoT systems},
    journal = {Journal of Systems Architecture},
    pages = {103630},
    year = {2025},
    doi = {https://doi.org/10.1016/j.sysarc.2025.103630},
    author = {Jiewei He and Chao Song and Ruilin Hu and Zheng Ren}
}
```

# Overview

In AIoT-enabled systems, log-based anomaly detection is critical for ensuring system security and reliability. Existing methods struggle with the concurrent, interleaved nature of logs from modern software systems and often neglect context information in log headers and parameters. To address these limitations, we propose LogHGAD, a novel log anomaly detection method based on hypergraph modeling. LogHGAD partitions logs into session windows using identifiers and constructs a hypergraph where nodes represent log events (with semantic embeddings as features) and hyperedges capture both temporal sequences and complex multivariate interactions (e.g., between processes, components, and entities). It uniquely integrates log header information (e.g., PID, component) and log parameters into the model. An end-to-end framework using a hypergraph convolutional network learns structural embeddings, fuses them with semantic features, and employs downstream classifiers for detection. Extensive experiments on five real-world log datasets demonstrate that LogHGAD outperforms seven state-of-the-art baselines in accuracy, precision, recall, specificity, and F1-Score, while maintaining practical efficiency.

<img width="5376" height="1891" alt="overview" src="https://github.com/user-attachments/assets/ab7828c8-b4f2-4576-b6bb-3e5f6b9cf99d" />



# Running

1. Download log datasets from [Loghub](https://github.com/logpai/loghub).
2. Run the drain.py to parse logs.
3. Run the preprocess.py to structure logs.
4. Run the graph_construction.py to construct hypergraphs.
5. Run the main_hgcn.py or main_svdd.py to train and evaluate the model.
