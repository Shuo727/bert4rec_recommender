# BERT4Rec Recommender

![BERT4Rec](https://img.shields.io/badge/model-BERT4Rec-green?logo=pytorch&logoColor=white)
![Sequential Recs](https://img.shields.io/badge/📈-Sequential%20Recs-green)
![Transformer Inside](https://img.shields.io/badge/⚡-Transformer%20Powered-yellow)
![Baseline](https://img.shields.io/badge/baseline-Popularity-grey)
![Dataset](https://img.shields.io/badge/dataset-MovieLens%2032M-orange)
![Notebook](https://img.shields.io/badge/run%20in-Jupyter-blue?logo=jupyter)
![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)
![Bake with Love](https://img.shields.io/badge/🍰-Bake%20with%20Love-pink)
![Code with Joy](https://img.shields.io/badge/💻-Code%20with%20Joy-lightblue)
![Fueled by Coffee](https://img.shields.io/badge/☕-Fueled%20by%20Coffee-brown)
![Built with ML](https://img.shields.io/badge/🤖-Built%20with%20ML-purple)
![Keep Learning](https://img.shields.io/badge/📚-Keep%20Learning-teal)
![Never Give Up](https://img.shields.io/badge/🔥-Never%20Give%20Up-red)

This mini project explored the BERT4Rec model on the MovieLens dataset, and compared it with a simple popularity baseline.
(NOTE that the configurable hyperparameters of the model is not tuned for optimal performance.)

 The high-level structure of this project is as the follows:

```bash
 bert4rec_recommender/
│
├── data/                      # Folder to store raw data and preprocessed data
│   └── raw/                   # Original MovieLens 32m dataset (download from: https://grouplens.org/datasets/movielens/32m/)
│
├── notebooks/                 # Jupyter Notebooks for experiments, analysis, and visualization
│   └── 01_EDA.ipynb           # Notebook for exploratory data analysis
│   └── 02_bert4rec.ipynb      # Notebook for BERT4Rec implementation and training
│   └── 03_evaluation.ipynb    # Notebook for evaluating model performance and metrics
│
├── src/                       # Folder for Python utility scripts and core implementations
│   ├── data_loader.py         # Data loading and preprocessing functions (e.g., creating sequences)
│   ├── model.py               # BERT4Rec model definition and architecture
│   ├── trainer.py             # Training loop for the BERT4Rec model
│   ├── eval.py                # Evaluation logic (e.g., calculating metrics like NDCG, Precision@K)
│   └── datasets.py            # Define useful data classes.
│
├── requirements.txt           # Python dependencies (e.g., torch, pandas, numpy)
└── README.md                  # Project description, key findings, etc.
```

## Results Summary

Evaluation followed the leave-one-out next-item prediction protocol.

### Top-5 Metrics
| Model        | Hit@5 | NDCG@5 | MRR@5 | Precision@5 |
|--------------|------:|-------:|------:|------------:|
| Popularity   | 0.015 | 0.009  | 0.007 | 0.003       |
| BERT4Rec     | 0.046 | 0.028  | 0.023 | 0.009       |

### Top-10 Metrics
| Model        | Hit@10 | NDCG@10 | MRR@10 | Precision@10 |
|--------------|-------:|--------:|-------:|-------------:|
| Popularity   | 0.027  | 0.013   | 0.008  | 0.003        |
| BERT4Rec     | 0.081  | 0.040   | 0.027  | 0.008        |

### Top-20 Metrics
| Model        | Hit@20 | NDCG@20 | MRR@20 | Precision@20 |
|--------------|-------:|--------:|-------:|-------------:|
| Popularity   | 0.047  | 0.018   | 0.010  | 0.002        |
| BERT4Rec     | 0.139  | 0.054   | 0.031  | 0.007        |

**Key Findings**
- BERT4Rec consistently outperforms the popularity baseline across all metrics.
- At Top-10, BERT4Rec improves Hit Ratio by **+0.054** and NDCG by **+0.027**.
- At Top-20, BERT4Rec more than triples Hit Ratio (0.139 vs 0.047).
- Precision is low overall due to the large candidate set, but BERT4Rec still achieves a clear relative gain.

## References
Sun, Fei, et al. BERT4Rec: Sequential Recommendation with Bidirectional Encoder Representations from Transformer. CIKM 2019.

## License
This project is licensed under the [MIT License](LICENSE).
