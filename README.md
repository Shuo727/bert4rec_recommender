# BERT4Rec Recommender

This project explores a baseline and advanced transformer-based recommendation system.

 The high-level structure of this project is as the follows:
 bert4rec_recommender/
│
├── data/                      # Folder to store raw data, preprocessed data, or output files
│   └── raw/                   # Original MovieLens dataset (e.g., ratings.csv, movies.csv)
│   └── processed/             # Processed data (e.g., sequences, train/test splits)
│
├── notebooks/                 # Jupyter Notebooks for experiments, analysis, and visualization
│   └── 01_EDA_and_baseline.ipynb   # Notebook for EDA and baseline model implementation
│   └── 02_bert4rec.ipynb      # Notebook for BERT4Rec implementation and training
│   └── 03_evaluation.ipynb    # Notebook for evaluating model performance and metrics
│
├── src/                       # Folder for Python utility scripts and core implementations
│   ├── data_loader.py         # Data loading and preprocessing functions (e.g., creating sequences)
│   ├── model.py               # BERT4Rec model definition and architecture
│   ├── trainer.py             # Training loop for the BERT4Rec model
│   ├── evaluator.py           # Evaluation logic (e.g., calculating metrics like NDCG, Precision@K)
│   └── utils.py               # Utility functions like saving models, handling configurations, etc.
│
├── requirements.txt           # Python dependencies (e.g., torch, pandas, numpy)
└── README.md                  # Project description, setup instructions, etc.