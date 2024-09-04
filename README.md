# High-Performance Recommendation System with CUDA-Accelerated Collaborative Filtering

This project implements a recommendation system using collaborative filtering techniques with CUDA acceleration. The model is trained on the MovieLens dataset and uses matrix factorization to predict user ratings for movies.

## Project Structure

```
/recommendation-system/
├── data/                        # Dataset directory (MovieLens dataset is downloaded by the script)
├── models/                      # Model definition
│   └── matrix_factorization.py
├── scripts/                     # Main scripts
│   └── train.py                 # Main training script
├── requirements.txt             # Python dependencies
├── README.md                    # Project documentation
└── .gitignore                   # Files to ignore
```

## Setup

### 1. Clone the repository:

```
git clone https://github.com/riyaz-maker/High-Performance Recommendation System with CUDA-Accelerated Collaborative Filtering.git
cd High-Performance Recommendation System with CUDA-Accelerated Collaborative Filtering
```

### 2. Install dependencies:

```bash
pip install -r requirements.txt
```

### 3. Run the training script:

```bash
python scripts/train.py
```

The model will be trained on the MovieLens dataset, and training progress will be displayed.

## Dependencies

- Python 3.7+
- PyTorch
- pandas
