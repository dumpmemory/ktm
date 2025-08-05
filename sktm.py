"""
Efficient implementation of knowledge tracing machines using scikit-learn.

There are various things that may explain why it cannot be directly applied to
tabular data.
Wins and fails are continuous features, not categorical data, so they should
not be one-hot encoded.
Cf. encode.py to see how they are encoded.

Factorization machines in their MCMC version cannot be implemented as sklearn
estimators because predictions are made at each epoch of training then
averaged.
It is some kind of stochastic averaging.

Author: Jill-Jênn Vie, 2025
"""
import argparse
from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_validate, GroupShuffleSplit, KFold
from scipy.sparse import load_npz
import pandas as pd
# from fm import FMClassifier


parser = argparse.ArgumentParser(description='Run simple KTM')
parser.add_argument(
    'csv_file', type=str, nargs='?', default='data/dummy/data.csv')
parser.add_argument('--feat', type=str, nargs='?', default='iswf')
options = parser.parse_args()


folder = Path(options.csv_file).parent
df = pd.read_csv(options.csv_file)
model = LogisticRegression(solver='liblinear')
# C=1e10 would be unregularized, max_iter=300 if slow
# model = FMClassifier(embedding_size=5, nb_iterations=200)
dataset = load_npz(folder / f'X-{options.feat}.npz')


cv = GroupShuffleSplit(n_splits=5, random_state=42)  # Strong gen
# cv = KFold(n_splits=5, shuffle=True, random_state=42)  # If weak gen
METRICS = ['accuracy', 'roc_auc', 'neg_log_loss']

# If one wants to verify that y-{feat}.npy agrees with df['correct']
# truth = np.load(options.target)
# assert all(df['correct'].values == truth)

cv_results = cross_validate(
    model, dataset, df['correct'],
    scoring=METRICS,  # Use all scores
    return_train_score=True, n_jobs=-1,  # Use all cores
    cv=cv, groups=df['user'], verbose=10
)
for metric in METRICS:
    print(metric, cv_results[f"test_{metric}"],
          cv_results[f"test_{metric}"].mean())
