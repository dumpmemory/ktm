"""
Factorization machines on sparse features
"""
import argparse
from datetime import datetime
from pathlib import Path
import json
import os
from sklearn.metrics import roc_auc_score, log_loss
from sklearn.utils.validation import check_is_fitted, validate_data
from sklearn.utils.multiclass import type_of_target
from scipy.sparse import load_npz, issparse
import pywFM
import numpy as np
from dataio import get_paths, load_folds
import sklearn
import scipy
import logging


# Location of libFM's compiled binary file
os.environ['LIBFM_PATH'] = str(Path('libfm/bin').absolute()) + '/'


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


class FMClassifier(sklearn.base.ClassifierMixin, sklearn.base.BaseEstimator):
    def __init__(self, embedding_size=20, nb_iterations=40, X_test=None, y_test=None):
        super().__init__()
        self.embedding_size = embedding_size
        self.nb_iterations = nb_iterations
        self.X_test = X_test
        self.y_test = y_test

    def fit(self, X, y):
        """
        X is usually sparse, nb_samples x nb_features
        y is binary
        """
        X, y = validate_data(self, X, y, accept_sparse=True)
        logging.warning('checksum X train %d y train %d', X.sum(), y.sum())

        self.classes_ = sorted(set(y))

        y_type = type_of_target(y, input_name='y', raise_unknown=True)
        if y_type != 'binary':
            raise ValueError(
                'Only binary classification is supported. The type of target '
                f'is {y_type}.'
            )

        if len(self.classes_) > 2:
            raise ValueError("wtf continuous")
        y = np.unique(y, return_inverse=True)[1]
        fm = pywFM.FM(
            task='classification',
            num_iter=self.nb_iterations,
            k2=self.embedding_size,
            rlog=True,
            seed=42,
            temp_path='tmp'
        )  # MCMC method
        # rlog contains the RMSE at each epoch, we do not need it here
        # TODO: if test is available, then store test predictions in local
        # variable
        if self.X_test is None:
            X_test = X[:1]
            y_test = y[:1]
        else:
            X_test = self.X_test
            y_test = self.y_test
        model = fm.run(X, y, X_test, y_test)

        # Store parameters
        self._mu = model.global_bias or 0
        self._W = np.array(model.weights)
        self._V = model.pairwise_interactions
        self._V2 = np.power(self._V, 2)
        self._rlog = model.rlog
        self._is_fitted = True
        if self.X_test is not None:
            self._test_pred = np.array(model.predictions)
        return self

    def predict_proba(self, X):
        check_is_fitted(self)
        if self.X_test is not None:
            return self._test_pred
        X = validate_data(self, X, reset=False, accept_sparse=True)
        X2 = X.copy()
        if scipy.sparse.issparse(X):
            X2.data **= 2
        else:
            X2 **= 2

        quadratic_term = (np.power(X @ self._V, 2).sum(axis=1)
                          - (X2 @ self._V2).sum(axis=1))

        if issparse(X):
            y_pred = (self._mu + X @ self._W +
                      0.5 * quadratic_term.A1)
        else:
            y_pred = (self._mu + X @ self._W +
                      0.5 * (np.power(X @ self._V, 2).sum(axis=1)
                             - (X2 @ self._V2).sum(axis=1)).A1)
        preds = sigmoid(y_pred)
        return np.column_stack((1 - preds, preds))

    def predict(self, X):
        check_is_fitted(self)
        if self.X_test is not None:
            return self._test_pred.round()
        X = validate_data(self, X, reset=False, accept_sparse=True)
        if len(self.classes_) == 1:
            return np.ones(len(X)) * self.classes_[0]
        return np.round(self.predict_proba(X)[:, 1])

    def __sklearn_is_fitted__(self):
        return hasattr(self, "_is_fitted") and self._is_fitted

    def _get_tags(self):
        return {"binary_only": True}

    def _more_tags(self):
        return {"binary_only": True}

    def __sklearn_tags__(self):
        tags = super().__sklearn_tags__()
        tags.classifier_tags.multi_class = False
        tags.input_tags.sparse = True
        return tags


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run FM')
    parser.add_argument('X_file', type=str, nargs='?')
    parser.add_argument('--iter', type=int, nargs='?', default=20)
    parser.add_argument('--d', type=int, nargs='?', default=20)
    parser.add_argument('--subset', type=int, nargs='?', default=0)
    parser.add_argument('--metrics', type=bool, nargs='?', const=True,
                        default=False)
    parser.add_argument('--folds', type=str, nargs='?', default='strong')
    options = parser.parse_args()

    df, X_file, folder, y_file, y_pred_file = get_paths(options, 'FM')
    X_sp = load_npz(X_file).tocsr()
    nb_samples, _ = X_sp.shape
    y = np.load(y_file).astype(np.int32)

    predictions = []
    params = {
        'task': 'classification',
        'num_iter': options.iter,
        'rlog': True,
        'learning_method': 'mcmc',
        'k2': options.d,
        'seed': 42
    }
    fm = pywFM.FM(**params)
    for i, (i_train, i_test) in enumerate(load_folds(options, df)):

        logging.warning('folds train %s test %s', i_train.shape, i_test.shape)
        logging.warning(
            'checksum indices train %d test %d', i_train.sum(), i_test.sum())

        X_train, X_test, y_train, y_test = (X_sp[i_train], X_sp[i_test],
                                            y[i_train], y[i_test])

        logging.warning(
            'checksum X train %d y train %d', X_train.sum(), y_train.sum())

        model = fm.run(X_train, y_train, X_test, y_test)
        y_pred_test = np.array(model.predictions)
        logging.warning('pred model 1 %s', y_pred_test[:5])

        # TODO double check that predictions are same with the classifier
        # called with test set

        model2 = FMClassifier(
            embedding_size=options.d, nb_iterations=options.iter)
        model2.X_test = X_test
        model2.y_test = y_test
        model2.fit(X_train, y_train)
        test_pred = model2.predict_proba(X_test)
        print('pred', test_pred[:5])
        assert test_pred.shape == y_test.shape
        diff = test_pred - y_pred_test
        logging.warning('diff %s', diff[:5])
        logging.warning('max diff %f', diff.max())

        predictions.append({
            'fold': i,
            'pred': y_pred_test.tolist(),
            'y': y_test.tolist()
        })

        if options.metrics:
            df_test = df.iloc[i_test]
            assert len(df_test) == len(y_pred_test)
            df_test['pred'] = y_pred_test
            df_test.to_csv(y_pred_file, index=False)

        print('Test predict:', y_pred_test)
        print('Test was:', y_test)
        print('Test ACC:', np.mean(y_test == np.round(y_pred_test)))
        try:
            print('Test AUC', roc_auc_score(y_test, y_pred_test))
            print('Test NLL', log_loss(y_test, y_pred_test))
        except ValueError:
            pass

        iso_date = datetime.now().isoformat()
        np.save(folder / 'w.npy', np.array(model.weights))
        np.save(folder / 'V.npy', model.pairwise_interactions)
        saved_results = {
            'predictions': predictions,
            'model': vars(options),
            'mu': model.global_bias,
        }
        with open(folder / f'results-{iso_date}.json', 'w') as f:
            json.dump(saved_results, f)

        break