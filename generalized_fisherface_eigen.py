# import warnings
import numpy as np
from scipy import linalg

from sklearn.base import BaseEstimator, TransformerMixin, ClassifierMixin
from sklearn.covariance import ledoit_wolf, empirical_covariance, shrunk_covariance
from sklearn.utils.multiclass import unique_labels
from sklearn.utils.validation import check_is_fitted
from sklearn.preprocessing import StandardScaler

from sklearn.discriminant_analysis import _cov, _class_cov, _class_means

class FisherLD(BaseEstimator):
    def __init__(self, n_components=None, shrinkage=None, priors=None):
        """Initialize Fisher's Linear Discriminant.

        Args:
            n_components (int, optional): number of components of projection.
                maximum possible used if not provided. Defaults to None.
            shrinkage (str, float, None): shrinkage for the covariance estimator.
                'auto': (automatic shrinkage using Ledoit-Wolf lemma)
                float: fixed shrinkage constant between 0 and 1 and
                None: no shrinkage
            priors (List[float], optional): class priors.
                calculated automatically if not provided. Defaults to None.
        """

        self.shrinkage = shrinkage
        self.priors = priors
        self.n_components = n_components

    def _solve_eigen(self, X, y, shrinkage):
        """
        Solve the generalized eigenvector problem.
        Finds the generalized eigenvectors for the within-class and
        between-class scatter matrix to find directions which maximize
        separation between classes.

        Args:
            X (np.ndarray): training data of shape (n_samples, n_features)
            y (List[int]): training labels of shape (n_samples,)
            shrinkage (str, float, None): shrinkage for the covariance estimator.
                'auto': (automatic shrinkage using Ledoit-Wolf lemma)
                float: fixed shrinkage constant between 0 and 1 and
                None: no shrinkage
        """

        self.class_means_ = _class_means(X, y)
        self.class_cov_ = _class_cov(
            X, y, self.priors_, shrinkage, covariance_estimator=None
        )


        # TODO: can we implement this in such a way that the generalization *interpolates* between eigenface and fisherface!?
        S_w = self.class_cov_  # within-class scatter matrix
        S_t = _cov(X, shrinkage, covariance_estimator=None)  # total scatter matrix
        S_b = S_t - S_w  # between-class scatter matrix

        # solve generalized eigenvector problem
        eigen_vals, eigen_vecs = linalg.eigh(S_b, S_w) # S_b v_i = lambda_i S_w v_i

        # sort eigenvalues and eigenvectors
        eigen_vals = np.sort(eigen_vals)[::-1]
        eigen_vecs = eigen_vecs[:, np.argsort(eigen_vals)[::-1]]

        self.eigen_vals_ = eigen_vals
        self.eigen_vecs_ = eigen_vecs

        # calculate the explained varaince ratio
        self.explained_variance_ratio = eigen_vals[:self._max_components] / np.sum(eigen_vals)

        # define transformation matrix
        self.transformation_matrix_ = eigen_vecs


    # NOTE: SVD solver seems to be much less prone to overfitting and doesn't require pca-first.
    # TODO figure out why. perhaps implement (though it doesn't use within-class between-class scatter.)

    def fit(self, X, y):
        """
        Fit the fisher's linear discriminant model.

        Args:
            X (np.ndarray): training data of shape (n_samples, n_features)
            y (List[int]): training labels of shape (n_samples,)

        Returns:
            self: Fitted estimator
        """

        self.classes_ = unique_labels(y)
        n_samples, dim = X.shape
        n_classes = len(self.classes_)

        if self.priors is None:  # estimate priors from sample
            _, y_t = np.unique(y, return_inverse=True)  # non-negative ints
            self.priors_ = np.bincount(y_t) / float(len(y))
        else:
            self.priors_ = np.asarray(self.priors)


        # the maximum number of components possible
        max_components = min(len(self.classes_) - 1, dim)

        # if `n_components` is not specified, use largest possible
        if self.n_components is None:
            self._max_components = max_components

        # if `n_components` is specified, check that it's valid
        else:
            if self.n_components > max_components:
                raise ValueError(
                    "n_components cannot be larger than min(n_features, n_classes - 1)."
                )
            self._max_components = self.n_components

        # solve generalized eigenvector problem
        self._solve_eigen( X, y, shrinkage=self.shrinkage)

        return self

    def transform(self, X):
        """
        Project data to linear subspace which maximizes classs separation.

        Args:
            X (np.ndarray): input data of shape (n_samples, n_features)

        Returns:
            np.ndarray: transformed data.
        """

        check_is_fitted(self) # only keep going model has already been fit

        X_new = np.dot(X, self.transformation_matrix_)

        return X_new[:, : self._max_components]