import numpy as np
from scipy import linalg

# TODO: remove unnecessary imports
from sklearn.base import BaseEstimator, TransformerMixin, ClassifierMixin
from sklearn.covariance import ledoit_wolf, empirical_covariance, shrunk_covariance
from sklearn.utils.multiclass import unique_labels
from sklearn.utils.validation import check_is_fitted
from sklearn.preprocessing import StandardScaler

from sklearn.discriminant_analysis import _cov, _class_cov, _class_means

class GeneralizedFisherLD(BaseEstimator):
    def __init__(self, n_components=None, alpha=0, beta=0, shrinkage=None, priors=None):
        """Initialize Fisher's Linear Discriminant.

        Args:
            n_components (int, optional): number of components of projection.
                maximum possible used if not provided. Defaults to None.
            alpha (float, optional): alpha parameter in range [0,1]. Defaults to 0.
            beta (float, optional): beta parameter in range [0,1]. Defaults to 0.
            shrinkage (str, float, None): shrinkage for the covariance estimator.
                'auto': (automatic shrinkage using Ledoit-Wolf lemma)
                float: fixed shrinkage constant between 0 and 1 and
                None: no shrinkage
            priors (List[float], optional): class priors.
                calculated automatically if not provided. Defaults to None.
        """

        self.n_components = n_components
        self.alpha = alpha
        self.beta = beta
        self.shrinkage = shrinkage
        self.priors = priors

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
                float: fixed shrinkage constant between 0 and 1
                None: no shrinkage
        """

        self.class_means_ = _class_means(X, y)
        self.class_cov_ = _class_cov(X, y, self.priors_, shrinkage, covariance_estimator=None)

        S_w = self.class_cov_  # within-class scatter matrix
        S_t = _cov(X, shrinkage, covariance_estimator=None)  # total scatter matrix
        S_b = S_t - S_w  # between-class scatter matrix

        # calculate the A_matrix and B_matrix
        A_matrix = S_t - (1 - self.beta)*S_w
        B_matrix = self.alpha * np.identity(np.shape(S_w)[0]) + (1 - self.alpha)*S_w

        # store scatter matrices
        self.S_w = S_w
        self.S_t = S_t
        self.S_b = S_b

        # store A_matrix and B_matrix
        self.A_matrix_ = A_matrix
        self.B_matrix_ = B_matrix


        # solve generalized eigenvector problem
        eigen_vals, eigen_vecs = linalg.eigh(A_matrix, B_matrix) # A v_i = lambda_i B v_i


        sorted_idx = np.argsort(eigen_vals)[::-1]
        eigen_vecs = eigen_vecs[:, sorted_idx]  # sort eigenvectors
        eigen_vals = eigen_vals[sorted_idx] # sort eigenvalues

        self.explained_variance_ratio_ = eigen_vals[:self._max_components] / np.sum(eigen_vals)

        self.transformation_matrix_ = eigen_vecs


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
        max_components = min(n_classes - 1, dim)

        # TODO FIXME: is this only relevant for alpha=0? if so only check in this case
        # NOTE: maybe larger number of components is possible with alpha > 0. if so, this could meaningfully improve performance!
        # if `n_components` is not specified, use largest possible
        if self.n_components is None:
            self._max_components = max_components

        # if `n_components` is specified, check that it's valid
        else:
            # # FIXME: add proper checks for valid number of components
            # if self.alpha == 0:
            #     if self.n_components > max_components:
            #         raise ValueError(
            #             "n_components cannot be larger than min(n_features, n_classes - 1)."
            #         )
            self._max_components = self.n_components

        self._solve_eigen(X, y, shrinkage=self.shrinkage)

        return self

    def transform(self, X):
        """
        Project data to linear subspace which maximizes classs separation.

        Args:
            X (np.ndarray): input data of shape (n_samples, n_features)

        Returns:
            np.ndarray: transformed data.
        """

        check_is_fitted(self)

        X_transformed = np.dot(X, self.transformation_matrix_)

        return X_transformed[:, : self._max_components]

    def within_class_scatter(self):
        """
        compute and return the within-class scatter

        Returns:
            float: within-class scatter
        """

        check_is_fitted(self)

        return np.linalg.det(self.transformation_matrix_.T @ self.S_b @ self.transformation_matrix_)

    def between_class_scatter(self):
        """
        compute and return the between-class scatter.

        Returns:
            float: between-class scatter.
        """

        check_is_fitted(self)

        return np.linalg.det(self.transformation_matrix_.T @ self.S_w @ self.transformation_matrix_)

    def class_scatter_ratio(self):
        """
        compute and return the ration of between-class scatter to within-class scatter.

        Returns:
            float: class scatter ratio
        """

        check_is_fitted(self)

        return self.between_class_scatter() / self.within_class_scatter()
