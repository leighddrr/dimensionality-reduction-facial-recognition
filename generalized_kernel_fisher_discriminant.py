"""Implements Generalization of Kernel Fisher's Discriminant."""

# NOTE: This implementation is based on the implementation of
# standard kernel fisher discriminant analysis in https://arxiv.org/abs/1906.09436

from scipy.sparse.linalg import eigsh
from sklearn.base import BaseEstimator, ClassifierMixin, TransformerMixin
from sklearn.metrics import pairwise_kernels
from sklearn.utils.multiclass import unique_labels
from sklearn.utils.validation import check_is_fitted
from sklearn.preprocessing import OneHotEncoder
import numpy as np


class GeneralizedKernelFisherLD(BaseEstimator, TransformerMixin):
    """
    Generalized Kernel Fisher's Discriminant.

    Adds parameters alpha and beta which take values in [0,1].

    At alpha=beta=0, this reduces to standard kernel fisher's discriminant.

    At alpha=beta=1, this reduces to standard kernel PCA.
    """


    def __init__(self, n_components=2, kernel='linear', alpha=0, beta=0,
                 **kwds):
        """
        Initializes the Generalized Kernel Fisher's Discriminant Transformer.

        Args:
            n_components (int, optional): number of components of projection.
                Defaults to 2.
            kernel (str, optional): the kernel to use.
                one of ['linear', 'poly', 'sigmoid', 'rbf','laplacian', 'chi2'].
                Defaults to 'linear'.
            alpha (float, optional): alpha parameter in range [0,1]. Defaults to 0.
            beta (float, optional): beta parameter in range [0,1]. Defaults to 0.
            **kwds: parameters to pass to the kernel function
        """

        self.kernel = kernel
        self.n_components = n_components

        self.alpha = alpha
        self.beta = beta

        self.kwds = kwds

        if kernel is None:
            self.kernel = 'linear'


    def fit(self, X, y):
        """
        Finds the projections onto new feature space given by kernel.

        Args:
            X (np.ndarray): training data of shape (n_samples, n_features)
            y (List[int]): training labels of shape (n_samples,)

        Returns:
            self: fitted transformer.
        """

        self.classes_ = unique_labels(y)

        self.X_ = X
        self.y_ = y

        y_onehot = OneHotEncoder().fit_transform(
            self.y_[:, np.newaxis])

        K = pairwise_kernels(
            X, X, metric=self.kernel, **self.kwds)

        m_classes = y_onehot.T @ K / y_onehot.T.sum(1)
        indices = (y_onehot @ np.arange(self.classes_.size)).astype('i')
        N = K @ (K - m_classes[indices])

        m_classes_centered = m_classes - K.mean(1)
        M = m_classes_centered.T @ m_classes_centered

        # NOTE: (Awni) I added this to interpolate between kPCA and kFLD
        A_matrix = (1-self.beta) * M + self.beta * K
        B_matrix = (1-self.alpha) * N + self.alpha * np.identity(np.shape(N)[0])

        # Find weights
        self.eigenvals_, self.weights_ = eigsh(A_matrix, self.n_components, B_matrix, which='LM')

        return self

    def transform(self, X):
        """
        Transform X via the projection onto the feature space given by the kernel.

        Args:
            X (np.ndarray): training data of shape (n_samples, n_features)

        Returns:
            np.ndarray: transformed data.
        """

        check_is_fitted(self)

        X_trfm = pairwise_kernels(X, self.X_, metric=self.kernel, **self.kwds) @ self.weights_

        return X_trfm