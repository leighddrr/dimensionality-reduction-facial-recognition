from sklearn.decomposition import PCA
from sklearn.decomposition import KernelPCA
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score


from sklearn.base import BaseEstimator

class EigenFacekNN(BaseEstimator):
    def __init__(self, n_components=50, k=5):
        """Initializes an implementation of `eigenface' with a kNN classifier.

        Args:
            n_components (int): the number of components to keep in PCA.
            k (int): number of neighbors for kNN classifier.
        """

        self.n_components = n_components
        self.k = k

        self.pca = PCA(n_components=n_components, svd_solver="auto", whiten=True)

        self.kNN_clf = KNeighborsClassifier(n_neighbors=k, weights='uniform', algorithm='auto', metric='minkowski', p=2)

    def get_params(self, deep=True):
        return {'n_components': self.n_components, 'k': self.k}

    def set_params(self, **parameters):
        for parameter, value in parameters.items():
            setattr(self, parameter, value)
        return self

    def fit(self, X, y):
        """fits EigenFacekNN estimator to given data.

        Args:
            X: Face images. Shape: [n_images, n_pixels]
            y (List[int]): Labels of face images. Shape: [n_images, ]
        """

        self.pca = self.pca.fit(X)

        X_pca = self.pca.transform(X)

        self.kNN_clf.fit(X_pca, y)

        return self


    def predict(self, X):
        """Predicts identity of given faces.

        Args:
            X: Face images. Shape: [n_images, n_pixels]

        Returns:
            List[int]: Predicted identities. Shape: [n_images, ]
        """

        X_pca = self.pca.transform(X)

        self.principle_components = self.pca.components_

        predictions = self.kNN_clf.predict(X_pca)

        return predictions

    def score(self, X, y):
        """returns the accuracy of the given data

        Args:
            X: Face images. Shape: [n_images, n_pixels]
            y (List[int]): True labels. Shape: [n_images, ]

        Returns:
            float: Accuracy.
        """

        preds = self.predict(X)
        return accuracy_score(y, preds)

    def get_eigenfaces():
        raise NotImplemented


class EigenFaceSVC(BaseEstimator):
    def __init__(self, n_components=50, C=1.0, kernel='rbf', gamma='auto'):
        """Initializes an implementation of `eigenface' with a kNN classifier.

        Args:
            n_components (int): the number of components to keep in PCA.
            C (float): Regularization parameter for SVC classifier.
            gamma (float): kernel coefficient for SVC classifier.
            kernel (string): Kernel type for SVC classifier.
        """

        self.n_components = n_components
        self.C = C
        self.kernel = kernel
        self.gamma = gamma

        self.pca = PCA(n_components=n_components, svd_solver="auto", whiten=True)

        self.SVC_clf = SVC(C=C, gamma=gamma, kernel=kernel, class_weight="balanced")

    def get_params(self, deep=True):
        return {'n_components': self.n_components, 'C': self.C, 'kernel': self.kernel, 'gamma': self.gamma}

    def set_params(self, **parameters):
        for parameter, value in parameters.items():
            setattr(self, parameter, value)
        return self

    def fit(self, X, y):
        """fits EigenFacekNN estimator to given data.

        Args:
            X: Face images. Shape: [n_images, n_pixels]
            y (List[int]): Labels of face images. Shape: [n_images, ]
        """

        self.pca = self.pca.fit(X)

        X_pca = self.pca.transform(X)

        self.SVC_clf.fit(X_pca, y)

        return self


    def predict(self, X):
        """Predicts identity of given faces.

        Args:
            X: Face images. Shape: [n_images, n_pixels]

        Returns:
            List[int]: Predicted identities. Shape: [n_images, ]
        """

        X_pca = self.pca.transform(X)

        self.principle_components = self.pca.components_

        predictions = self.SVC_clf.predict(X_pca)

        return predictions

    def score(self, X, y):
        """returns the accuracy of the given data

        Args:
            X: Face images. Shape: [n_images, n_pixels]
            y (List[int]): True labels. Shape: [n_images, ]

        Returns:
            float: Accuracy.
        """

        preds = self.predict(X)
        return accuracy_score(y, preds)

    def get_eigenfaces():
        raise NotImplemented


class EigenFacekPCAkNN(BaseEstimator):
    def __init__(self, n_components=50, kernel='rbf', gamma=None, k=5):
        """Initializes a kernelPCA implementation of `eigenface' with a kNN classifier.

        Args:
            n_components (int): the number of components to keep in kernel PCA.
            k (int): number of neighbors for kNN classifier.
        """

        self.n_components = n_components
        self.kernel = kernel
        self.gamma = gamma
        self.k = k

        self.kpca = KernelPCA(n_components=n_components, kernel=kernel, gamma=gamma, eigen_solver='auto')

        self.kNN_clf = KNeighborsClassifier(n_neighbors=k, weights='uniform', algorithm='auto', metric='minkowski', p=2)

    def get_params(self, deep=True):
        return {'n_components': self.n_components, 'kernel': self.kernel, 'gamma': self.gamma, 'k': self.k}

    def set_params(self, **parameters):
        for parameter, value in parameters.items():
            setattr(self, parameter, value)
        return self

    def fit(self, X, y):
        """fits EigenFacekPCAkNN estimator to given data.

        Args:
            X: Face images. Shape: [n_images, n_pixels]
            y (List[int]): Labels of face images. Shape: [n_images, ]
        """

        self.kpca = self.kpca.fit(X)

        X_kpca = self.kpca.transform(X)

        self.kNN_clf.fit(X_kpca, y)

        return self


    def predict(self, X):
        """Predicts identity of given faces.

        Args:
            X: Face images. Shape: [n_images, n_pixels]

        Returns:
            List[int]: Predicted identities. Shape: [n_images, ]
        """

        X_kpca = self.kpca.transform(X)

        predictions = self.kNN_clf.predict(X_kpca)

        return predictions

    def score(self, X, y):
        """returns the accuracy of the given data

        Args:
            X: Face images. Shape: [n_images, n_pixels]
            y (List[int]): True labels. Shape: [n_images, ]

        Returns:
            float: Accuracy.
        """

        preds = self.predict(X)
        return accuracy_score(y, preds)

    def get_eigenfaces():
        raise NotImplemented
