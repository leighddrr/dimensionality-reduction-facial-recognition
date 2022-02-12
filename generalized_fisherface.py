import numpy as np
import torch
from torch import nn
from sklearn.decomposition import PCA

class Fisherface():
    '''An implementation of fisherface'''
    def __init__(self, X, y, n_components=None, pca_first=True, verbose=True):

        self.n_samples = np.shape(X)[0]
        self.dim = np.shape(X)[1]

        self.n_classes = len(np.unique(y))


        # set default number components to be number of classes - 1 (if no other number given)
        if n_components is None:
            # this is the maximum number of components s.t. S_W is non-singular
            self.n_components = self.n_classes - 1
        else:
            self.n_components = n_components

        if pca_first:
            # perform PCA first to address S_W, the within-class scatter matrix, being singular

            # the PCA matrix
            if verbose:
                print('Performing PCA for initial dimensionality reduction')
            pca = PCA(n_components=self.n_samples - self.n_classes).fit(X)
            self.W_pca = torch.tensor(pca.components_.T, dtype=torch.float32) # (dim x n_samples - n_classes) matrix
            if verbose:
                print('Done.')

            # the fisher's linear discriminant matrix
            # TODO: consider what the best choice for intialization would be. currently deterministically identity.
            W_fld = torch.eye(self.n_samples - self.n_classes, self.n_components) # (n_samples - n_classes x n_compoenents) matrix
            self.W_fld = nn.Parameter(W_fld)

        else:
            # don't perform PCA, perform standard fisher's linear discriminant
            self.W_pca = torch.eye(self.dim)

            W_fld = torch.Tensor(self.n_samples - self.n_classes, self.n_components) # (n_samples - n_classes x n_compoenents) matrix
            self.W_fld = nn.Parameter(W_fld)


        # NOTE: we assume normalization prior to calling this...
        classes = np.sort(np.unique(y)) # classes in the dataset
        points_by_class = [X[y==class_] for class_ in classes] # the sets X_i
        n_points_by_class = [len(X[y==class_]) for class_ in classes]
        class_means = [np.mean(class_points, axis=0) for class_points in points_by_class] # mu_i's
        global_mean = np.expand_dims(np.mean(X, axis=0),-1)

        # between-class scatter matrix
        if verbose:
            print('Computing the between-class and within-class scatter matrices...')
        self.S_B = np.sum([n_points_by_class[class_] * (class_means[class_] - global_mean).T @ (class_means[class_] - global_mean) for class_ in classes], axis=0)
        self.S_W = np.sum([(points_by_class[class_] - class_means[class_]).T @ (points_by_class[class_] - class_means[class_]) for class_ in classes], axis=0)

        self.S_B = torch.tensor(self.S_B, dtype=torch.float32)
        self.S_W = torch.tensor(self.S_W, dtype=torch.float32)
        if verbose:
            print('Done.')


    def loss(self):
        numerator = torch.det(self.W_fld.t() @ self.W_pca.t() @ self.S_B @ self.W_pca @ self.W_fld)
        denominator = torch.det(self.W_fld.t() @ self.W_pca.t() @ self.S_W @ self.W_pca @ self.W_fld)
        loss = numerator / denominator
        return loss

    @property
    def W_opt(self):
        return self.W_pca @ self.W_fld

    def fit(self, X, y):
        pass

    def transform(self, X):
        pass