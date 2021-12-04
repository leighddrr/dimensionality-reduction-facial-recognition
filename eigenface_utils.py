import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.decomposition import PCA


# region load data
import sklearn.datasets

min_faces_per_person=50
resize = 0.5
color = False

# TODO: confirm whether there is a data leakage issue here
print('Downloading training set...')
dataset_train = sklearn.datasets.fetch_lfw_people(color=color, resize=resize)
print('Done.')

print('Downloading testing set...')
dataset_test = sklearn.datasets.fetch_lfw_pairs(subset='test', color=color, resize=resize)
print('Done.\n\n')

n_train_samples, h, w = dataset_train.images.shape
n_features = h*w

X_train = dataset_train.data
y_train = dataset_train.target
y_dict = dataset_train.target_names

test_pairs = dataset_test.pairs
y_test = dataset_test.target
n_test_pairs = len(test_pairs)

print('# of training images: ', n_train_samples)
print(f'image resolution: {w}x{h}')
print(f'# of face identities: ', len(y_dict))
print('# of testing image pairs', n_test_pairs)

# endregion


# region modeling

# Normalize
normalizer = StandardScaler().fit(X_train)
X_train_transformed = normalizer.transform(X_train)

# number of components to keep
n_components = 50

pca = PCA(n_components=n_components, svd_solver="auto", whiten=False).fit(X_train_transformed)

eigenfaces = pca.components_.reshape((n_components, h, w))

print(f'Explained Variance: {sum(pca.explained_variance_ratio_)} (with {n_components} components)')

# plot top eigenfaces
n_plots = 10
fig, axs = plt.subplots(figsize=(14, 6), ncols=n_plots//2, nrows=2)
for ax, eigenface in zip(axs.flat, eigenfaces):
    ax.imshow(eigenface, cmap='gray')

# endregion

eps_default = 60.34
def eigenface_classifier(face1, face2, eps=eps_default, normalized=False, vectorized=False):
    '''simple threshold as proposed in eigenface paper'''

    # # flatten images into R^n vectiors
    if not vectorized:
        face1 = face1.flatten()
        face2 = face2.flatten()

    if not normalized:
        face1, face2 = normalizer.transform([face1, face2])

    # compute the eigenface projections of each face image
    face1_projection, face2_projection = pca.transform([face1, face2])
    # print(face1_projection)

    # compute the distance between the two projections
    dist = np.linalg.norm(face1_projection - face2_projection, ord=2)

    return dist < eps

def compute_acc(clf, params):
    preds = [clf(x1, x2, vectorized=False, **params) for (x1, x2) in test_pairs]
    acc = sklearn.metrics.accuracy_score(preds, y_test)
    return acc

def plot_2_face_vecs(face1, face2, figsize=(8, 10), normalized=False):
    '''plots two face vectors'''

    if normalized:
        face1, face2 = normalizer.inverse_transform([face1, face2])

    face1 = face1.reshape(h,w)
    face2 = face2.reshape(h,w)


    fig, (ax1, ax2) = plt.subplots(figsize=figsize, ncols=2)
    ax1.imshow(face1, cmap='gray')
    ax2.imshow(face2, cmap='gray')
    return fig, (ax1,ax2)