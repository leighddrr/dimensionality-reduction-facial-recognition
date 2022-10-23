import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def test_parameter(model, param_values, param_name, params_dict, X_train, y_train, X_test, y_test):
    """tests model's score for given values of an isolated paramter while keeping other parameters fixed.

    Args:
        model: model constructor.
        param_values (List): List of values of the parameter to try.
        param_name (string): the parameters name.
        params_dict (dict): dictionary of values for the other parameters.
        X_train: training features.
        y_train: training labels.
        X_test: testing features.
        y_test: testing labels.

    Returns:
        (List[float], List[float]): (train_scores, test_scores)
    """

    train_scores = []
    test_scores = []

    for param_value in param_values:
        params = {**params_dict, **{param_name: param_value}}
        clf = model(**params).fit(X_train, y_train)
        train_scores.append(clf.score(X_train, y_train))
        test_scores.append(clf.score(X_test, y_test))

    return train_scores, test_scores


def get_cv_results(cv_results, param1, param2):
    '''creates pandas dataframe comparing scores across two variable parameters'''
    data_param1 = cv_results[f'param_{param1}'].data
    data_param2 = cv_results[f'param_{param2}'].data

    index = np.sort(np.unique(data_param1))
    index = [f'{param1} = {p}' for p in index]

    columns = np.sort(np.unique(data_param2))
    columns = [f'{param2} = {p}' for p in columns]

    results = pd.DataFrame(index=index, columns=columns)

    for i, score in enumerate(cv_results['mean_test_score']):
        p1 = data_param1[i]
        p2 = data_param2[i]

        row = f'{param1} = {p1}'
        col = f'{param2} = {p2}'

        results.loc[row, col] = score

    return results


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
