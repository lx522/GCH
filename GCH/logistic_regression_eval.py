import numpy as np
from sklearn import metrics
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, ShuffleSplit, train_test_split
from sklearn.multiclass import OneVsRestClassifier
from sklearn.preprocessing import OneHotEncoder, normalize
import  datetime


def fit_logistic_regression(X, y, dataset, data_random_seed=1, repeat=1,):# X=representations（7650，128） y=labels(7650,)
    # transfrom targets to one-hot vector
    one_hot_encoder = OneHotEncoder(categories='auto', sparse=False)
    # print(y)
    y = one_hot_encoder.fit_transform(y.reshape(-1, 1)).astype(np.bool)# [[False False False ... False  True False], [False False False ... False False False], [False False False ... False False False], ..., [False  True False ... False False False], [False False False ... False  True False], [False False False ...  True False

    # normalize x
    X = normalize(X, norm='l2')

    # set random state
    rng = np.random.RandomState(data_random_seed)  # this will ensure the dataset will be split exactly the same
                                                   # throughout training

    accuracies = []
    for _ in range(repeat):# repeat=3 取值从0到2 repeat=1 取值只有0
        # different random split after each repeat
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.8, random_state=rng)
        # X_train(1530,128) X_test(6120,128) y_train(1530,8) y_test(6120,8)
        # grid search with one-vs-rest cssifiers
        logreg = LogisticRegression(solver='liblinear')
        c = 2.0 ** np.arange(-10, 11)
        cv = ShuffleSplit(n_splits=5, test_size=0.5)
        clf = GridSearchCV(estimator=OneVsRestClassifier(logreg), param_grid=dict(estimator__C=c),
                           n_jobs=5, cv=cv, verbose=0)
        clf.fit(X_train, y_train)
        y_pred = clf.predict_proba(X_test)# y_pred=[[1.22353738e-04 9.99118903e-01 2.09661653e-05 ... 7.24145499e-04,  2.30386481e-05 9.36655426e-05], [1.05599843e-03 9.98226329e-01 9.58181195e-04 ... 1.46149688e-04,  2.17813249e-05 2.10386325e-04], [4.60045286e-04 7.40835530e-04 9.99487880e-01 ... 5.18688
        y_pred = np.argmax(y_pred, axis=1)# y_pred=[1 1 2 7 6 5 1 4 2 4 6 6 0 4 6 6 1 1 6 6 3 2 1 3 4 6 6 1 4 4 6 6 1 6 3 5 6, 6 6 6 6 2 6 0 5 1 4 2 4 1 6 2 1 3 6 5 1 5 4 6 6 5 1 6 2 1 5 7 6 3 1 1 2 1, 2 5 6 6 5 3 6 4 3 1 1 4 5 6 4 2 3 6 6 3 3 5 7 1 4 6 0 1 1 5 1 2 1 6 5 5 1, 6 6 1 3 6 3 4 1 6 6 4 3 6 6 3 1 1 6 5 1 3 1 0 6 1 1 3 6 6 3 7 5 6 6 6 4 4, 1 3 1 6 3 3 2 6 6 7 6 1 6 2 6 0 1 0 1 0 3 7 6 1 6 4 1 6 3 6 4 4 6 6 6 4 2, 2 1 5 4 0 2 0 6 3 6 6 4 3 6 4 5 4 5 6 3 1 1 1 5 1 1 5 6 6 5 4 6 6 4 7 1 4, 6 4 6 1 6 6 6 2 5 0 6 3 1 4 1 3 5 1 6 2 1 2 4 2 3 1 3 3 2 5 3 5 1 4]
        y_pred = one_hot_encoder.transform(y_pred.reshape(-1, 1)).astype(np.bool)# y_pred=[[False  True False ... False False False], [False  True False ... False False False], [False False  True ... False False False], ..., [False False False ...  True False False], [False  True False ... False False False], [False False False ... False False False]]

        test_acc = metrics.accuracy_score(y_test, y_pred)# test_acc=0.9281045751633987,0.9326797385620915,0.9349673202614379
        accuracies.append(test_acc)# [0.9281045751633987, 0.9326797385620915, 0.9349673202614379]
    print(np.mean(accuracies))# (''+''+'')/3= 0.931917211328976
    starttime = datetime.datetime.now()# starttime= datetime.datetime(2023, 9, 22, 19, 46, 55, 977843)

    f = open("result_" + dataset + ".txt", "a")# f= <_io.TextIOWrapper name='result_amazon-photos.txt' mode='a' encoding='cp936'>
    i = 1
    f.write(str(np.mean(accuracies)) + str(starttime)+str(i) + "\n")
    i = i+1
    f.close()
    return accuracies


def fit_logistic_regression_preset_splits(X, y, train_masks, val_masks, test_mask,dataset):
    # transfrom targets to one-hot vector
    one_hot_encoder = OneHotEncoder(categories='auto', sparse=False)
    y = one_hot_encoder.fit_transform(y.reshape(-1, 1)).astype(np.bool)

    # normalize x
    X = normalize(X, norm='l2')

    accuracies = []
    for split_id in range(train_masks.shape[1]):
        # get train/val/test masks
        train_mask, val_mask = train_masks[:, split_id], val_masks[:, split_id]

        # make custom cv
        X_train, y_train = X[train_mask], y[train_mask]
        X_val, y_val = X[val_mask], y[val_mask]
        X_test, y_test = X[test_mask], y[test_mask]

        # grid search with one-vs-rest classifiers
        best_test_acc, best_acc = 0, 0
        for c in 2.0 ** np.arange(-10, 11):
            clf = OneVsRestClassifier(LogisticRegression(solver='liblinear', C=c))
            clf.fit(X_train, y_train)

            y_pred = clf.predict_proba(X_val)
            y_pred = np.argmax(y_pred, axis=1)
            y_pred = one_hot_encoder.transform(y_pred.reshape(-1, 1)).astype(np.bool)
            val_acc = metrics.accuracy_score(y_val, y_pred)
            if val_acc > best_acc:
                best_acc = val_acc
                y_pred = clf.predict_proba(X_test)
                y_pred = np.argmax(y_pred, axis=1)
                y_pred = one_hot_encoder.transform(y_pred.reshape(-1, 1)).astype(np.bool)
                best_test_acc = metrics.accuracy_score(y_test, y_pred)

        accuracies.append(best_test_acc)
    print(np.mean(accuracies))
    f = open("result_" + dataset +  ".txt", "a")
    f.write(str(np.mean(accuracies))+"\n")
    f.close()
    return accuracies
