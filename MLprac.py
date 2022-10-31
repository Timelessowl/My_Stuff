import numpy as np
import pandas as pd
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score


def Pandas_output_init():

    # sys.stdout = open("result", 'a')
    pd.options.display.expand_frame_repr = False
    pd.options.display.max_rows = None
    pd.options.display.max_columns = None
    pd.options.display.max_colwidth = None


def main():

    Pandas_output_init()

    data = pd.read_csv("heart.csv")
    x_train, x_test, y_train, y_test = train_test_split(data.drop('target', axis=1), data['target'], test_size=0.25, random_state=13)
    # clf = SGDClassifier( max_iter=1000, learning_rate="constant", random_state=13,eta0=0.1, alpha=0, loss="log")
    clf = SGDClassifier(penalty="l1", max_iter=1000, learning_rate="optimal", random_state=13, alpha=0.1, loss="log")
    clf.fit(x_train, y_train)
    pred = clf.predict_proba(x_test)
    print(data.columns)
    print(clf.coef_)
    # print(accuracy_score(y_test, pred))
    # print(np.sqrt(sum(np.square(clf.coef_[0][:]))))


if __name__ == "__main__":
    main()
