
import numpy as np  # linear algebra
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt  # for plotting
import seaborn as sns  # for plotting again... for advanced plots.
import os

for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# here we load the data in data frame of python.
df = pd.read_csv('train.csv')  # loading the data
df.head(5)  # lets see the first 5 rows of the datase

# загрузка наших данных
df_My = pd.read_csv('test.csv')  # loading the data
df_My.head(1)  # просмотр данных

from scipy import stats

z = np.abs(stats.zscore(df))
df_outliers = df[(z >= 3.5).any(axis=1)]  # it says that any of the column with value of z above 3
print(df_outliers.shape)
df_clean = df[(z < 3.5).all(axis=1)]  # pay attention to the condition.
# df_clean.shape
# (6, 14)
#
# (297, 14)
# [4]
# data = pd.get_dummies(df_clean,columns =['cp','restecg','slope','ca','thal'])
# data_My = pd.get_dummies(df_My ,columns =['cp','restecg','slope','ca','thal'])

# numeric_cols=['age','trestbps','chol','thalach','oldpeak']
# from sklearn.preprocessing import StandardScaler
# standardScaler = StandardScaler()
# data[numeric_cols] = standardScaler.fit_transform(data[numeric_cols])
# data_My[numeric_cols] = standardScaler.fit_transform(data_My[numeric_cols])

y = df['target']
y = np.array(y)
x = df.drop(columns=['target'])
x_My = df_My
y_My = pd.read_csv('solution.csv')
# x.columns
# Index(['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg', 'thalach',
#        'exang', 'oldpeak', 'slope', 'ca', 'thal'],
#       dtype='object')
print(x_My.columns)


from sklearn.model_selection import KFold

kf = KFold(n_splits=5)
kf.get_n_splits(x)

# metrics for SVM
SVM_accuracy = []
SVM_precision = []
SVM_recall = []

# metrics for CatBoost
CatBoost_accuracy = []
CatBoost_precision = []
CatBoost_recall = []

# metrics for Random Forest
RF_accuracy = []
RF_precision = []
RF_recall = []

# metrics for KNN
KNN_accuracy = []
KNN_precision = []
KNN_recall = []

# metrics for Logistic Regression
LR_accuracy = []
LR_precision = []
LR_recall = []

# metrics for CatBoost
AdaBoost_accuracy = []
AdaBoost_precision = []
AdaBoost_recall = []
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from catboost import CatBoostClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.feature_selection import SelectFromModel
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import ExtraTreesClassifier

# importing libraries of performance Metrics
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score

# Making the classifier Objects
clf_svm = SVC()  # SVM object
clf_rf = RandomForestClassifier(max_depth=100, random_state=0)  # Random Forest Object
clf_knn = KNeighborsClassifier(n_neighbors=30)  # KNN object
clf_lr = LogisticRegression(C=1, class_weight=None, penalty='l2', solver='newton-cg')  # Logistic regression model
clf_CatBoost = CatBoostClassifier(iterations=50, learning_rate=0.01, depth=3)
clf_AdaBoost = AdaBoostClassifier()

i = 1  # count the number of folds

for train_index, test_index in kf.split(x):

    print("\nNumber of fold: %d" % i)
    i += 1
    # Splitting the data
    X_train, X_test = x[train_index], x_My
    y_train, y_test = y[train_index], y_My
    print("test")
    # Training and Evaluating SVM
    model = clf_svm.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    # качество алгоритма
    SVM_accuracy.append(accuracy_score(y_test, y_pred) * 100)
    SVM_precision.append(precision_score(y_test, y_pred, pos_label=1) * 100)
    SVM_recall.append(recall_score(y_test, y_pred, pos_label=1) * 100)
    # print("Working on SVM", i)

    # Training and Evaluating CatBoost
    model = clf_CatBoost.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    CatBoost_accuracy.append(accuracy_score(y_test, y_pred) * 100)
    CatBoost_precision.append(precision_score(y_test, y_pred, pos_label=1) * 100)
    CatBoost_recall.append(recall_score(y_test, y_pred, pos_label=1) * 100)
    # print("Working on CatBoost", i)

    # Training and Evaluating AdaBoost
    model = clf_AdaBoost.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    AdaBoost_accuracy.append(accuracy_score(y_test, y_pred) * 100)
    AdaBoost_precision.append(precision_score(y_test, y_pred, pos_label=1) * 100)
    AdaBoost_recall.append(recall_score(y_test, y_pred, pos_label=1) * 100)

    # Training and Evaluating Random Forest
    model = clf_rf.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    RF_accuracy.append(accuracy_score(y_test, y_pred) * 100)
    RF_precision.append(precision_score(y_test, y_pred, pos_label=1) * 100)
    RF_recall.append(recall_score(y_test, y_pred, pos_label=1) * 100)
    # print("Working on Random Forest")

    # Training and Evaluating KNN
    model = clf_knn.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    KNN_accuracy.append(accuracy_score(y_test, y_pred) * 100)
    KNN_precision.append(precision_score(y_test, y_pred, pos_label=1) * 100)
    KNN_recall.append(recall_score(y_test, y_pred, pos_label=1) * 100)
    # print("Working on KNN")

    # Training and Evaluating LR
    model = clf_lr.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    LR_accuracy.append(accuracy_score(y_test, y_pred) * 100)
    LR_precision.append(precision_score(y_test, y_pred, pos_label=1) * 100)
    LR_recall.append(recall_score(y_test, y_pred, pos_label=1) * 100)
    # print("Working on Logistic Regression")
