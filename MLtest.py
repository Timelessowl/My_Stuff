import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.neighbors import KNeighborsRegressor
from sklearn.linear_model import  *
from sklearn.metrics import mean_squared_error


def Pandas_output_init():

    pd.options.display.expand_frame_repr = False
    pd.options.display.max_rows = None
    pd.options.display.max_columns = None
    pd.options.display.max_colwidth = None


def DrawPlot(x, y, p):
    plt.scatter(x[:, p], x[:, 0])
    plt.xlabel("B")
    plt.ylabel("Crime rate")

    plt.show()



def main():

    # Pandas_output_init()
    # data = pd.read_csv("math_students.csv", delimiter= ',')
    data = load_boston()
    x, y = data['data'], data['target']

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2)


    knn = KNeighborsRegressor(n_neighbors=5, weights='uniform', p=2)
    knn.fit(x_train, y_train)
    prediction = knn.predict(x_test)
    print(mean_squared_error(y_test, prediction))

    grid_searcher = GridSearchCV(KNeighborsRegressor(),
                                 param_grid={'n_neighbors': range(1, 40, 2),
                                             'weights': ['uniform', 'distance'],
                                             'p': [1, 2, 3]},
                                 cv=5)
    grid_searcher.fit(x_train, y_train)
    best_predict = grid_searcher.predict(x_test)
    print(mean_squared_error(y_test, best_predict))






if __name__ == "__main__":
    main()