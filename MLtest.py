import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_boston, make_moons
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.neighbors import KNeighborsRegressor
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.metrics import mean_squared_error, mean_absolute_error


def Pandas_output_init():

    pd.options.display.expand_frame_repr = False
    pd.options.display.max_rows = None
    pd.options.display.max_columns = None
    pd.options.display.max_colwidth = None


def DrawPlot(array):
    plt.plot(range(1, 40, 2), array)
    plt.ylabel('Negative mean squared error')
    plt.xlabel('Number of neighbors')

    plt.show()


def KnnPredictSqErr(x_train, y_train, x_test, y_test, n, weights, p):
    _knn = KNeighborsRegressor(n_neighbors=n, weights=weights, p=p)
    _knn.fit(x_train, y_train)
    prediction = _knn.predict(x_test)
    return mean_squared_error(y_test, prediction)


def LinearPredictSqErr(type, x_train, y_train, x_test, y_test):
    if type == "Ridge":
        linear = Ridge()
    elif type =="Lasso":
        linear = Lasso()
    elif type == "Linear":
        linear = LinearRegression()

    linear.fit(x_train, y_train)
    pred = linear.predict(x_test)
    return mean_squared_error(y_test, pred)


def main():

    data = load_boston()
    x, y = data['data'], data['target']
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

    # print(mean_absolute_error(y, pred))
    # grid_searcher = GridSearchCV(KNeighborsRegressor(),
    #                              param_grid={'n_neighbors': range(1, 40, 2),
    #                                          'weights': ['uniform', 'distance'],
    #                                          'p': [1, 2, 3]},
    #                              cv=5)
    # grid_searcher.fit(x_train, y_train)
    # print(grid_searcher.best_params_)
    # grid_searcher.fit(x, y)
    # print(grid_searcher.best_params_)
    # best_pred = grid_searcher.predict(x)
    # print(mean_squared_error(y, best_pred))
    print(KnnPredictSqErr(x_train, y_train, x_test, y_test, 5, 'uniform', 2))
    print(LinearPredictSqErr("Linear", x_train, y_train, x_test, y_test))
    print(LinearPredictSqErr("Ridge", x_train, y_train, x_test, y_test))
    print(LinearPredictSqErr("Lasso", x_train, y_train, x_test, y_test))


if __name__ == "__main__":
    main()
