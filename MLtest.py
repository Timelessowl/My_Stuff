import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_boston, make_moons
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score, cross_validate
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
        linear = Ridge(alpha=0.1)
    elif type == "Lasso":
        linear = Lasso(alpha=0.1)
    elif type == "Linear":
        linear = LinearRegression()

    linear.fit(x_train, y_train)
    prediction = linear.predict(x_test)
    return mean_squared_error(y_test, prediction), mean_absolute_error(y_test, prediction), prediction[2999]


def main():
    Pandas_output_init()
    data_train = pd.read_csv("california_housing_train.csv")
    data_test = pd.read_csv("california_housing_test.csv")
    x_train, x_test = data_train.drop(['median_house_value'], axis=1), data_test.drop(["median_house_value"], axis=1)
    y_train, y_test = data_train["median_house_value"], data_test['median_house_value']
    print(y_test.tail(5))
    # print("X_TRAIN:\n\r", x_train.head(10))
    # print("Y_TRAIN:\n\r", y_train.head(10))
    # print("X_TEST:\n\r", x_test.head(10))
    # print("Y_TEST:\n\r", y_test.head(10))
    # x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

    # knn = KNeighborsRegressor(n_neighbors=5, weights='uniform', p=2)
    #
    # grid_searcher = GridSearchCV(KNeighborsRegressor(),
    #                              param_grid={'n_neighbors': range(10, 1000, 10),
    #                                          'weights': ['uniform', 'distance'],
    #                                          'p': [1, 2, 3]},
    #                              cv=5)
    # grid_searcher.fit(x_train, y_train)
    # print(grid_searcher.best_params_)
    # # grid_searcher.fit(x, y)
    # # print(grid_searcher.best_params_)
    # best_pred = grid_searcher.predict(x_test)
    # print(mean_squared_error(y_test, best_pred))

    # print(KnnPredictSqErr(x_train, y_train, x_test, y_test, 5, 'uniform', 2))
    print("Linear SqErr:", LinearPredictSqErr("Linear", x_train, y_train, x_test, y_test))
    print("Ridge SqErr:", LinearPredictSqErr("Ridge", x_train, y_train, x_test, y_test))
    print("Lasso SqErr:", LinearPredictSqErr("Lasso", x_train, y_train, x_test, y_test))
    # print(cross_validate(knn, x, y, cv=5))


if __name__ == "__main__":
    main()
