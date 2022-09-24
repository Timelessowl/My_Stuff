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



def main():

    data = load_boston()
    x, y = data['data'], data['target']
    print(x.shape)
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

    knn = KNeighborsRegressor(n_neighbors=5, weights='uniform', p=2)
    linear = LinearRegression()
    ridge = Ridge()
    lasso = Lasso()
    knn.fit(x, y)
    pred = knn.predict(x)
    print(mean_squared_error(y, pred))
    # print(mean_absolute_error(y, pred))
    grid_searcher = GridSearchCV(KNeighborsRegressor(),
                                 param_grid={'n_neighbors': range(1, 40, 2),
                                             'weights': ['uniform', 'distance'],
                                             'p': [1, 2, 3]},
                                 cv=5)
    # grid_searcher.fit(x_train, y_train)
    # print(grid_searcher.best_params_)
    # grid_searcher.fit(x, y)
    # print(grid_searcher.best_params_)
    # best_pred = grid_searcher.predict(x)
    # print(mean_squared_error(y, best_pred))
    linear.fit(x, y)
    linPred = linear.predict(x)
    print(mean_squared_error(y, linPred))
    ridge.fit(x, y)
    lasso.fit(x, y)
    rPred, lPred = ridge.predict(x), lasso.predict(x)
    print(mean_squared_error(y, rPred))
    print(mean_squared_error(y, lPred))





if __name__ == "__main__":
    main()
