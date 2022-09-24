import numpy as np
from sklearn.datasets import load_boston, make_moons
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split

def main():
    x, y = make_moons(n_samples=1000, noise=0.5, random_state=10)
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.5)



if __name__ == "__main__":
    main()
