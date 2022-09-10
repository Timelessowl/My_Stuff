import numpy as np
import pandas as pd

def Roll():
    _result = np.random.randint(1, 7)
    if (_result == 1 or _result == 2):
        return "You Loser! You rolled {}".format(_result)
    elif (_result in range(3, 6)):
        return "Not Bad! You rolled {}".format(_result)
    else:
        return "Lucker! You rolled {}".format(_result)

def main():
    rolls = int(input("Please enter the number of rolls You want to make "))

    if (rolls == 0):
        print("Then why the fuck You opened the dice programm")
        return
    if (rolls != 1):
        summ = 0
        maxRoll = 0
        minRoll = 7

    for i in range(rolls):
        res = Roll()
        print(res)
        summ += int(res[-1])
        if int(res[-1]) > maxRoll:
            maxRoll = int(res[-1])
        elif int(res[-1]) < minRoll:
            minRoll = int(res[-1])

    print("Max You rolled is {}".format(maxRoll), "\nMin You rolled is {}".format(minRoll))
    print("Summary You rolled {} points".format(summ))
    print("Fuck You, I'm Closing now")

if __name__ == "__main__":
    main()
