import numpy as np

def Roll():
    _result = np.random.randint(1,6)
    if (_result == 1 or _result == 2):
        return "You Loser! You rolled {}".format(_result)
    elif (_result in range(3, 6) ):
        return "Not Bad! You rolled {}".format(_result)
    else:
        return "Lucker! You rolled {}".format(_result)


def __main__():

    rolls = int(input("Please enter the number of rolls You want to make "))
    if (rolls == 0):
        print("Then why the fuck You opened the dice programm")
        return
    if (rolls != 1):
        maxRoll = 0
        minRoll = 7

    for i in range(rolls):
        res = Roll()
        print(res)
        # if int(res[-1]) > maxRoll:
        #     maxRoll = res[-1]
        # elif res[-1] < minRoll:
        #     minRoll = res[-1]

    # results = [np.random.randint(1, 6) for i in range(rolls)]
    # counter = 5
    # while (results != []):
    #     print(i, end = ' ')
    #     print(test[i])
    #     #print("You rolled ", results[-1])
    #
    #     match results[-1]:
    #         case 1:
    #             pass
    #         case 2:
    #             pass
    #         case 3:
    #             continue
    #         case 4:
    #             continue
    #         case 5:
    #             pass
    #         case 6:
    #             pass
    #     test.pop(-1)

    print("Max You rolled is {}".format(maxRoll), "\n Min You rolled is {}".format(minRoll))
    print("Fuck You, I'm Closing now")


__main__()