import numpy as np

rolls = int(input("please enter the number of rolls You want to make "))
if (rolls == 0):
    print("Then why the fuck You opened the dice programm")
    exit(0)

results = [np.random.randint(1, 6) for i in range(rolls)]

if (rolls == 1):
    print("You rolled ", results[0])
else :
    while (results != []):

        for i in range(rolls):
            print(results[i])
        match rolls:
            case 1:
                pass
            case 2:
                pass


print("test")
print("Fuck You")