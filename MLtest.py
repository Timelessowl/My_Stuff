import sys
import pandas as pd
import matplotlib.pyplot as plt



def Pandas_output_init():

    pd.options.display.expand_frame_repr = False
    pd.options.display.max_rows = None
    pd.options.display.max_columns = None
    pd.options.display.max_colwidth = None


def DrowPlot(data):
    plt.figure(figsize=(10, 7))
    plt.title('Absences distribution')
    data['absences'].hist()
    plt.xlabel('absences')
    plt.ylabel('number of students')
    plt.show()



def main():

    Pandas_output_init()
    data = pd.read_csv("math_students.csv", delimiter= ',')



    # data['alc'] = (5 * data['Dalc'] + 2 * data['Walc']) / 7
    # print(data['activities'].mode())
    # DrowPlot(data)
    # print("Сред оценка хикк", data[data['romantic'] == 'no']['G3'].mean())
    # print("Сред оценка пар", data[data['romantic'] == 'yes']['G3'].mean())
    # print("их разность", data[data['romantic'] == 'no']['G3'].mean() - data[data['romantic'] == 'yes']['G3'].mean())
    # mostcommon = data['activities'].value_counts().idxmax()
    # print(data[data['activities'] == mostcommon]['absences'].value_counts())
    # print(mostcommon)
    print(data[data['school'] == 'MS']['age'].min())



if __name__ == "__main__":
    main()