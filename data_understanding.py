__author__ = "Wendi Tan"
__email__ = "wtan37@ucsc.edu"
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


def box(path, usecols, lables):
    data = pd.read_excel(path, header=None, usecols=usecols)
    leave = []
    stay = []
    for i in range(1, data.shape[0]):
        if data.iloc[i, 1] == "LEAVE":
            leave.append(data.iloc[i, 0])
        else:
            stay.append(data.iloc[i, 0])
    fig1, ax1 = plt.subplots()
    ax1.set_xlabel(lables[1])
    ax1.set_ylabel(lables[0])
    ax1.set_title('Basic Plot')
    ax1.boxplot([leave, stay])
    plt.xticks([1, 2], ['LEAVE', 'STAY'])
    print('leave customers:', pd.DataFrame(leave).describe())
    print('stay customers:', pd.DataFrame(stay).describe())
    plt.show()


def data_explore(df, var1, var2):
    leave = []
    stay = []
    for i in range(1, df.shape[0]):
        if df.iloc[i, var2] == "LEAVE":
            leave.append(df.iloc[i, var1])
        else:
            stay.append(df.iloc[i, var1])
    print(pd.DataFrame(leave).describe(datetime_is_numeric=False))
    print(pd.DataFrame(stay).describe(datetime_is_numeric=False))


def stack(df, var1, var2):  # var1=CONSIDERING_CHANGE_OF_PLAN, var2=chrun
    choice_leave = {}
    choice_stay = {}
    for i in range(1, df.shape[0]):
        if df.iloc[i, var2] == 'LEAVE':
            if not df.iloc[i, var1] in choice_leave:
                choice_leave[df.iloc[i, var1]] = 1
            else:
                choice_leave[df.iloc[i, var1]] += 1
        else:
            if not df.iloc[i, var1] in choice_stay:
                choice_stay[df.iloc[i, var1]] = 1
            else:
                choice_stay[df.iloc[i, var1]] += 1

    leave = [choice_leave[i] for i in sorted(choice_leave)]
    stay = [choice_stay[i] for i in sorted(choice_stay)]
    labels = sorted(choice_leave)
    print(choice_leave)

    width = 0.35  # the width of the bars: can also be len(x) sequence
    fig, ax = plt.subplots()
    ax.bar(labels, leave, width, label='leave')
    ax.bar(labels, stay, width, bottom=leave, label='stay')

    ax.set_ylabel('Number of people')
    ax.set_title('Churn distribution')
    ax.legend()

    plt.show()

if __name__ == '__main__':
    wb_path = "Customer_Churn.xlsx"
    df = pd.read_excel(wb_path, header=None)
    #box(wb_path, [6, 11], [df.iloc[0, 6], df.iloc[0, 11]])
    #stack(df, 10, 11)
    # print(pd.DataFrame([df.iloc[i, 1] for i in range(1, df.shape[0])]).describe())
    # print(pd.DataFrame(df).describe())
    data_explore(df, 8, 11)

