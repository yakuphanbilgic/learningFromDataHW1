#The optimal values of m and b can be actually calculated with way less effort than doing a linear regression.
#this is just to demonstrate gradient descent

from numpy import *
import matplotlib.pyplot as plt

def run():
    points = genfromtxt("classification_train.txt", skip_header=1);
    x = []
    y = []
    countZero = 0;
    countOne = 0;

    for i in range(len(points)):
        x.append(points[i][0]);
        y.append(points[i][1]);

    for j in range(len(points)):
        if (points[j][2] == 1):
            countOne = countOne + 1;
        else:
            countZero = countZero + 1;

    plt.scatter(x, y, s=2, c = 'purple')
    plt.xlabel('feature 1')
    plt.ylabel('feature 2')
    plt.show()

    print(countZero / (countZero + countOne))
    print(countOne / (countZero + countOne))

if __name__ == '__main__':
    run()