#The optimal values of m and b can be actually calculated with way less effort than doing a linear regression.
#this is just to demonstrate gradient descent

from numpy import *
import matplotlib.pyplot as plt

def run():
    points = genfromtxt("classification_train.txt", skip_header=1);
    testingData = genfromtxt("classification_test.txt", skip_header=1);
    testingDataWithoutLabel = genfromtxt("classification_test.txt", skip_header=1, usecols=(0,1));

    classOne = []
    classZero = []
    classOneTesting = []
    classZeroTesting = []

    countOneTrain = 0
    countZeroTrain = 0

    countOneTest = 0
    countZeroTest = 0

    sumXclassOne = 0
    sumYclassOne = 0
    sumXclassZero = 0
    sumYclassZero = 0

    for j in range(len(points)):
        if (points[j][2] == 1):
            temp = []
            temp.append(points[j][0])
            temp.append(points[j][1])
            classOne.append(temp)
            sumXclassOne = sumXclassOne + points[j][0]
            sumYclassOne = sumYclassOne + points[j][1]
            countOneTrain = countOneTrain + 1;
        else:
            temp = []
            temp.append(points[j][0])
            temp.append(points[j][1])
            classZero.append(temp)
            sumXclassZero = sumXclassZero + points[j][0]
            sumYclassZero = sumYclassZero + points[j][1]
            countZeroTrain = countZeroTrain + 1;

    probabilityOne = countOneTrain / (countOneTrain + countZeroTrain)
    probabilityZero = countZeroTrain / (countOneTrain + countZeroTrain)

    ## CALCULATING MEAN HERE

    sumXclassOne = sumXclassOne / countOneTrain
    sumYclassOne = sumYclassOne / countOneTrain

    sumXclassZero = sumXclassZero / countZeroTrain
    sumYclassZero = sumYclassZero / countZeroTrain

    meanClassOne = [[sumXclassOne],[sumYclassOne]]
    meanClassZero = [[sumXclassZero], [sumYclassZero]]

    ## COVARIANCE NOW

    covX01classOne = 0
    covY10classOne = 0

    covX00classOne = 0
    covY11classOne = 0

    for n in range(len(classOne)):
        ## i = 0 k = 1
        covX01classOne = covX01classOne + (classOne[n][0] - meanClassOne[0][0]) \
                                        * (classOne[n][1] - meanClassOne[1][0])
        ## i = 0 k = 0
        covX00classOne = covX00classOne + (classOne[n][0] - meanClassOne[0][0]) ** 2

        ## i = 1 k = 1
        covY11classOne = covY11classOne + (classOne[n][1] - meanClassOne[1][0]) ** 2

    covY11classOne = covY11classOne / len(classOne)
    covX01classOne = covX01classOne / len(classOne)
    covX00classOne = covX00classOne / len(classOne)

    ## CLASSZERO COVARIANCE

    covX01classZero = 0
    covY10classZero = 0

    covX00classZero = 0
    covY11classZero = 0

    for n in range(len(classZero)):
        ## i = 0 k = 1
        covX01classZero = covX01classZero + (classZero[n][0] - meanClassZero[0][0]) \
                         * (classZero[n][1] - meanClassZero[1][0])
        ## i = 0 k = 0
        covX00classZero = covX00classZero + (classZero[n][0] - meanClassZero[0][0]) ** 2

        ## i = 1 k = 1
        covY11classZero = covY11classZero + (classZero[n][1] - meanClassZero[1][0]) ** 2

    covY11classZero = covY11classZero / len(classZero)
    covX01classZero = covX01classZero / len(classZero)
    covX00classZero = covX00classZero / len(classZero)

    # MAKING THEM MATRIX

    covarianceClassOne = [[covX00classOne, covX01classOne], [covX01classOne, covY11classOne]]
    covarianceClassZero = [[covX00classZero, covX01classZero], [covX01classZero, covY11classZero]]

    print("Covariance matrix for class one")
    print(covarianceClassOne)
    print("Covariance matrix for class zero")
    print(covarianceClassZero)

    print("Mean vector for class one")
    print(meanClassOne)
    print("Mean vector for class zero")
    print(meanClassZero)

    accuracy = 0

    for i in range(len(testingData)):
        valueOfFunctionOne = discriminantClassOne(testingDataWithoutLabel[i], meanClassOne, covarianceClassOne, probabilityOne)
        valueOfFunctionZero = discriminantClassZero(testingDataWithoutLabel[i], meanClassZero, covarianceClassZero, probabilityZero)

        if(valueOfFunctionOne > valueOfFunctionZero):
            if(testingData[i][2] == 1):
                print("Prediction of data #",i, "is label one and its correct")
                accuracy = accuracy + 1
            else:
                print("Prediction of data #",i, "is label one BUT ITS WRONG")

        if(valueOfFunctionZero > valueOfFunctionOne):
            if(testingData[i][2] == 0):
                print("Prediction of data #",i, "is label zero and its correct")
                accuracy = accuracy + 1
            else:
                print("Prediction of data #",i, "is label zero BUT ITS WRONG")

    print("ACCURACY IS FUCKIN ", accuracy / len(testingData))

def discriminantClassOne(x, meanClassOne, covarianceClassOne, probabilityOne):
    transposedMean = transpose(meanClassOne)
    inverseCovariance = linalg.inv(matrix(covarianceClassOne))
    lnP = log(probabilityOne)

    return (matmul(matmul(transposedMean, inverseCovariance), x) + lnP - 0.5 * (matmul(matmul(transposedMean, inverseCovariance), meanClassOne)))

def discriminantClassZero(x, meanClassZero, covarianceClassZero, probabilityZero):
    transposedMean = transpose(meanClassZero)
    inverseCovariance = linalg.inv(matrix(covarianceClassZero))
    lnP = log(probabilityZero)

    return (matmul(matmul(transposedMean, inverseCovariance), x) + lnP - 0.5 * (matmul(matmul(transposedMean, inverseCovariance), meanClassZero)))


if __name__ == '__main__':
    run()
