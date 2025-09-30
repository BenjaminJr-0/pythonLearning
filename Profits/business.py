import numpy as np
import csv
from matplotlib import pyplot as plt
import copy
import math

x_train = []
y_train = []

with open("train.csv", 'r') as file:
    csvreader = csv.reader(file)
    next(csvreader)

    for row in csvreader:
        x_train.append(row[4])
        y_train.append(row[80])
        



print("Type of x_train:",type(x_train))
print("First five elements of x_train are:\n", x_train[:5])

print("Type of y_train:",type(y_train))
print("The first five elements of y_train are:\n",y_train[:5])

#rint ('The shape of x_train is:', x_train.shape)
#rint ('The shape of y_train is: ', y_train.shape)
print ('Number of training examples (m):', len(x_train))

#plots the x and y
plt.plot(x_train, y_train)
#displays the graph
plt.show
#saves to a graph
plt.savefig('map.png')
#displays a scratter graph
plt.scatter(x_train, y_train, marker='X', c='r')
plt.title("Profits Vs. Population by City")
plt.ylabel('Profits')
plt.xlabel('Population')
plt.savefig('scatter.png')