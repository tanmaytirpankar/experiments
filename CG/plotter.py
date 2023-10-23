# Program to create a 2D line plot from coordinates in a CSV file

import matplotlib.pyplot as plt
import csv

x = []
y = []

with open('../build-debug/data_float.csv', 'r') as csvfile:
    plots = csv.reader(csvfile, delimiter=',')
    for row in plots:
        x.append(float(row[0]))
        y.append(float(row[1]))

plt.plot(x, y, label='float!')

x = []
y = []
with open('../build-debug/data_double.csv', 'r') as csvfile:
    plots = csv.reader(csvfile, delimiter=',')
    for row in plots:
        x.append(float(row[0]))
        y.append(float(row[1]))
plt.plot(x, y, label='double!')

plt.xlabel('x')
plt.ylabel('y')
plt.title('Interesting Graph\nCheck it out')
plt.legend()
plt.show()