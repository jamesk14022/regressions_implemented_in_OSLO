from sklearn.datasets import make_regression
from mpl_toolkits import mplot3d
from matplotlib import pyplot
import numpy as np


# create a simple non linear relationship between X and y
def f(x):
	return np.random.normal(x**2, 500)

# TPS corresponded to the radisl basis kernal
def rbk(r):
	return r**2 + np.log(r)
	

X = Z = np.array([np.arange(100)])
y = f(X)

print('x', X.flatten())
print('y', y.flatten())

fig = pyplot.figure(1)
ax = pyplot.axes(projection = '3d')
ax.scatter(X.flatten(), Z.flatten(), y)
pyplot.show()

# polyharmonic spline is a linear combination of radial basis functions 
