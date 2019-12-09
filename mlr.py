from sklearn.datasets import make_regression
import numpy as np
from numpy import corrcoef, std, mean, reshape, transpose, linalg, dot, insert
from mpl_toolkits import mplot3d
from matplotlib import pyplot 


# multiple linear regression is slightly more tricky than simple linear regression but there is still a 
# closed form solution for the estimation of parameters. 

#currently only works for 3 features + intercept, any more parameters makes it hard to visualise

X, y, coefs = make_regression(n_samples = 100,  n_features = 2, n_informative = 1, noise = 8, coef = True)

# insert column of 1s before independant values to estimate intercept
X = insert(X, 0, 1, axis = 1)
print('X', X)

X_transpose = transpose(X)
b_hat_1 = linalg.inv(dot(X_transpose, X))
b_hat_2 = dot(X_transpose, y)
b_hat = dot(b_hat_1, b_hat_2)


print('estimated coefs', b_hat)
print('actual coefs', coefs)

#contour plotting - I can use this to plot the regression plane later
#making prediction based on my estimates of the parameters 
def f(x, z, b_hat):
	  return b_hat[0] + b_hat[1]*x + b_hat[2]*z

x_flat, z_flat = np.meshgrid(X[:, 1].flatten(), X[:, 2].flatten())
y_hat = f(x_flat, z_flat, b_hat)

fig = pyplot.figure()
ax = pyplot.axes(projection='3d')

ax.contourf3D(x_flat, z_flat, y_hat, 50)
ax.scatter3D(X[:, 1].flatten(), X[:, 2].flatten(), y)
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('z')
pyplot.show()