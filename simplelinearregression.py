from sklearn.datasets import make_regression
import numpy as np
from numpy import corrcoef, std, mean, reshape
from matplotlib import pyplot 
X, y, coefs = make_regression(n_samples = 70,  n_features = 1, noise = 5, coef = True)

# simple regression with one parameter

# in this case, the intercept term and the parameter coeffecient estimation are the answers to a simple 
# miniisation problem, a problem which reduces to simple formula, which can be seen here https://en.wikipedia.org/wiki/Simple_linear_regression

# we need, 

# sample correlation coefficent between x and y
# uncorrected sample standard deviations from x and y
# mean of x
# mean of y


y = reshape(y, (-1, 1))

sample_corr_co = corrcoef(X.flatten(), y.flatten())[0 , 1]
print('corr coef', sample_corr_co)
X_std = std(X)
print('X std', X_std)
y_std = std(y)
print('y std', y_std)
X_mean = mean(X)
y_mean = mean(y)


b_hat = sample_corr_co * (y_std / X_std)
a_hat = y_mean - b_hat * X_mean
y_hat_a = a_hat + b_hat * X

print('Estimated slope is ', b_hat)
print('True slope is ', coefs)

pyplot.scatter(X, y)
pyplot.plot(X, y_hat_a, 'r')
pyplot.show()
