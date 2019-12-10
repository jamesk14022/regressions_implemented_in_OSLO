from sklearn.datasets import make_regression
from scipy.stats import norm
import numpy as np
import math
from numpy import corrcoef, std, mean, reshape, transpose, linalg, dot, insert, diag, sqrt, mean, nditer
from mpl_toolkits import mplot3d
from matplotlib import pyplot 

# multiple linear regression is slightly more tricky than simple linear regression but there is still a 
# closed form solution for the estimation of parameters. 

#currently only works for 3 features + intercept, any more parameters makes it hard to visualise

n = 100
p = 2

def f(x, z, b_hat):
	return b_hat[0] + b_hat[1]*x + b_hat[2]*z

def res_sum_squares(y, X, b_hat):
	res = y - dot(X, b_hat)
	return dot(transpose(res), res)

# not working
def tot_sum_squares(y):
	return dot(y - mean(y), transpose(y - mean(y)))

# OLS estimate for var
def res_var(rss, n):
	return rss / (n - p)

# returns covariance matrix of b_hat
def covar_b_hat(inv, var):
	return var * inv

# intervals calculated using the quantile function of the standard normal distribution
def conf_intervals(b_hat, var_est, inverse_design, certainty):
	intervals = []
	quant = norm.ppf(1 - (certainty / 2)) 
	for idx, j in enumerate(b_hat):
		abs = quant * math.sqrt(var_est * diag(inverse_design)[idx])
		intervals.append([j - quant, j + quant])
	return intervals 

X, y, coefs = make_regression(n_samples = n,  n_features = p, n_informative = 2, noise = 8, coef = True)

# insert column of 1s before independant values to estimate intercept
X = insert(X, 0, 1, axis = 1)
print('X', X)

#closed form solution for parameter estimation
X_transpose = transpose(X)
inverse_design = linalg.inv(dot(X_transpose, X))
b_hat = dot(inverse_design, dot(X_transpose, y))


print('estimated coefs', b_hat)
print('actual coefs', coefs)

#contour plotting - I can use this to plot the regression plane later
#making prediction based on my estimates of the parameters 
x_flat, z_flat = np.meshgrid(X[:, 1].flatten(), X[:, 2].flatten())
y_hat = f(x_flat, z_flat, b_hat)

print('y', y.shape)

rss = res_sum_squares(y, X, b_hat)
tss = tot_sum_squares(y)
r_sq = 1 - rss / tss
var_est = res_var(rss, n)
covar_matrix = covar_b_hat(inverse_design, var_est)
se_estimators = sqrt(diag(covar_matrix))

conf = conf_intervals(b_hat, var_est, inverse_design, 0.05)
print('STDe', se_estimators)
print('RSS', rss)
print('TSS', tss)
print('R Squared', r_sq)
print('95% Confidence intervals', conf)

fig = pyplot.figure()
ax = pyplot.axes(projection='3d')

ax.contourf3D(x_flat, z_flat, y_hat, 50)
ax.scatter3D(X[:, 1].flatten(), X[:, 2].flatten(), y)
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('z')
pyplot.show()

# doesnt seem to perform well (in terms of r squared) if the independant variable have
# small absolute estimated parameters

#se of intercept and coefficients 
#confidence intervals of predictions 

