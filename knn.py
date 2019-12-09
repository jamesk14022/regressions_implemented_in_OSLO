from sklearn.datasets import make_regression
from mpl_toolkits import mplot3d
from matplotlib import pyplot
import numpy as np

X, y, coefs = make_regression(n_samples = 50, n_features = 2, n_informative = 2, coef = True)

print('x co ord vector', X[:, 0].flatten())
print('x2 co ord vector', X[:, 1].flatten())

# decide on range of values to be covered by the map
x_flat, y_flat = np.meshgrid(X[:, 0].flatten(), X[:, 1].flatten())


print('X', x_flat.shape)
print('y', y_flat.shape)


def distance_2d(x_point, y_point, x, y):
	return np.hypot(x_point - x, y_point - y)

def make_y_prediction(x_flat, y_flat, X, y):
	# treat as x, y
	dist = []
	k = 12
	for i, f in np.nditer([X[:, 0], X[:, 1]]):
		dist.append(distance_2d(i, f, x_flat, y_flat))
		#index of k smallest values
	idx = np.argpartition(dist, 3)
	return np.sum(y[idx[:k]]) / k

y_preds = []
for x , m in np.nditer([x_flat, y_flat]):
	y_preds.append(make_y_prediction(x, m, X, y))

y_preds = np.array(y_preds).reshape(50, 50)

fig = pyplot.figure()
ax = pyplot.axes(projection='3d')
ax.contour3D(x_flat, y_flat, y_preds, 50, cmap = 'binary')
ax.scatter3D(X[:, 0].flatten(), X[:, 1].flatten(), y)
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('z')
pyplot.show()