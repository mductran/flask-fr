import numpy as np
from matplotlib import pyplot as plt
from sklearn.svm import SVC

mu_vec1 = np.array([0, 0])
cov_mat1 = np.array([[2, 0], [0, 2]])
x1_samples = np.random.multivariate_normal(mu_vec1, cov_mat1, 100)
mu_vec1 = mu_vec1.reshape(1, 2).T  # to 1-col vector

mu_vec2 = np.array([1, 2])
cov_mat2 = np.array([[1, 0], [0, 1]])
x2_samples = np.random.multivariate_normal(mu_vec2, cov_mat2, 100)
mu_vec2 = mu_vec2.reshape(1, 2).T

fig = plt.figure()

plt.scatter(x1_samples[:, 0], x1_samples[:, 1], marker='+')
plt.scatter(x2_samples[:, 0], x2_samples[:, 1], c='green', marker='o')

X = np.concatenate((x1_samples, x2_samples), axis=0)
Y = np.array([0] * 100 + [1] * 100)

C = 1.0  # SVM regularization parameter
clf = SVC(kernel='linear', gamma=0.7, C=C)
clf.fit(X, Y)

# plot linear classifier
w = clf.coef_[0]
a = -w[0] / w[1]
xx = np.linspace(-5, 5)
yy = a * xx - (clf.intercept_[0]) / w[1]

plt.plot(xx, yy, 'k-')

# plot non-linear classifier
C = 1.0  # SVM regularization parameter
clf = SVC(kernel='rbf', gamma=0.7, C=C)
clf.fit(X, Y)

h = .02  # step size in the mesh
# create a mesh to plot in
x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                     np.arange(y_min, y_max, h))

# Plot the decision boundary. For that, we will assign a color to each
# point in the mesh [x_min, m_max]x[y_min, y_max].
Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])

# Put the result into a color plot
Z = Z.reshape(xx.shape)
plt.contour(xx, yy, Z, cmap=plt.cm.Paired)
