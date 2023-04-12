import numpy as np
import math
import matplotlib.pyplot as plt
from sklearn import svm, datasets


def plot_results(models, titles, X, y, plot_sv=False):
    # Set-up 2x2 grid for plotting.
    fig, sub = plt.subplots(1, len(titles))  # 1, len(list(models)))

    X0, X1 = X[:, 0], X[:, 1]
    xx, yy = make_meshgrid(X0, X1)

    if len(titles) == 1:
        sub = [sub]
    else:
        sub = sub.flatten()
    for clf, title, ax in zip(models, titles, sub):
        # print(title)
        plot_contours(ax, clf, xx, yy, cmap=plt.cm.coolwarm, alpha=0.8)
        ax.scatter(X0, X1, c=y, cmap=plt.cm.coolwarm, s=20, edgecolors="k")
        if plot_sv:
            sv = clf.support_vectors_
            ax.scatter(sv[:, 0], sv[:, 1], c="k", s=60)

        ax.set_xticks(())
        ax.set_yticks(())
        ax.set_title(title)
        ax.set_aspect("equal", "box")
    fig.tight_layout()
    plt.show()


def make_meshgrid(x, y, h=0.02):
    """Create a mesh of points to plot in

    Parameters
    ----------
    x: data to base x-axis meshgrid on
    y: data to base y-axis meshgrid on
    h: stepsize for meshgrid, optional

    Returns
    -------
    xx, yy : ndarray
    """
    x_min, x_max = x.min() - 1, x.max() + 1
    y_min, y_max = y.min() - 1, y.max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    return xx, yy


def plot_contours(ax, clf, xx, yy, **params):
    """Plot the decision boundaries for a classifier.

    Parameters
    ----------
    ax: matplotlib axes object
    clf: a classifier
    xx: meshgrid ndarray
    yy: meshgrid ndarray
    params: dictionary of params to pass to contourf, optional
    """
    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    out = ax.contourf(xx, yy, Z, **params)
    return out


C_hard = 1000000.0  # SVM regularization parameter
C = 10
n = 100


# Data is labeled by a circle

radius = np.hstack([np.random.random(n), np.random.random(n) + 1.5])
angles = 2 * math.pi * np.random.random(2 * n)
X1 = (radius * np.cos(angles)).reshape((2 * n, 1))
X2 = (radius * np.sin(angles)).reshape((2 * n, 1))

X = np.concatenate([X1, X2], axis=1)
y = np.concatenate([np.ones((n, 1)), -np.ones((n, 1))], axis=0).reshape([-1])


def article_a(X, y, C=10):
    clf1, title1 = linear_kernel(X, y, C=C)
    clf2, title2 = polynomial_kernel(X, y, C=C, degree=2, coef=0)
    clf3, title3 = polynomial_kernel(X, y, C=C, degree=3, coef=0)

    plot_results([clf1, clf2, clf3], [title1, title2, title3], X, y)


def article_b(X, y, C=10):
    clf1, title1 = linear_kernel(X, y, C=C)
    clf2, title2 = polynomial_kernel(X, y, C=C, degree=2, coef=1)
    clf3, title3 = polynomial_kernel(X, y, C=C, degree=3, coef=1)

    plot_results([clf1, clf2, clf3], [title1, title2, title3], X, y)


def article_c(X, y, C):
    gammas = np.power(10, np.arange(-5, 3), dtype=float)
    rbfs = np.array(
        [rbf_kernel(X, y, C=C, gamma=gamma) for gamma in gammas if gamma != 10.0]
    )
    poly_kernel = np.array(
        [
            polynomial_kernel(X, y, C=C, degree=2, coef=1),
            rbf_kernel(X, y, C=C, gamma=10.0),
        ]
    )

    plot_results(rbfs[:, 0], rbfs[:, 1], X, y)
    plot_results(poly_kernel[:, 0], poly_kernel[:, 1], X, y)


def linear_kernel(X, y, C):
    linear_svc = svm.SVC(kernel="linear", C=C)
    clf = linear_svc.fit(X, y)
    return clf, "linear kernel"


def polynomial_kernel(X, y, C, degree, coef):
    poly_svc = svm.SVC(kernel="poly", gamma="auto", coef0=coef, degree=degree, C=C)
    clf = poly_svc.fit(X, y)

    return clf, f"{'' if coef == 0 else 'non'}hom poly {degree}"


def rbf_kernel(X, y, C, gamma):
    rbf_svc = svm.SVC(kernel="rbf", gamma=gamma, C=C)
    clf = rbf_svc.fit(X, y)

    return clf, f"rbf {gamma}"


def perturb(labels):
    np.random.RandomState(0)

    return np.array(
        [np.random.choice([abs(label), label], p=[0.1, 0.9]) for label in labels]
    )


article_c(X, perturb(y), C=C)
