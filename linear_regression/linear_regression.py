import math

import numpy
from matplotlib import pyplot


def hypoteses(theta0, theta1, x):
    return theta0 + (theta1 * x)


def cost(theta0, theta1, X, y):
    # RMSE
    cost_value = 0
    for (xi, yi) in zip(X, y):
        cost_value += 0.5 * ((hypoteses(theta0, theta1, xi) - yi) ** 2)
    return math.sqrt(cost_value)


def plot_line(theta0, theta1, X, y):
    max_x = numpy.max(X) + 100
    min_x = numpy.min(X) - 100

    xplot = numpy.linspace(min_x, max_x, 1000)
    yplot = theta0 + theta1 * xplot
    pyplot.plot(xplot, yplot, color="#ff0000", label="Regression Line")
    pyplot.scatter(X, y)
    pyplot.axis([-10, 10, 0, 200])
    pyplot.show()


def derivatives(theta0, theta1, X, y):
    dtheta0 = 0
    dtheta1 = 0

    for xi, yi in zip(X, y):
        dtheta0 += hypoteses(theta0, theta1, xi) - yi
        dtheta1 += (hypoteses(theta0, theta1, xi) - yi) * xi

    dtheta0 /= len(X)
    dtheta1 /= len(X)

    return dtheta0, dtheta1


def update_parameters(theta0, theta1, X, y, alpha=0.005):
    dtheta0, dtheta1 = derivatives(theta0, theta1, X, y)
    theta0 = theta0 - (alpha * dtheta0)
    theta1 = theta1 - (alpha * dtheta1)

    return theta0, theta1


def fit(X, y):
    theta0 = numpy.random.rand()
    theta1 = numpy.random.rand()
    for i in range(0, 1000):
        theta0, theta1 = update_parameters(theta0, theta1, X, y, 0.005)
    return theta0[0], theta1[0]


def generate_model(theta0, theta1):
    def model(x, theta0=theta0, theta1=theta1):
        return hypoteses(theta0, theta1, x)

    return model
