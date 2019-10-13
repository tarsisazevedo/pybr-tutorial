import numpy
from matplotlib import pyplot


def hypoteses(theta0, theta1, x):
    return theta0 + (theta1 * x)


def cost(theta0, theta1, X, y):
    # R2
    mse_model = 0
    mse_baseline = 0
    y_mean = sum(y) / len(y)
    for (xi, yi) in zip(X, y):
        y_pred = hypoteses(theta0, theta1, xi)
        mse_model += (y_pred - yi) ** 2
        mse_baseline += (y_pred - y_mean) ** 2
    r2 = 1 - (mse_model / mse_baseline)
    return r2


def plot_lines(lines_to_plot, X, y):
    max_x = numpy.max(X) + 100
    min_x = numpy.min(X) - 100

    theta0 = lines_to_plot[0][0]
    theta1 = lines_to_plot[0][1]
    xplot = numpy.linspace(min_x, max_x, 1000)
    yplot = theta0 + theta1 * xplot
    pyplot.scatter(X, y)
    pyplot.axis("tight")
    for theta0, theta1 in lines_to_plot[1:]:
        yplot = theta0 + theta1 * xplot
        pyplot.plot(xplot, yplot, color="#ff0000", label="Regression Line")
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
    # gradient descent
    dtheta0, dtheta1 = derivatives(theta0, theta1, X, y)
    theta0 = theta0 - (alpha * dtheta0)
    theta1 = theta1 - (alpha * dtheta1)

    return theta0, theta1


def fit(X, y, epochs=10000, learning_rate=0.005, plot_line=True):
    theta0 = numpy.random.rand()
    theta1 = numpy.random.rand()
    lines_to_plot = []
    for i in range(0, epochs):
        if plot_line and i % (epochs / 10) == 0:
            lines_to_plot.append((theta0, theta1))
        theta0, theta1 = update_parameters(theta0, theta1, X, y, learning_rate)
    if plot_line:
        plot_lines(lines_to_plot, X, y)
    return theta0, theta1


def generate_model(theta0, theta1):
    def model(x, theta0=theta0, theta1=theta1):
        return hypoteses(theta0, theta1, x)

    return model
