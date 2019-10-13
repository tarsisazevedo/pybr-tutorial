from linear_regression import linear_regression


def test_hypoteses_function():
    theta0 = 1
    thtea1 = 2
    x = 4.5
    y = linear_regression.hypoteses(theta0, thtea1, x)
    assert y == 10.0


def test_cost_function():
    X = [1, 2, 3, 4]
    y = [5, 6, 7, 8]
    theta0 = 1
    theta1 = 2
    assert linear_regression.cost(theta0, theta1, X, y) == 0.7142857142857143


def test_derivative():
    X = [1, 2, 3, 4]
    y = [4, 3, 2, 1]
    theta0 = 1
    theta1 = 2

    dt0, dt1 = linear_regression.derivatives(theta0, theta1, X, y)
    assert dt0 == 3.5
    assert dt1 == 12.5


def test_update_parameters():
    X = [1, 2, 3, 4]
    y = [4, 3, 2, 1]
    theta0 = 1
    theta1 = 2
    alpha = 0.05

    t0, t1 = linear_regression.update_parameters(theta0, theta1, X, y, alpha)
    assert t0 == 0.825
    assert t1 == 1.375


def test_fit():
    X = [1, 2, 3, 4]
    y = [5, 6, 7, 8]
    theta0, theta1 = linear_regression.fit(X, y, plot_line=False)
    assert theta0 > 0
    assert theta1 > 0
