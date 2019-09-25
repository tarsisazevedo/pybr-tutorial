from linear_regression import linear_regression

def test_hypoteses_function():
    theta0 = 1
    thtea1 = 2
    x = 4.5
    y = linear_regression.hypoteses(theta0, thtea1, x)
    assert y == 10.0
