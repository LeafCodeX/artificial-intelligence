import numpy as np
import matplotlib.pyplot as plt
import random as rnd

from data import get_data, inspect_data, split_data

data = get_data()
inspect_data(data)

train_data, test_data = split_data(data)


# MSE (Mean Squared Error)
# Wartość przewidywana y0 (1.1 - 1.2) str. 1, końcowy MSE (1.3) str. 2
def mse_calculation(x, y, theta):
    y0 = float(theta[0]) + float(theta[1]) * x  # y= ax+b (regresja liniowa, f.k.)
    m = len(x)
    mse = np.sum((y0 - y) ** 2) / m
    return mse

# Simple Linear Regression
# predict MPG (y, dependent variable) using Weight (x, independent variable) using closed-form solution
# y = theta_0 + theta_1 * x - we want to find theta_0 and theta_1 parameters that minimize the prediction error

# We can calculate the error using MSE metric:
# MSE = SUM (from i=1 to n) (actual_output - predicted_output) ** 2

# get the columns
y_train = train_data['MPG'].to_numpy()  # wektor jednowymiarowy
x_train = train_data['Weight'].to_numpy()

y_test = test_data['MPG'].to_numpy()
x_test = test_data['Weight'].to_numpy()  # x1 x2

# TODO: calculate closed-form solution
theta_best = [0, 0]
# y=mx+c  // y=ax+b
# Dodanie kolumny jednostkowej (bias term) do macierzy cech o liczbie wierszy = x_train (1.8 - 1.10) str. 4
matrix_X = np.ones_like(x_train)
matrix_X = np.column_stack((matrix_X, x_train))
# Obliczenie rozwiązania jawnego (closed-form solution) ze wzoru (1.13) str. 4:
# np.linalg.inv(): odwrotność macierzy, matrix_X.T: transponowana macierz
theta_best = np.linalg.inv(matrix_X.T.dot(matrix_X)).dot(matrix_X.T).dot(y_train)
print('\nTheta (closed-form solution):', theta_best)

# TODO: calculate error
mse = mse_calculation(x_train, y_train, theta_best)
print('MSE Train set (closed-form solution): %.15f' % mse)
mse = mse_calculation(x_test, y_test, theta_best)
print('MSE Test set (closed-form solution): %.15f' % mse)

# plot the regression line
x = np.linspace(min(x_test), max(x_test), 100)
y = float(theta_best[0]) + float(theta_best[1]) * x
plt.plot(x, y, color='purple', label='Closed-Form Solution')
plt.scatter(x_test, y_test, label='Test Data Points')
plt.xlabel('Weight')
plt.ylabel('MPG')
plt.legend(loc='upper right', fontsize='small')
plt.title('Closed-form solution')
plt.show()

# TODO: standardization
# z = (x - mean) / std
# standaryzacja Z - (Z-score normalization)
# przekształcanie wartości zmiennych tak, aby wartość średnia populacji
# wynosiła 0 i odchylenie standardowe 1 (1.15) str. 4
# mean - średnia populacji, std - odchylenie standardowe
std_x = np.std(x_train)  # zapewnienie, że skalowane dane będą miały odchylenie standardowe wynoszące 1
mean_x = np.sum(x_train) / len(x_train)
z_train_x = ((x_train - mean_x) / std_x)
z_test_x = ((x_test - mean_x) / std_x)

std_y = np.std(y_train)
mean_y = np.sum(y_train) / len(y_train)
z_train_y = ((y_train - mean_y) / std_y)
z_test_y = ((y_test - mean_y) / std_y)


# TODO: calculate theta using Batch Gradient Descent
theta_best = [rnd.random(), rnd.random()]
# lerning rate - kontroluje, jak bardzo dostosowujemy parametry modelu w odniesieniu do gradientu kosztów
# iterations - ilość razy jaką algorytm będzie aktualizował parametry modelu
learning_rate = 0.0001
iterations_of_learning = 100000
# uwzględnienie wyrazu wolnego
matrix_X = np.ones_like(z_train_x)
matrix_X = np.column_stack((matrix_X, z_train_x))
# The Gradient Descent loop
for i in range(iterations_of_learning):
    # gradient funkcji kosztu w odniesieniu do parametrów modelu (theta)
    # matrix_X.dot(theta_best) - z_train_y: oblicza błędy predykcji, odejmując rzeczywiste wartości od prognoz
    # matrix_X.T.dot(...): Mnoży transponowaną macierz cech przez błędy predykcji, co daje nam
    # sumę wkładów gradientu ze wszystkich przykładów szkoleniowych.
    # 2/m * ...: Oblicza średni gradient, dzieląc go przez liczbę przykładów szkoleniowych (m) i mnożąc przez 2,
    # zgodnie ze wzorem gradientu funkcji kosztu.
    m = len(z_train_x)
    mse_gradient = (2 / m) * matrix_X.T.dot(matrix_X.dot(theta_best) - z_train_y) # (1.7) str. 3
    # aktualizuje parametry modelu, odejmując iloczyn współczynnika uczenia i gradientu od bieżących wartości theta.
    # przesuwa parametry w kierunku, który minimalnie zmniejsza funkcję kosztu.
    theta_best = theta_best - mse_gradient * learning_rate # gradient prosty (1.14) str. 4
    # print(mse_gradient, theta_best)
print('\nTheta (Batch Gradient Descent):', theta_best)

# TODO: calculate error
# obliczenie błędu bez odwracania stand. i Batch Gradient Descent
mse_after_gradient = mse_calculation(z_train_x, z_train_y, theta_best)
print('MSE Train set (Batch Gradient Descent): %.15f' % mse_after_gradient)
mse_after_gradient = mse_calculation(z_test_x, z_test_y, theta_best)
print('MSE Test set (Batch Gradient Descent): %.15f' % mse_after_gradient)

# plot the regression line
x = np.linspace(min(z_test_x), max(z_test_y), 100)
y = float(theta_best[0]) + float(theta_best[1]) * x
plt.plot(x, y, color='purple', label='Batch Gradient Descent')
plt.scatter(z_test_x, z_test_y, label='Test Data Points')
plt.xlabel('Weight')
plt.ylabel('MPG')
plt.legend(loc='upper right', fontsize='small')
plt.title('Batch Gradient Descent')
plt.show()
