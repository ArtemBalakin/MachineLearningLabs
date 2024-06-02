import numpy as np
import matplotlib.pyplot as plt
import os

# Создаем папку для результатов, если она еще не существует
if not os.path.exists('result2'):
    os.makedirs('result2')

# Параметры системы
theta = 11
lambda_param = 6
g = lambda t: 10 * np.sin(3 * t + 1)
gamma = 0.25  # Коэффициент адаптации

# Начальные условия
x0 = 1
theta0 = 1

# Временной интервал
t = np.linspace(0, 10, 1000)

# Эталонная модель и возмущение
xm = lambda t: g(t) / lambda_param
delta = lambda t: (1+t)**(-1/8) * (1 - (1+t)**(-1/4)) - 3/8 * (1+t)**(-5/4)  # Возмущение из вашего документа

# Инициализация системы с возмущениями
theta_hat = np.zeros_like(t)
theta_hat[0] = theta0
x = np.zeros_like(t)
x[0] = x0
u = np.zeros_like(t)

# Адаптивная система управления
for i in range(1, len(t)):
    dt = t[i] - t[i - 1]
    disturbance = delta(t[i])
    u[i] = gamma * x[i - 1]**2 - lambda_param * x[i - 1] + lambda_param * g(t[i])
    x[i] = x[i - 1] + dt * (theta * x[i - 1] + u[i] + disturbance)
    error = xm(t[i]) - x[i - 1]
    theta_hat[i] = theta_hat[i - 1] - gamma * x[i - 1] * error * dt

# Построение графиков
plt.figure()
plt.plot(t, x, label='x(t) с возмущениями')
plt.plot(t, xm(t), label='xm(t)')
plt.xlabel('Время')
plt.ylabel('Состояние')
plt.legend()
plt.title('Адаптивная система управления с возмущением')
plt.grid(True)
plt.savefig('result2/adaptive_system_with_disturbance.png')
plt.show()

plt.figure()
plt.plot(t, u, label='u(t) с возмущениями')
plt.xlabel('Время')
plt.ylabel('Управляющее воздействие')
plt.legend()
plt.title('Управляющее воздействие адаптивной системы с возмущением')
plt.grid(True)
plt.savefig('result2/control_with_disturbance.png')
plt.show()
