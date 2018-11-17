import numpy as np 
import matplotlib.pyplot as plt 
plot_x = np.linspace(-1, 6, 141)
# print(plot_x)
plot_y = (plot_x - 2.5)**2-1

plt.plot(plot_x, plot_y)
plt.show()

# loss function
def J(theta):
	return (theta - 2.5)**2-1

# Loss Function's differerntial
def dJ(theta):
	return 2*(theta-2)-1

eta = 0.1 # learning rate
theta = 0.0 # initital position
epsilon = 1e-8 # precision
while True:
	last_theta = theta
	theta = theta - eta * dJ(theta)
	if(abs(J(theta)-J(last_theta)) < epsilon):
		break
print(theta)
print(J(theta))