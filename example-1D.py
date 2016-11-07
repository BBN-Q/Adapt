from scipy.spatial import Delaunay
import matplotlib.pyplot as plt
import numpy as np
import itertools

from adapt.refine import refine_scalar_field, smallest_length, average_length, refine_1D

NUM_X 	= 250
NUM_Y 	= 250
NUM_COARSE_X = 5
NUM_COARSE_Y = 5

NOISE_LEVEL = 0.1

ITERATION  = 100
MAX_POINTS = 50	# Maximum number of points to take

def ideal_tc(x, xc=9.0, k=20.0):
	return x*1.0/(1.0 + np.exp(-k*(x-xc)))

xs = np.linspace(0, 20, NUM_X)

coarse_xs = np.linspace(xs[0], xs[-1], NUM_COARSE_X)
values_orig = ideal_tc(coarse_xs) + NOISE_LEVEL*np.random.random(NUM_COARSE_X)

# Evaluate values at original mesh points
points = coarse_xs
values = values_orig

# Find new points and update values
for i in range(ITERATION):
	new_points = refine_1D(points, values, all_points=False,
								criterion="difference", threshold = "one_sigma")
	if new_points is None:
		print("No more points can be added.")
		break

	# Update points and values
	points     = np.append(points, new_points, axis=0)
	new_values = ideal_tc(new_points)
	new_values += NOISE_LEVEL*np.random.random(new_values.shape)
	values     = np.append(values, new_values, axis=0) 

	if len(points) > MAX_POINTS:
		print("Reach maximum number of points! Stop.")
		break

plt.title("Ideal vs. Refined")
plt.plot(xs, ideal_tc(xs), 'k-', lw=1.0, label='Ideal Function')
plt.plot(points, values, 'bo', label='Final points')
plt.plot(coarse_xs, values_orig, 'ro', label="Initial Coarse Points")
plt.legend()
plt.show()
