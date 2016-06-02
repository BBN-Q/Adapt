from scipy.spatial import Delaunay
import matplotlib.pyplot as plt
import numpy as np
import itertools

def lg(x, xc, k=50.0):
	return 1.0/(1.0 + np.exp(-k*(x-xc)))
def f(x, y, x0=0.8, y0=0.09, k=50.0):
	xc = x0 / (y/y0 - 1) 
	return lg(x, xc, k=k)
def ff(v):
	return f(*v, x0=0.8, y0=0.09, k=50) - f(*v, x0=3, y0=0.09, k=25.0) + np.random.random()*0.1

xs = np.linspace(0, 1, 250)
ys = np.linspace(0.1, 1, 250)
xx, yy = np.meshgrid(xs, ys)

extent = (xs[0], xs[-1], ys[0], ys[-1])
aspect = (xs[-1]-xs[0])/(ys[-1]-ys[0])
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16,8), sharey=True, sharex=True)
#ax1.imshow(f(xx, yy, x0=0.8, y0=0.09, k=50) - f(xx, yy, x0=3, y0=0.09, k=25.0), origin='lower', extent=extent, aspect=aspect)
ax1.imshow(ff((xx, yy)), origin='lower', extent=extent, aspect=aspect)

coarse_xs = list(np.linspace(xs[0], xs[-1], 8))
coarse_ys = list(np.linspace(ys[0], ys[-1], 8))
points    = [coarse_xs, coarse_ys]
points    = list(itertools.product(*points))

from adapt.refine import refine_scalar_field, smallest_length, average_length

for i in range(5):
	values = np.apply_along_axis(ff, 1, points)
	points = refine_scalar_field(points, values, all_points=True)

print("Ended up with {} points in total.".format(len(points)))
smallest = smallest_length(points)
average = average_length(points)
print("Smallest element edge length: {}".format(smallest))
print("Average element edge length: {}".format(average))
print("Approximate savings with respect to square grid at smallest feature size: {}.".format(len(points)/((1.0/smallest)**2)))
print("Approximate savings with respect to square grid at average feature size: {}.".format(len(points)/((1.0/average)**2)))

mesh = Delaunay(points)
#ax1.triplot(mesh.points[:,0], mesh.points[:,1], mesh.simplices.copy(), 'w-')
values = np.apply_along_axis(ff, 1, mesh.points) 
ax2.tripcolor(mesh.points[:,0], mesh.points[:,1], mesh.simplices.copy(), values, shading='gouraud')
plt.show()
