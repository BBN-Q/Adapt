from scipy.spatial import Delaunay
import numpy as np
from numpy import linalg as la
import cProfile

def cdiff(array):
	return np.roll(array, -1, axis=0) - array 

def dir_derivs(simplex, mesh, values):
	s_points  = mesh.points[simplex] # Points on the simplex
	s_values  = values[simplex] # Values at those points
	delta_r   = la.norm(cdiff(s_points), axis=1) # Norms of vectors between points
	delta_f   = cdiff(s_values) # Difference in the values between points
	return delta_f, delta_r

def smallest_length(points):
	mesh = Delaunay(points)
	delta_rs = np.zeros((len(mesh.simplices), len(mesh.simplices[0])))
	for i, simplex in enumerate(mesh.simplices):
		delta_rs[i] = la.norm(cdiff(mesh.points[simplex]), axis=1)
	return np.min(delta_rs)

def average_length(points):
	mesh = Delaunay(points)
	delta_rs = np.zeros((len(mesh.simplices), len(mesh.simplices[0])))
	for i, simplex in enumerate(mesh.simplices):
		delta_rs[i] = la.norm(cdiff(mesh.points[simplex]), axis=1)
	return np.mean(delta_rs)

def refine_scalar_field(points, values):
	# dimensions = len(data[0])-1
	# points = data[:,0:dimensions]
	# values = data[:,-1]

	mesh = Delaunay(points)

	new_points = []
	delta_fs = np.zeros((len(mesh.simplices), len(mesh.simplices[0])))
	delta_rs = np.zeros((len(mesh.simplices), len(mesh.simplices[0])))

	for i, simplex in enumerate(mesh.simplices):
		delta_f, delta_r  = dir_derivs(simplex, mesh, values)
		delta_fs[i] = np.abs(delta_f)
		delta_rs[i] = delta_r

	delta_fs = delta_fs*delta_rs
	deltas   = delta_fs/delta_fs.max()
	deltas   = deltas > 0.5

	for i, (d, simp) in enumerate(zip(deltas, mesh.simplices)):
		half_delta_r = 0.5*cdiff(mesh.points[simp])
		for d_i, simp_i, halfr_i in zip(d, simp, half_delta_r):
			if d_i:
				new_points.append(mesh.points[simp_i] + halfr_i)

	if len(new_points) > 0:
		a = np.array(new_points)
		b = np.ascontiguousarray(a).view(np.dtype((np.void, a.dtype.itemsize * a.shape[1])))
		_, idx = np.unique(b, return_index=True)
		unique_a = a[idx]
		new_points = np.append(unique_a, mesh.points, axis=0)
		print("{} new points added.".format(len(unique_a)))
		return new_points
	else:
		raise Exception("Couldn't refine mesh.")