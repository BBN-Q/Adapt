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

def pv_from_dat(data):
	dimensions = len(data[0])-1
	points = data[:,0:dimensions]
	values = data[:,dimensions:]
	return points, values

def refine_scalar_field(points, values, all_points=False,
						criterion="integral", threshold="one_sigma"):
	mesh = Delaunay(points)

	new_points = []
	delta_fs = np.zeros((len(mesh.simplices), len(mesh.simplices[0])))
	delta_rs = np.zeros((len(mesh.simplices), len(mesh.simplices[0])))

	for i, simplex in enumerate(mesh.simplices):
		delta_f, delta_r  = dir_derivs(simplex, mesh, values)
		delta_fs[i] = np.abs(delta_f)
		delta_rs[i] = delta_r

	if criterion == "integral":
		delta_fs = delta_fs*delta_rs
	elif criterion == "difference":
		delta_fs = delta_fs
	else:
		raise ValueError("Invalid criterion specified. Must be one of integral, difference.")
	
	# Scale the errors
	deltas = delta_fs/delta_fs.max()

	if threshold == "mean":
		deltas = deltas > np.mean(deltas)
	elif threshold == "half":
		deltas = deltas > 0.5
	elif threshold == "one_sigma":
		deltas = deltas > (np.mean(deltas) + np.std(deltas))
	elif threshold == "two_sigma":
		deltas = deltas > (np.mean(deltas) + 2*np.std(deltas))
	else:
		raise ValueError("Invalid threshold specified. Must be one of mean, half.")


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
		print("{} new points added.".format(len(unique_a)))
		if all_points:
			return np.append(points, unique_a, axis=0)
		return unique_a
	else:
		raise Exception("Couldn't refine mesh.")