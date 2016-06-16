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

def reach_average_length(points, avg_len):
	""" Check if the average length of points is below the threshold """
	return average_length(points) < avg_len

def filter_threshold(delta_f, threshold_value):
	""" Return indices of  the data whose values are above the acceptable level """
	return delta_f > threshold_value

def filter_grand(delta_rs, delta_fs, threshold = "one_sigma", criterion = "difference",
				resolution = 0, noise_level = 0):
	""" Filter points base on a combination of filters """

	# Filter resolution
	filter_res = filter_threshold(delta_rs, resolution)
	# Filter noise
	filter_noise = filter_threshold(delta_fs, noise_level)

	if criterion == "integral":
		credit = delta_fs*delta_rs
	elif criterion == "difference":
		credit = delta_fs
	else:
		raise ValueError("Invalid criterion specified. Must be one of 'integral', 'difference'.")

	# Filter criterion
	if threshold == "mean":
		filter_thres = filter_threshold(credit, np.mean(credit))
	elif threshold == "half":
		filter_thres = filter_threshold(credit, 0.5*credit.max())
	elif threshold == "one_sigma":
		filter_thres = filter_threshold(credit, np.mean(credit)+np.std(credit))
	elif threshold == "two_sigma":
		filter_thres = filter_threshold(credit, np.mean(credit)+2*np.std(credit))
	else:
		raise ValueError("Invalid threshold specified. Must be one of 'mean', 'half', 'one_sigma', 'two_sigma'.")

	return filter_thres*filter_noise*filter_res

def refine_scalar_field(points, values, all_points=False,
						criterion="difference", threshold="one_sigma",
						resolution=0, noise_level=0):

	scale_factors = []
	points = np.array(points)
	for i in range(points.shape[1]):
		scale_factors.append(1.0/np.mean(points[:,i]))
		points[:,i] = points[:,i]*scale_factors[-1]

	mesh = Delaunay(points)

	new_points = []
	delta_fs = np.zeros((len(mesh.simplices), len(mesh.simplices[0])))
	delta_rs = np.zeros((len(mesh.simplices), len(mesh.simplices[0])))

	for i, simplex in enumerate(mesh.simplices):
		delta_f, delta_r  = dir_derivs(simplex, mesh, values)
		delta_fs[i] = np.abs(delta_f)
		delta_rs[i] = delta_r

	deltas = filter_grand(delta_rs, delta_fs, threshold = threshold, criterion = criterion,
							resolution=resolution, noise_level=noise_level)


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
		for i, sf in enumerate(scale_factors):
			unique_a[:,i] = unique_a[:,i]/sf
		return unique_a
	else:
		return None