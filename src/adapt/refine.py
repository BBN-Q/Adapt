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

def filter_threshold(delta_f, threshold_value):
	""" Return indices of  the data whose values are above the acceptable level """
	return delta_f >= threshold_value

def filter_grand(delta_rs, delta_fs, threshold = "one_sigma", criterion = "integral",
				resolution = 0, noise_level = 0):
	""" Filter points base on a combination of filters """

	# Filter resolution
	filter_res = filter_threshold(delta_rs, resolution)
	# Filter noise
	filter_noise = filter_threshold(delta_fs, noise_level)

	if criterion == "integral":
		metric = delta_fs*delta_rs
	elif criterion == "integralsq":
		metric = delta_fs*delta_rs*delta_rs
	elif criterion == "difference":
		metric = delta_fs
	elif criterion == "spikes":
		integrals = delta_fs*delta_rs
		spread = np.std(integrals, axis=1)
		metric = np.outer(spread, np.ones(len(integrals[0])))
	else:
		raise ValueError("Invalid criterion specified. Must be one of 'integral', 'integralsq', 'difference', 'spikes'.")

	# Filter criterion
	if threshold == "mean":
		filter_thres = filter_threshold(metric, np.mean(metric))
	elif threshold == "half":
		filter_thres = filter_threshold(metric, 0.5*metric.max())
	elif threshold == "one_sigma":
		filter_thres = filter_threshold(metric, np.mean(metric)+np.std(metric))
	elif threshold == "two_sigma":
		filter_thres = filter_threshold(metric, np.mean(metric)+2*np.std(metric))
	elif threshold == "non_zero":
		filter_thres = abs(metric) > 1e-10
	else:
		raise ValueError("Invalid threshold specified. Must be one of 'mean', 'half', 'one_sigma', 'two_sigma', 'non_zero")

	return filter_thres*filter_noise*filter_res

def well_scaled_delaunay_mesh(points):
	scales  = []
	offsets = []
	points  = np.array(points)
	for i in range(points.shape[1]):
		offsets.append(points[:,i].min())
		scales.append(1.0/(points[:,i].max()-points[:,i].min()))
		points[:,i] -= offsets[-1]
		points[:,i] *= scales[-1]

	mesh = Delaunay(points)
	return mesh, scales, offsets

def refine_1D(points, values, all_points=False,
			  criterion="integral", threshold="one_sigma",
			  resolution=0, noise_level=0):

	# Do not assume our data is sorted
	sort_inds = np.argsort(points)
	rs_sorted = points[sort_inds]
	fs_sorted = values[sort_inds]
	delta_rs  = np.diff(rs_sorted)
	delta_fs  = np.diff(fs_sorted)

	do_refine = filter_grand(delta_rs, delta_fs, threshold = threshold, criterion = criterion,
							resolution=resolution, noise_level=noise_level)

	new_points = rs_sorted[:-1] + 0.5*delta_rs
	new_points = new_points[np.where(do_refine)]

	if len(new_points) > 0:
		# print("{} new points added.".format(len(new_points)))
		if all_points:
			return np.append(points, new_points)
		return new_points
	else:
		return None

def refine_scalar_field(points, values, all_points=False,
						criterion="integral", threshold="one_sigma",
						resolution=0, noise_level=0):

	mesh, scales, offsets = well_scaled_delaunay_mesh(points)

	values = np.array(values)
	new_points = []
	delta_fs = np.zeros((len(mesh.simplices), len(mesh.simplices[0])))
	delta_rs = np.zeros((len(mesh.simplices), len(mesh.simplices[0])))

	for i, simplex in enumerate(mesh.simplices):
		delta_f, delta_r  = dir_derivs(simplex, mesh, values)
		delta_fs[i] = np.abs(delta_f)
		delta_rs[i] = np.abs(delta_r)

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
		# print("{} new points added.".format(len(unique_a)))
		if all_points:
			return np.append(points, unique_a, axis=0)
		for i, (sf, of) in enumerate(zip(scales,offsets)):
			unique_a[:,i] = unique_a[:,i]/sf + of
		return unique_a
	else:
		return None
