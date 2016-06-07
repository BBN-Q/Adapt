"""
	Description of the problem:	In which order should the points (in parameter space) be measured?

	The refinement method (src/adapt/refine.py) outputs a set of new points (in parameter space) \
	whose value needs to be measured by experiment. We need to find a strategy to determine \
	the good (not necessary best) order to measure them.

	The problem can be rephrased as follow: Giving a set of points, find a (relatively) short path \
	that passes through all of them.
"""

import numpy as np

def generate_points(num, dim=2):
	""" Generate a matrix of dimension dim of random numbers.

	num: number of points in each dimension\
	dim: number of dimensions
	"""
	dims = np.ones(dim)*num
	nums = np.zeros(dims)
	for index,x in np.ndenumerate(nums):
		nums[index] = np.random.random()
	return nums

