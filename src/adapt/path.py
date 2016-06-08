"""
	Find a (relatively) good path passing all points

	1. Sort in one or many axes
	2. Option:
		direction:
			+1: ascending (default)
			 0: ascending or descending
			-1: descending
	3. Future development:
		3.1. Clustering the mesh
		3.2. When direction==0 (user does not care),
			the algorithm determines the best direction
"""

import numpy as np
from operator import itemgetter

def sorted_path(grid, axes=0, direction=1, func=None):
	""" Return a path from a grid with sorted axes.

	grid: a set of points \
	axes: the order of axes to be sorted \
		e.g. If we want ascending x, then ascending y, then axes = [0,1] \
		note that the actual sorting actions will be in reversed order, \
		i.e. we sort y first, then sort x.\
	direction: an integer or list of integer taking values in \
		+1: ascending (default) \
		 0: ascending or descending \
		-1: descending \
		If an integer: apply to all axes \
		If a list of integer: length should match that of axes. \
	func: user-defined, evaluate the weight of a point used for sorting
		If not None, axes and direction are unnecessary
	"""


	if func is not None:
		return sorted(grid, key=func)

	# Now that func is None
	# Check the type of axes and direction
	if isinstance(axes,int) or isinstance(axes,float):
		axes = [axes]

	NUM_AXES = len(axes)
	if isinstance(direction,int) or isinstance(direction,float):
		direction = np.ones(NUM_AXES)*direction
	if len(direction)!=len(axes):
		raise ValueError("Values of parameters do not match.")

	# Check values of axes and direction
	axes = [int(axis) for axis in axes]
	direction = [0 if drct==0 else int(drct/abs(drct)) for drct in direction]
	# Reverse the axes and direction
	axes.reverse()
	direction.reverse()

	# Start to sort
	sorted_grid = grid
	for i,axis in enumerate(axes):
		sorted_grid = sorted(sorted_grid, key=itemgetter(axis), reverse=direction[i]<-0.5)

	return np.array(sorted_grid)
