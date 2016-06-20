"""
	Snap a point into a grid
"""

import numpy as np
import numpy.linalg as lg

def snap_res(value, res=0):
	""" Snap a point based on resolution

	Return to a snapped value closest to value and is multiple of res
	"""

	# Check if user mistakenly gives res=0
	if res==0:
		return value # do nothing

	nvalue = int(value/res) * res

	# Cut off extra decimals
	base10 = int(np.floor(np.log10(res)))
	return round(nvalue, -base10)

def snap_list(value, accepts, order=2):
	""" Find closest one to value in the list accepts

	ord: order of norm
	"""
	distances = np.zeros(len(accepts))
	is_vector = isinstance(value,int) or isinstance(value,float)
	for i, item in enumerate(accepts):
		if is_vector:
			distances = [abs(value-item) for item in accepts]
		else:
			distances = [lg.norm(value-item, ord=order) for item in accepts]
	min_index = np.argmin(distances)
	return np.amin(accepts[min_index])