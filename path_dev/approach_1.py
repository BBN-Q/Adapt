"""
	Approach #1 to solve the optimal path problem

	Step 1: Delaunay triangulization of the points
	Step 2: For each point, find neighbors and distances on the Delaunay mesh
	step 3: Start from a point, move to nearest neighbor
		Step 3.1: Mark visited points and remove from the set
		Step 3.2: If all neighbors are visited, find 2nd-ordered neighbors (neighbors of neighbors)
"""

import numpy as np
import numpy.linalg as lg
from scipy.spatial import Delaunay

import matplotlib.pylab as plt

class Node():
	""" Basic object represents a point on a mesh """
	def __init__(self, point, index=None):
		""" Initiate a node
		"""

		# order: visit order, if decided
		self.point = point
		self.index = index
		self.order = None # unvisited
		self.neighbors = [] # list of neighbors
		self._nearest = None # points to the nearest neighbor

	def weigh(self,nb):
		""" Calculate the weight of two adjacent points """
		return lg.norm(self.point-nb.point, ord=1)

	def weights(self, nbs):
		""" Update weights between self and neighbors """
		return [self.weigh(nb) for nb in nbs]

	def add_neighbors(self, nodes, nb_list):
		""" Add new neighbors """
		for nb_index in nb_list:
			self.neighbors.append(nodes[nb_index])

	def expand_neighbors(self):
		""" Find 2nd-ordered neighbors """
		new_nb = []
		for nb in self.neighbors:
			for nb_next in nb.neighbors:
				new_nb.append(nb_next)
		for nb in new_nb:
			self.neighbors.append(nb)

	def unvisited_neighbors(self):
		""" Find unvisited neighbors """
		unvisited = [nb for nb in self.neighbors if nb.order is None]
		return unvisited

	def nearest(self):
		""" Find nearest unvisited neighbor """
		unvisited_nb = self.unvisited_neighbors()
		if len(unvisited_nb)<1:
			self.expand_neighbors()
			unvisited_nb = self.unvisited_neighbors()

		if len(unvisited_nb)<1:
			return None
		weights_all = self.weights(unvisited_nb)
		nearest_nb = unvisited_nb[np.where(weights_all==np.amin(weights_all))[0][0]]
		self._nearest = nearest_nb
		return nearest_nb

if __name__ == "__main__":

	from path_prob import generate_points
	NUM = 100 # number of points
	points = generate_points(NUM)
	mesh = Delaunay(points)
	
	# Update points into nodes
	nodes = []
	for index, point in enumerate(points):
		node = Node(point, index=index)
		nodes.append(node)

	# Add 1st-order neighbors
	indices, indptr = mesh.vertex_neighbor_vertices
	for index, node in enumerate(nodes):
		node.add_neighbors(nodes,indptr[indices[node.index]:indices[node.index+1]])
	
	total_weight = 0
	total_visited = 0
	path = np.zeros(NUM)
	path_points = np.zeros((NUM,2))
	current_node = nodes[0]
	print(current_node.point)

	# Start travelling
	while current_node is not None:
		current_node.order = total_visited
		print(current_node.index)
		path[total_visited] = current_node.index
		path_points[total_visited] = current_node.point

		next_node = current_node.nearest()
		if next_node is not None:
			total_weight += current_node.weigh(next_node)

		current_node = next_node
		total_visited += 1

	print("Total points visited: %d" % total_visited)
	print("Total weight = %g" % total_weight)
	# Plot
	plt.plot(points[:,0], points[:,1],'o')
	plt.plot(path_points[:,0], path_points[:,1],'-')
	plt.show()






