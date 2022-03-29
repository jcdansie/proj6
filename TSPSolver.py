#!/usr/bin/python3
from queue import PriorityQueue

from which_pyqt import PYQT_VER
if PYQT_VER == 'PYQT5':
	from PyQt5.QtCore import QLineF, QPointF
elif PYQT_VER == 'PYQT4':
	from PyQt4.QtCore import QLineF, QPointF
elif PYQT_VER == 'PYQT6':
	from PyQt6.QtCore import QLineF, QPointF
else:
	raise Exception('Unsupported Version of PyQt: {}'.format(PYQT_VER))




import time
import numpy as np
from TSPClasses import *
import heapq
import itertools



class TSPSolver:
	def __init__( self, gui_view ):
		self._scenario = None

	def setupWithScenario( self, scenario ):
		self._scenario = scenario


	''' <summary>
		This is the entry point for the default solver
		which just finds a valid random tour.  Note this could be used to find your
		initial BSSF.
		</summary>
		<returns>results dictionary for GUI that contains three ints: cost of solution,
		time spent to find solution, number of permutations tried during search, the
		solution found, and three null values for fields not used for this
		algorithm</returns>
	'''

	def defaultRandomTour( self, time_allowance=60.0 ):
		results = {}
		cities = self._scenario.getCities()
		ncities = len(cities)
		foundTour = False
		count = 0
		bssf = None
		start_time = time.time()
		while not foundTour and time.time()-start_time < time_allowance:
			# create a random permutation
			perm = np.random.permutation( ncities )
			route = []
			# Now build the route using the random permutation
			for i in range( ncities ):
				route.append( cities[ perm[i] ] )
			bssf = TSPSolution(route)
			count += 1
			if bssf.cost < np.inf:
				# Found a valid route
				foundTour = True
		end_time = time.time()
		results['cost'] = bssf.cost if foundTour else math.inf
		results['time'] = end_time - start_time
		results['count'] = count
		results['soln'] = bssf
		results['max'] = None
		results['total'] = None
		results['pruned'] = None
		return results


	''' <summary>
		This is the entry point for the greedy solver, which you must implement for
		the group project (but it is probably a good idea to just do it for the branch-and
		bound project as a way to get your feet wet).  Note this could be used to find your
		initial BSSF.
		</summary>
		<returns>results dictionary for GUI that contains three ints: cost of best solution,
		time spent to find best solution, total number of solutions found, the best
		solution found, and three null values for fields not used for this
		algorithm</returns>
	'''

	def greedy( self,time_allowance=60.0 ):
		pass



	''' <summary>
		This is the entry point for the branch-and-bound algorithm that you will implement
		</summary>
		<returns>results dictionary for GUI that contains three ints: cost of best solution,
		time spent to find best solution, total number solutions found during search (does
		not include the initial BSSF), the best solution found, and three more ints:
		max queue size, total number of states created, and number of pruned states.</returns>
	'''

	pruned = 0
	def branchAndBound( self, time_allowance=60.0 ):
		results = {}
		cities = self._scenario.getCities()
		num_cities = len(cities)
		count = 0
		self.pruned = 0

		# generate BSSF
		bssf = None
		foundTour = False
		while not foundTour:
			# create a random permutation
			perm = np.random.permutation(num_cities)
			route = []
			# Now build the route using the random permutation
			for i in range(num_cities):
				route.append(cities[perm[i]])
			bssf = TSPSolution(route)
			if bssf.cost < np.inf:
				# Found a valid route
				foundTour = True

		# generate initial reduced cost matrix
		initial_matrix = np.matrix(np.ones((num_cities,num_cities)) * np.inf)
		for i in range(num_cities):
			for j in range(num_cities):
				initial_matrix[i,j] = cities[i].costTo(cities[j])
		initial_state = self.State(initial_matrix, num_cities, 0, [])
		self.reduce(initial_state)

		# generate priority queue
		queue = PriorityQueue()
		queue.put(initial_state)

		start_time = time.time()
		while not queue.empty() and time.time()-start_time < time_allowance:
			# pop first, explore children
			tempState = queue.get()

			# handle complete solution
			if len(tempState.route) == tempState.size: # may be off by one, size + 1?
				cityRoute = []
				for i in range(len(tempState.route)):
					cityRoute.append(cities[tempState.route[i]])
				bssf = TSPSolution(cityRoute)
				count += 1
			elif tempState.bound > bssf.cost:
				self.pruned += 1
			else:
				self.exploreState(tempState, bssf, queue)

		end_time = time.time()
		results['cost'] = bssf.cost
		results['time'] = end_time - start_time
		results['count'] = count
		results['soln'] = bssf
		results['max'] = None
		results['total'] = None
		results['pruned'] = self.pruned
		return results

	def reduce(self, state):
		# check rows
		for i in range(state.size):
			# find row min
			rowMin = np.inf
			for j in range(state.size):
				if state.matrix[i, j] < rowMin:
					rowMin = state.matrix[i, j]
				if rowMin == 0:
					break
			#reduce row
			if rowMin != 0 and rowMin != np.inf:
				state.bound += rowMin
				for j in range(state.size):
					state.matrix[i, j] -= rowMin
		# check columns
		for j in range(state.size):
			# find column min
			columnMin = np.inf
			for i in range(state.size):
				if state.matrix[i, j] < columnMin:
					columnMin = state.matrix[i, j]
				if columnMin == 0:
					break
			# reduce column
			if columnMin != 0 and columnMin != np.inf:
				state.bound += columnMin
				for i in range(state.size):
					state.matrix[i, j] -= columnMin
		return

	def exploreState(self, state, bssf, queue):
		if len(state.route) == 0:
			row = 0
			state.route.append(0)
		else:
			row = state.route[len(state.route) - 1]

		# explore each child
		for j in range(state.size):
			edge_dist = state.matrix[row, j]
			if edge_dist == np.inf:
				continue
			new_state = self.State(state.matrix.copy(), state.size, state.bound + edge_dist, state.route.copy())
			new_state.route.append(j)
			# remove rows and columns
			for a in range(new_state.size):
				new_state.matrix[row, a] = np.inf
				new_state.matrix[a, j] = np.inf
			# reduce
			self.reduce(new_state)
			# add to queue or prune
			if new_state.bound < bssf.cost:
				queue.put(new_state)
			else:
				self.pruned += 1
		return

	class State():
		def __init__(self, matrix, size, bound, route):
			self.matrix = matrix
			self.size = size
			self.bound = bound
			self.route = route

		def __lt__(self, other):
			return self.bound < other.bound

		def __eq__(self, other):
			return self.bound == other.bound

		def __repr__(self):
			return "State bound:% s" % (self.bound)


	''' <summary>
		This is the entry point for the algorithm you'll write for your group project.
		</summary>
		<returns>results dictionary for GUI that contains three ints: cost of best solution,
		time spent to find best solution, total number of solutions found during search, the
		best solution found.  You may use the other three field however you like.
		algorithm</returns>
	'''

	def fancy( self,time_allowance=60.0 ):
		pass
