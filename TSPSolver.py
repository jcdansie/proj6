#!/usr/bin/python3
import copy
from queue import PriorityQueue

from which_pyqt import PYQT_VER
if PYQT_VER == 'PYQT5':
	from PyQt5.QtCore import QLineF, QPointF
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
		solutionFound = False
		numSolutions = 1
		startIndex = 0
		cities = self._scenario.getCities()
		num_cities = len(cities)
		while not solutionFound and startIndex < len(cities):
			results = {}
			route = [cities[startIndex]]
			visited = np.ones(num_cities)
			visited[startIndex] = 0
			start_time = time.time()
			for i in range(1, num_cities):
				minCity = None
				minCityCost = np.inf
				lastCity = route[len(route) - 1]
				markAsVisited = -1
				for j in range(num_cities):
					if visited[j] == 0:
						continue
					tempCity = cities[j]
					if lastCity.costTo(tempCity) < minCityCost:
						minCity = tempCity
						minCityCost = lastCity.costTo(tempCity)
						markAsVisited = j
				if minCity is not None:
					route.append(minCity)
				if markAsVisited >= 0:
					visited[markAsVisited] = 0
			if len(route) == num_cities:
				solution = TSPSolution(route)
				if not np.isinf(solution.cost):
					solutionFound = True
				else:
					numSolutions += 1
					startIndex += 1
			else:
				numSolutions += 1
				startIndex += 1

		end_time = time.time()
		results['cost'] = solution.cost
		results['time'] = end_time - start_time
		results['count'] = numSolutions
		results['soln'] = solution
		results['max'] = None
		results['total'] = None
		results['pruned'] = None
		return results

	# Branch and Bound algorithm
	pruned = 0
	states = 0
	queueMax = 0
	def branchAndBound( self, time_allowance=60.0 ):
		results = {}
		cities = self._scenario.getCities()
		num_cities = len(cities)
		count = 0
		self.pruned = 0
		self.states = 0
		self.queueMax = 0

		# generate initial BSSF
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

		# generate initial reduced cost matrix and lower bound
		initial_matrix = np.matrix(np.ones((num_cities,num_cities)) * np.inf)
		for i in range(num_cities):
			for j in range(num_cities):
				initial_matrix[i,j] = cities[i].costTo(cities[j])
		initial_state = self.State(initial_matrix, num_cities, 0, [])
		self.states += 1
		self.reduce(initial_state)

		# generate priority queue
		queue = PriorityQueue()
		queue.put(initial_state)
		self.queueMax += 1

		# begin branch and bound
		start_time = time.time()
		while not queue.empty() and time.time()-start_time < time_allowance:
			# pop first, explore children
			tempState = queue.get()
			# prune
			if(tempState.bound) > bssf.cost:
				self.pruned += 1
			# handle complete solution
			elif len(tempState.route) == tempState.size:
				cityRoute = []
				for i in range(len(tempState.route)):
					cityRoute.append(cities[tempState.route[i]])
				tempSolution = TSPSolution(cityRoute)
				if not np.isinf(tempSolution.cost):
					bssf = TSPSolution(cityRoute)

				count += 1
			# explore
			else:
				self.exploreState(tempState, bssf, queue)
		# add states not dequeued to number pruned
		for i in range(queue.qsize()):
			tempState = queue.get()
			if tempState.bound > bssf.cost:
				self.pruned += 1

		# return values
		end_time = time.time()
		results['cost'] = bssf.cost
		results['time'] = end_time - start_time
		results['count'] = count
		results['soln'] = bssf
		results['max'] = self.queueMax
		results['total'] = self.states
		results['pruned'] = self.pruned
		return results

	# function for reducing a matrix
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

	# function for exploring a state
	def exploreState(self, state, bssf, queue):
		# if first iteration start at first city
		if len(state.route) == 0:
			row = 0
			state.route.append(0)
		# start at last city in current state's route
		else:
			row = state.route[len(state.route) - 1]

		# explore each child
		for j in range(state.size):
			edge_dist = state.matrix[row, j]
			if edge_dist == np.inf:
				#self.pruned += 1
				continue
			self.states += 1
			# create child state by copying current state
			new_state = self.State(state.matrix.copy(), state.size, state.bound + edge_dist, state.route.copy())
			new_state.route.append(j)
			# remove rows and columns
			for a in range(new_state.size):
				new_state.matrix[row, a] = np.inf
				new_state.matrix[a, j] = np.inf
			# reduce
			self.reduce(new_state)
			# add to queue
			if new_state.bound < bssf.cost:
				queue.put(new_state)
				# increase queue max size
				if queue.qsize() > self.queueMax:
					self.queueMax = queue.qsize()
			# prune
			else:
				self.pruned += 1
		return

	# class for a state
	# contains a matrix, bound, size of the matrix, and the current route
	class State():
		def __init__(self, matrix, size, bound, route):
			self.matrix = matrix
			self.size = size
			self.bound = bound
			self.route = route

		# for comparing two states (used in priority queue)
		# includes length of route to encourage digging deeper
		def __lt__(self, other):
			return self.bound/len(self.route) < other.bound/len(other.route)

		def __eq__(self, other):
			return self.bound/len(self.route) == other.bound/len(other.route)

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
	# 2-opt algorithm
	def fancy( self,time_allowance=60.0 ):
		results = {}
		cities = self._scenario.getCities()
		ncities = len(cities)
		improved = True
		count = 0

		bssf = self.greedy()['soln']		# initial bssf
		start_time = time.time()

		# Repeat these steps until there is no update with the cost
		while improved and time.time()-start_time < time_allowance:
			improved = False

			for i in range(ncities - 1):
				for j in range(i + 2, ncities):
					new_route = copy.copy(bssf.route)
					new_route[i + 1], new_route[j] = new_route[j], new_route[i + 1]		# swap the destination ([path[i], path[i+1]] => [path[i], path[j]])
					new_bssf = TSPSolution(new_route)
					if bssf.cost > new_bssf.cost:	# update the bssf if the new bssf has a smaller cost
						bssf = new_bssf
						count += 1
						improved = True

		end_time = time.time()
		results['cost'] = bssf.cost
		results['time'] = end_time - start_time
		results['count'] = count
		results['soln'] = bssf
		results['max'] = None
		results['total'] = None
		results['pruned'] = None
		return results


	# 3-opt algorithm
	def opt3( self,time_allowance=60.0 ):
		results = {}
		cities = self._scenario.getCities()
		ncities = len(cities)
		improved = True
		count = 0

		bssf = self.greedy()['soln']		# initial bssf
		start_time = time.time()

		# Repeat these steps until there is no update with the cost
		while improved and time.time()-start_time < time_allowance:
			improved = False

			for i in range(ncities - 2):
				for j in range(i + 2, ncities - 1):
					for k in range(j + 1, ncities):

						new_route = copy.copy(bssf.route)
						new_route[i + 1], new_route[j], new_route[k] = new_route[i + 1], new_route[k], new_route[j]	# swap the destination ([path[i], path[i+1]] => [path[i], path[j]])
						new_bssf = TSPSolution(new_route)
						if bssf.cost > new_bssf.cost:	# update the bssf if the new bssf has a smaller cost
							bssf = new_bssf
							count += 1
							improved = True

						new_route = copy.copy(bssf.route)
						new_route[i + 1], new_route[j], new_route[k] = new_route[j], new_route[i + 1], new_route[k]		# swap the destination ([path[i], path[i+1]] => [path[i], path[j]])
						new_bssf = TSPSolution(new_route)
						if bssf.cost > new_bssf.cost:	# update the bssf if the new bssf has a smaller cost
							bssf = new_bssf
							count += 1
							improved = True

						new_route = copy.copy(bssf.route)
						new_route[i + 1], new_route[j], new_route[k] = new_route[j], new_route[k], new_route[i + 1]		# swap the destination ([path[i], path[i+1]] => [path[i], path[j]])
						new_bssf = TSPSolution(new_route)
						if bssf.cost > new_bssf.cost:	# update the bssf if the new bssf has a smaller cost
							bssf = new_bssf
							count += 1
							improved = True


						new_route = copy.copy(bssf.route)
						new_route[i + 1], new_route[j], new_route[k] = new_route[k], new_route[i + 1], new_route[j]	# swap the destination ([path[i], path[i+1]] => [path[i], path[j]])
						new_bssf = TSPSolution(new_route)
						if bssf.cost > new_bssf.cost:	# update the bssf if the new bssf has a smaller cost
							bssf = new_bssf
							count += 1
							improved = True

						new_route = copy.copy(bssf.route)
						new_route[i + 1], new_route[j], new_route[k] = new_route[k], new_route[j], new_route[i + 1]	# swap the destination ([path[i], path[i+1]] => [path[i], path[j]])
						new_bssf = TSPSolution(new_route)
						if bssf.cost > new_bssf.cost:	# update the bssf if the new bssf has a smaller cost
							bssf = new_bssf
							count += 1
							improved = True

		end_time = time.time()
		results['cost'] = bssf.cost
		results['time'] = end_time - start_time
		results['count'] = count
		results['soln'] = bssf
		results['max'] = None
		results['total'] = None
		results['pruned'] = None
		return results


