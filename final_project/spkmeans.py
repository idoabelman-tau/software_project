import sys
import numpy as np 
import pandas as pd
import myspkmeans

DEFAULT_MAX_ITER = 300

def distance_squared(mu_j, x_l):
	diff = mu_j - x_l
	return np.dot(diff, diff)

def init_centroids(x, K, d, N):
	np.random.seed(0)
	i = 1
	a = np.random.choice(N)
	centroids = x[a][np.newaxis]
	distances = np.full(N, np.inf)
	index_lst = [a]
	P = np.full(N, np.inf)
	for i in range(1,K):
		for l in range(N):
			for c in range(i):
				distances[l] = min(distances[l], distance_squared(centroids[c], x[l]))
		for l in range(N):
			P[l] = distances[l] / sum(distances)
		ind = np.random.choice(N, p=P)
		index_lst.append(ind)
		centroids = np.append(centroids, [x[ind]], axis=0)

	return index_lst, centroids

def print_matrix(mat):
	for row in mat:
		print(",".join(f"{num:.4f}" for num in row))

def main():
	try:
		if len(sys.argv) != 4:
			print("Invalid Input!")
			exit()

		_, K_arg, goal, input_filename = sys.argv

		if K_arg.isdigit():
			K = int(K_arg)
		else:
			print("Invalid Input!")
			exit()

		if K < 0:
			print("Invalid Input!")
			exit()

		input = pd.read_csv(input_filename, header=None)

		if goal == "wam":
			weighted_adjacency_matrix = myspkmeans.calc_weighted_adjacency_matrix(input.values.tolist())
			print_matrix(weighted_adjacency_matrix)
		
		elif goal == "ddg":
			weighted_adjacency_matrix = myspkmeans.calc_weighted_adjacency_matrix(input.values.tolist())
			diagonal_degree_matrix = myspkmeans.calc_diagonal_degree_matrix(weighted_adjacency_matrix)
			print_matrix(diagonal_degree_matrix)
			
		elif goal == "lnorm":
			weighted_adjacency_matrix = myspkmeans.calc_weighted_adjacency_matrix(input.values.tolist())
			diagonal_degree_matrix = myspkmeans.calc_diagonal_degree_matrix(weighted_adjacency_matrix)
			lnorm = myspkmeans.calc_lnorm(weighted_adjacency_matrix, diagonal_degree_matrix)
			print_matrix(lnorm)

		else:
			print("Invalid Input!")
			exit()
	
	except Exception:
		print("An Error Has Occurred")


if __name__ == '__main__':
	main()