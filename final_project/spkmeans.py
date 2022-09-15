import sys
import numpy as np 
import pandas as pd
import myspkmeans

MAX_ITER = 300
EPSILON = 0

def distance_squared(mu_j, x_l):
	diff = mu_j - x_l
	return np.dot(diff, diff)

def init_centroids(x, K, N):
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

		elif goal == "jacobi":
			eigenvalues, V = myspkmeans.jacobi(input.values.tolist())
			print(",".join(f"{num:.4f}" for num in eigenvalues))
			print_matrix(V)

		elif goal == "spk":
			weighted_adjacency_matrix = myspkmeans.calc_weighted_adjacency_matrix(input.values.tolist())
			diagonal_degree_matrix = myspkmeans.calc_diagonal_degree_matrix(weighted_adjacency_matrix)
			lnorm = myspkmeans.calc_lnorm(weighted_adjacency_matrix, diagonal_degree_matrix)
			eigenvalues, V = myspkmeans.jacobi(lnorm)
			n = len(V)
			eigenvectors = [[V[j][i] for j in range(n)] for i in range(n)] # get the eigenvectors from the columns
			eigenvalues_vectors = list(zip(eigenvalues, eigenvectors))
			eigenvalues_vectors.sort(reverse=True, key=lambda x:x[0]) # sort the eigenvalues and vectors together by the values
			if K == 0:
				# obtain K from eigengap heuristic
				deltas = [abs(eigenvalues_vectors[i][0] - eigenvalues_vectors[i+1][0]) for i in range(n//2 - 1)]
				max_delta = max(deltas)
				K = deltas.index(max_delta) + 1 # +1 because it's a 0 based index

			K_largest_vectors = [x[1] for x in eigenvalues_vectors[:K]]
			U = np.array(K_largest_vectors).T
			# normalize the rows forming T
			for i in range(n):
				U[i] = U[i] / np.sqrt(np.dot(U[i], U[i]))

			index_lst, initial_centroids = init_centroids(U, K, n)
			print(*index_lst, sep = ",")

			final_centroids = myspkmeans.fit(U.tolist(), initial_centroids.tolist(), K, MAX_ITER, EPSILON)
			for centroid in final_centroids:
				print(",".join(f"{num:.4f}" for num in centroid))


		else:
			print("Invalid Input!")
			exit()
	
	except Exception:
		print("An Error Has Occurred")


if __name__ == '__main__':
	main()