import sys
import math

DEFAULT_MAX_ITER = 200
DEFAULT_EPSILON = 0.001

# calculate ||v1-v2||2 (euclidean norm of difference between v1 and v2) with their dimension given in d
def euclidean_diff_norm(v1, v2, d):
	return math.sqrt(sum((v1[i]-v2[i])**2 for i in range(d)))

def find_k_means(x, K, d, max_iter, eps=DEFAULT_EPSILON):
	mu = [x[i].copy() for i in range(K)]

	for _ in range(max_iter):
		cluster_sums = [[0]*d for _ in range(K)]
		cluster_counts = [0] * K
		for x_i in x:
			# find the mu to which x_i has the lowest distance, assigning it to that cluster
			# explicit cluster lists are not maintained, only the cluster sums and counts to later calculate means
			min_distance = float('inf')
			min_j = -1
			for j, mu_j in enumerate(mu):
				distance = euclidean_diff_norm(x_i, mu_j, d)
				if distance < min_distance:
					min_distance = distance
					min_j = j

			for n in range(d):
				cluster_sums[min_j][n] += x_i[n]
			cluster_counts[min_j] += 1

		mu_converged = True # should represent the condition "all ||delta mu||2 < eps" after the loop
		for k in range(K):
			new_mu_k = [cluster_sums[k][n] / cluster_counts[k] for n in range(d)]
			# if the norm of the change in any mu is bigger than epsilon we didn't converge yet
			if euclidean_diff_norm(mu[k], new_mu_k, d) >= eps:
				mu_converged = False
			mu[k] = new_mu_k

		if mu_converged:
			break

	return mu



def main():
	try:
		if len(sys.argv) == 4:
			_, K_arg, input_filename, output_filename = sys.argv
			max_iter = DEFAULT_MAX_ITER
		elif len(sys.argv) == 5:
			_, K_arg, max_iter_arg, input_filename, output_filename = sys.argv
			if max_iter_arg.isdigit():
				max_iter = int(max_iter_arg)
			else:
				print("Invalid Input!")
				exit()
		else:
			print("Invalid Input!")
			exit()

		if K_arg.isdigit():
			K = int(K_arg)
		else:
			print("Invalid Input!")
			exit()

		if max_iter <= 0 or K <= 0:
			print("Invalid Input!")
			exit()

		with open(input_filename, "r") as input_file:
			x = [[float(num) for num in line.split(",")] for line in input_file]

		centroids = find_k_means(x, K, len(x[0]), max_iter)
		
		with open(output_filename, "w") as output_file:
			for centroid in centroids:
				centroid_str = ",".join(f"{num:.4f}" for num in centroid)
				output_file.write(f"{centroid_str}\n")
	except Exception:
		print("An Error Has Occurred")


if __name__ == '__main__':
	main()
