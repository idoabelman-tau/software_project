import sys
import numpy as np 
import pandas as pd
import mykmeanssp

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

def main():
	try:
		if len(sys.argv) == 5:
			_, K_arg, eps_arg, input_filename1, input_filename2 = sys.argv
			max_iter = DEFAULT_MAX_ITER
		elif len(sys.argv) == 6:
			_, K_arg, max_iter_arg, eps_arg, input_filename1, input_filename2 = sys.argv
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

		try:
			eps = float(eps_arg)
		except ValueError:
			print("Invalid Input!")
			exit()

		if max_iter <= 0 or K <= 0:
			print("Invalid Input!")
			exit()

		input1 = pd.read_csv(input_filename1, header=None)
		input2 = pd.read_csv(input_filename2, header=None)
		points = pd.merge(input1, input2, on=0)
		points.set_index(points[0], inplace=True)	
		N, d = points.shape
		d -= 1
		points = points.sort_index(axis = 0)
		x = points.to_numpy()[:,1:]

		index_lst, initial_centroids = init_centroids(x, K, d, N)
		print(*index_lst, sep = ",")

		final_centroids = mykmeanssp.fit(x.tolist(), initial_centroids.tolist(), K, max_iter, eps)
		for centroid in final_centroids:
			print(",".join(f"{num:.4f}" for num in centroid))
	
	except Exception:
		print("An Error Has Occurred")


if __name__ == '__main__':
	main()