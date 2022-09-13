#ifndef SPKMEANS
#define SPKMEANS
#include "matrix.h"

/*
 * Calculate the weighted adjacency matrix as defined in the project document 1.1.1 based on the given matrix of datapoints.
 * Returns NULL on error.
 * It is the caller's responsibility to free the returned matrix by passing it to free_matrix.
 */
matrix *calc_weighted_adjacency_impl(matrix *datapoints);

/*
 * Calculate the diagonal degree matrix as defined in the project document 1.1.2 based on the given weighted adjacency matrix.
 * Returns NULL on error.
 * It is the caller's responsibility to free the returned matrix by passing it to free_matrix.
 */
matrix *calc_diagonal_degree_impl(matrix *weighted_adjacency_matrix);


/*
 * Calculate the normalized graph laplacian (lnorm) as defined in the project document 1.1.3 based on the given weighted adjacency matrix and diagonal degree matrix.
 * The calculation is done in-place by modifying weighted_adjacency_matrix. diagonal_degree_matrix is also modified during
 * the calculation.
 */
void calc_lnorm_impl(matrix *weighted_adjacency_matrix, matrix *diagonal_degree_matrix);

/*
 * Fits K centroids to the given datapoints using the kmeans algorithm.
 * Centroids should be a matrix of K initial centroids. It will be modified and after the function runs
 * will contain the fitted centroids. epsilon represents the convergence criteria, max_iter represents
 * the maximum iterations to do if the criteria isn't met.
 * Returns 0 on success, nonzero on error.
 */
int fit_impl(matrix *datapoints, size_t K, size_t max_iter, matrix *centroids, double epsilon);

#endif
