#ifndef MATRIX
#define MATRIX

/* struct representing a matrix of double values. Should be used as pointer, initialized by init_matrix
 * and freed by free_matrix.
 * members:
 * buffer - points to buffer of the actual data (inner usage only)
 * content - pointer to rows of the matrix, can be indexed as 2d array
 * rows - number of rows in the matrix
 * columns - number of columns in the matrix
 */
typedef struct
{
    double *buffer;
    double **content;
    size_t rows;
    size_t columns;
} matrix;

/*
 * Initialize matrix of the given dimensions. All values are intialized to zero.
 * Returns NULL on error.
 * It is the caller's responsibility to free it by passing it to free_matrix.
 */
matrix *init_matrix(size_t rows, size_t columns);

/*
 * Frees the given matrix and all its data.
 */
void free_matrix(matrix *mat);

/*
 * Given two vectors (double array pointers) and their dimension and calculate
 * the euclidean norm of their difference - ||v2-v1||2
 */
double euclidean_diff_norm(double *v1, double *v2, size_t d);

/*
 * Given two matrices and calculates the result of their multiplication.
 * Assumes the matrices can be multiplied - i.e. first->columns == second->rows.
 * Returns NULL on error.
 * It is the caller's responsibility to free the returned matrix by passing it to free_matrix.
 */
matrix *multiply_matrices(matrix *first, matrix *second);

/*
 * Given a file name and reads it as a csv into a matrix.
 * Assumes the file is a valid csv that represents a matrix.
 * Returns NULL on error.
 * It is the caller's responsibility to free the returned matrix by passing it to free_matrix.
 */
matrix *read_file_to_matrix(char *file_name);

/*
 * Given a matrixs and prints it to stdout as rows of comma separated values.
 */
void print_matrix(matrix *mat);

/*
 * Given a matrixs and zeroes all its values.
 */
void zero_matrix(matrix *mat);

#endif