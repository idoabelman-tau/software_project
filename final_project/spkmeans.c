#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "matrix.h"

#ifndef INFINITY
#define INFINITY (1.0/0.0)
#endif
#define EPSILON 0.00001

/* a lean representation of a rotation matrix used for jacobi */
typedef struct {
    size_t i;
    size_t j;
    double c;
    double s;
} rot_matrix;

/* doc in header */
matrix *calc_weighted_adjacency_impl(matrix *datapoints){
    size_t i;
    size_t j;
    double current_weight;
    matrix *weighted_adjacency_matrix;

    weighted_adjacency_matrix = init_matrix(datapoints->rows, datapoints->rows);
    if (weighted_adjacency_matrix == NULL)
    {
        return NULL;
    }
    

    /* diagonal is not set as it is set to 0 at init */
    for (i = 0; i < datapoints->rows; i++)
    {
        for (j = i + 1; j < datapoints->rows; j++)
        {
            current_weight = exp(-euclidean_diff_norm(datapoints->content[i], datapoints->content[j], datapoints->columns)/2);
            weighted_adjacency_matrix->content[i][j] = current_weight;
            weighted_adjacency_matrix->content[j][i] = current_weight;
        }
    }

    return weighted_adjacency_matrix;
}

/* doc in header */
matrix *calc_diagonal_degree_impl(matrix *weighted_adjacency_matrix){
    size_t i;
    size_t j;
    matrix *diagonal_degree_matrix;

    diagonal_degree_matrix = init_matrix(weighted_adjacency_matrix->rows, weighted_adjacency_matrix->columns);
    if (diagonal_degree_matrix == NULL)
    {
        return NULL;
    }
    

    /* sum each of the rows into the cell that's on the diagonal. Only diagonal is set as the rest was initialized to 0. */
    for (i = 0; i < weighted_adjacency_matrix->rows; i++)
    {
        for (j = 0; j < weighted_adjacency_matrix->columns; j++)
        {
            diagonal_degree_matrix->content[i][i] += weighted_adjacency_matrix->content[i][j];
        }
    }

    return diagonal_degree_matrix;
}

/* doc in header */
void calc_lnorm_impl(matrix *weighted_adjacency_matrix, matrix *diagonal_degree_matrix){
    size_t i;
    size_t j;

    /* turn diagonal_degree_matrix into D^(-1/2) */
    for (i = 0; i < diagonal_degree_matrix->rows; i++)
    {
        diagonal_degree_matrix->content[i][i] = 1/sqrt(diagonal_degree_matrix->content[i][i]);
    }
    
    /* multiply W by D^(-1/2) on the left by multiplying the rows by its diagonal values getting D^(-1/2)*W */
    for (i = 0; i < diagonal_degree_matrix->rows; i++)
    {
        for (j = 0; j < weighted_adjacency_matrix->columns; j++)
        {
            weighted_adjacency_matrix->content[i][j] *= diagonal_degree_matrix->content[i][i];
        }
    }
    
    /* multiply D^(-1/2)*W by D^(-1/2) on the right by multiplying the columns by its diagonal values getting D^(-1/2)*W*D^(-1/2) */
    for (i = 0; i < diagonal_degree_matrix->rows; i++)
    {
        for (j = 0; j < weighted_adjacency_matrix->columns; j++)
        {
            weighted_adjacency_matrix->content[j][i] *= diagonal_degree_matrix->content[i][i];
        }
    }

    /* calculate I - D^(-1/2)*W*D^(-1/2) */
    for (i = 0; i < weighted_adjacency_matrix->rows; i++)
    {
        for (j = 0; j < weighted_adjacency_matrix->columns; j++)
        {
            if (i == j)
            {
                weighted_adjacency_matrix->content[i][j] = 1 - weighted_adjacency_matrix->content[i][j];
            }
            else
            {
                weighted_adjacency_matrix->content[i][j] = -weighted_adjacency_matrix->content[i][j];
            }
        }
        
    }
}

/* doc in header */
int fit_impl(matrix *datapoints, size_t K, size_t max_iter, matrix *centroids, double epsilon){
    size_t i;
    size_t j;
    size_t k;
    int *cluster_counts;
    matrix *cluster_sums;
    int centroids_converged;
    matrix *new_centroids;

    cluster_sums = init_matrix(K, datapoints->columns);
    if (cluster_sums == NULL)
    {
        return 1;
    }
    
    cluster_counts = calloc(K, sizeof(size_t));
    if (cluster_counts == NULL)
    {
        free_matrix(cluster_sums);
        return 1;
    }

    new_centroids = init_matrix(K, datapoints->columns);
    if(new_centroids == NULL)
    {
        free_matrix(cluster_sums);
        free(cluster_counts);
        return 1;
    }

    for(k = 0; k< max_iter; k++){
    
        for(i=0; i<datapoints->rows; i++){ 
            double distance;
            double min_distance = INFINITY;
            int min_j = 0;
            size_t n;

            for(j=0; j<K; j++){ 
                distance = euclidean_diff_norm(centroids->content[j], datapoints->content[i], datapoints->columns);
                    if (distance < min_distance){
                        min_distance = distance;
                        min_j = j;
                    }
                    
            }
            for(n = 0; n < datapoints->columns; n++){
                cluster_sums->content[min_j][n] += datapoints->content[i][n];
            }
            cluster_counts[min_j] += 1;

        }

        centroids_converged = 1;
        for(i = 0 ; i < K; i++){
            for(j = 0 ; j < datapoints->columns; j++){
                new_centroids->content[i][j] = (cluster_sums->content[i][j])/(cluster_counts[i]); 
            }
            
            if (euclidean_diff_norm(centroids->content[i], new_centroids->content[i], datapoints->columns) >= epsilon) {
                centroids_converged = 0;
            }
        }
        for(i = 0 ; i < K; i++){
            for(j = 0 ; j < datapoints->columns; j++){
                centroids->content[i][j] = new_centroids->content[i][j];
            }
        }
        if(centroids_converged){
            break;
        }

        zero_matrix(new_centroids);
        memset(cluster_counts, 0, sizeof(size_t)*K);
        zero_matrix(cluster_sums);      
    }

    free(cluster_counts);
    free_matrix(cluster_sums);
    free_matrix(new_centroids);

    return 0;
}

/* Jacobi */

/* create a representation of the rotation matrix P for the next stage of the jacobi algorithm
 * including finding the pivot and calculating c and s */
rot_matrix* calc_rotation_matrix(matrix* A){
    rot_matrix *rotation;
    double val = 0;
    size_t a;
    size_t b;
    size_t i = 0;
    size_t j = 0;
    double theta;
    int sign_theta;
    double t;

    rotation = calloc(1, sizeof(rot_matrix));

    /* find the largest off-diagonal value and set its indices as the rotation matrix' i and j */
    for(a = 0; a < A->rows; a++){
       for(b = 0; b < A->columns; b++){
            if(a!=b && fabs(A->content[a][b]) >= val ){
                i = a;
                j = b;
                val = fabs(A->content[a][b]);
            }

       } 
    }
    rotation->i = i;
    rotation->j = j;

    theta = (A->content[j][j] - A->content[i][i]) / (2*A->content[i][j]);
    sign_theta = theta >= 0 ? 1 : -1; 
    t = sign_theta / (fabs(theta) + sqrt(pow(theta, 2) + 1));
    rotation->c = 1 / (sqrt(pow(t, 2) + 1));
    rotation->s = t*rotation->c;
    return rotation;
}

/* calculate the sum of squares of the off diagonal elements in mat */
double off_squared(matrix *mat){
    size_t a;
    size_t b;
    double sum = 0;

    for(a = 0 ; a < mat->rows ; a++){
        for(b = 0; b < mat->columns ; b++){
           if(a != b){
                sum += pow(mat->content[a][b],2);
            }
        }
    }
    return sum;
}

/* doc in header */
int jacobi_impl(matrix *A, double *eigenvalues, matrix *V){
    matrix *A_tag;
    matrix *temp_V;
    size_t r;
    size_t a;
    size_t k;
    size_t i;
    size_t j;
    rot_matrix* rotation; 
    double off_A;
    double off_A_tag;
    double c;
    double s;
    
    for(a = 0 ; a < A->rows ; a++){
        V->content[a][a] = 1;
    }

    temp_V = init_matrix(V->rows, V->columns);
    if (temp_V == NULL)
    {
        return 1;
    }
    

    A_tag = init_matrix(A->rows, A->columns); 
    if (A_tag == NULL)
    {
        free_matrix(temp_V);
        return 1;
    }
    

    for(k=0; k < 100; k++)
    {
        rotation = calc_rotation_matrix(A);
        i = rotation->i;
        j = rotation->j;
        c = rotation->c;
        s = rotation->s;
        free(rotation);

        matrix_copy(A_tag, A);
        
        /* calculate A' from A using the short transformation in 1.2.1.6 */
        for(r = 0 ; r < A->rows ; r++){
            if(r != i && r!= j){
                A_tag->content[r][i] = c*A->content[r][i] - s*A->content[r][j];
                A_tag->content[i][r] = c*A->content[r][i] - s*A->content[r][j];
                A_tag->content[r][j] = c*A->content[r][j] + s*A->content[r][i];
                A_tag->content[j][r] = c*A->content[r][j] + s*A->content[r][i];
                    
            }   
        }
        A_tag->content[i][i] = pow(c,2) * A->content[i][i] + pow(s,2) * A->content[j][j] - 2*c*s*A->content[i][j];
        A_tag->content[j][j] = pow(s,2) * A->content[i][i] + pow(c,2) * A->content[j][j] + 2*c*s*A->content[i][j];
        A_tag->content[i][j] = 0;
        A_tag->content[j][i] = 0;

        matrix_copy(temp_V, V);

        /* multiply V by the rotation matrix on the left */
        for (a = 0; a < V->rows ; a++ ){
            V->content[a][i] = c*temp_V->content[a][i] - s*temp_V->content[a][j];
            V->content[a][j] = c*temp_V->content[a][j] + s*temp_V->content[a][i];
        }
        
        off_A = off_squared(A);
        off_A_tag = off_squared(A_tag);
        if(fabs(off_A - off_A_tag) <= EPSILON){
            matrix_copy(A, A_tag);
            break;
        }

        matrix_copy(A, A_tag);
    }

    free_matrix(A_tag);
    free_matrix(temp_V);

    for(a = 0; a < A->rows; a++ ) {
        eigenvalues[a] = A->content[a][a];
    }
    return 0;
}

/* main entry point for C program */
int main(int argc, char *argv[]) {
    char *goal;
    char *file_name;
    matrix *input_data;

    if (argc != 3)
    {
        printf("Invalid Input!\n");
        return 1;
    }

    goal = argv[1];
    file_name = argv[2];

    input_data = read_file_to_matrix(file_name);
    if (input_data == NULL)
    {
        printf("An Error Has Occurred\n");
        return 1;
    }

    if(strcmp(goal, "wam") == 0) {
        matrix *weighted_adjacency_matrix;
        
        weighted_adjacency_matrix = calc_weighted_adjacency_impl(input_data);
        if (weighted_adjacency_matrix == NULL)
        {
            printf("An Error Has Occurred\n");
            free_matrix(input_data);
            return 1;
        }
        
        print_matrix(weighted_adjacency_matrix);
        free_matrix(weighted_adjacency_matrix);
    }
    
    else if(strcmp(goal, "ddg") == 0) {
        matrix *weighted_adjacency_matrix;
        matrix *diagonal_degree_matrix;

        weighted_adjacency_matrix = calc_weighted_adjacency_impl(input_data);
        if (weighted_adjacency_matrix == NULL)
        {
            printf("An Error Has Occurred\n");
            free_matrix(input_data);
            return 1;
        }
        
        diagonal_degree_matrix = calc_diagonal_degree_impl(weighted_adjacency_matrix);
        if (diagonal_degree_matrix == NULL)
        {
            printf("An Error Has Occurred\n");
            free_matrix(input_data);
            free_matrix(weighted_adjacency_matrix);
            return 1;
        }

        print_matrix(diagonal_degree_matrix);
        free_matrix(diagonal_degree_matrix);
        free_matrix(weighted_adjacency_matrix);
    }

    else if(strcmp(goal, "lnorm") == 0) {
        matrix *weighted_adjacency_matrix;
        matrix *diagonal_degree_matrix;

        weighted_adjacency_matrix = calc_weighted_adjacency_impl(input_data);
        if (weighted_adjacency_matrix == NULL)
        {
            printf("An Error Has Occurred\n");
            free_matrix(input_data);
            return 1;
        }
        
        diagonal_degree_matrix = calc_diagonal_degree_impl(weighted_adjacency_matrix);
        if (diagonal_degree_matrix == NULL)
        {
            printf("An Error Has Occurred\n");
            free_matrix(input_data);
            free_matrix(weighted_adjacency_matrix);
            return 1;
        }
        
        calc_lnorm_impl(weighted_adjacency_matrix, diagonal_degree_matrix);
        print_matrix(weighted_adjacency_matrix);
        free_matrix(diagonal_degree_matrix);
        free_matrix(weighted_adjacency_matrix);
    }
    
    else if(strcmp(goal, "jacobi") == 0) {
        int err;
        matrix* V;
        double* eigenvalues;
        size_t i;

        V = init_matrix(input_data->rows, input_data->columns);
        if (V == NULL)
        {
            printf("An Error Has Occurred\n");
            free_matrix(input_data);
            return 1;
        }

        eigenvalues = calloc(input_data->rows, sizeof(double));
        if (eigenvalues == NULL)
        {
            printf("An Error Has Occurred\n");
            free_matrix(input_data);
            free_matrix(V);
            return 1;
        }
        
        err = jacobi_impl(input_data, eigenvalues, V);

        if (err != 0)
        {
            printf("An Error Has Occurred\n");
            free_matrix(input_data);
            free_matrix(V);
            free(eigenvalues);
            return 1;
        }
        
        for(i = 0; i < input_data->rows; i++ ){
            if(i == input_data->rows - 1){
                printf("%.4f\n", eigenvalues[i]);
            }
            else
            {
                printf("%.4f,", eigenvalues[i]);
            }
        }

        print_matrix(V);

        free_matrix(V);
        free(eigenvalues);
    }

    else {
        printf("Invalid Input!\n");
        free_matrix(input_data);
        return 1;
    }

    free_matrix(input_data);
    return 0;
}
