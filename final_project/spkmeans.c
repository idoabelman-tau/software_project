#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "matrix.h"

#ifndef INFINITY
#define INFINITY (1.0/0.0)
#endif

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
            return 1;
        }
        
        print_matrix(weighted_adjacency_matrix);
        free_matrix(weighted_adjacency_matrix);
    }
    
    if(strcmp(goal, "ddg") == 0) {
        matrix *weighted_adjacency_matrix;
        matrix *diagonal_degree_matrix;

        weighted_adjacency_matrix = calc_weighted_adjacency_impl(input_data);
        if (weighted_adjacency_matrix == NULL)
        {
            printf("An Error Has Occurred\n");
            return 1;
        }
        
        diagonal_degree_matrix = calc_diagonal_degree_impl(weighted_adjacency_matrix);
        if (diagonal_degree_matrix == NULL)
        {
            printf("An Error Has Occurred\n");
            free_matrix(weighted_adjacency_matrix);
            return 1;
        }

        print_matrix(diagonal_degree_matrix);
        free_matrix(diagonal_degree_matrix);
        free_matrix(weighted_adjacency_matrix);
    }

    if(strcmp(goal, "lnorm") == 0) {
        matrix *weighted_adjacency_matrix;
        matrix *diagonal_degree_matrix;

        weighted_adjacency_matrix = calc_weighted_adjacency_impl(input_data);
        if (weighted_adjacency_matrix == NULL)
        {
            printf("An Error Has Occurred\n");
            return 1;
        }
        
        diagonal_degree_matrix = calc_diagonal_degree_impl(weighted_adjacency_matrix);
        if (diagonal_degree_matrix == NULL)
        {
            printf("An Error Has Occurred\n");
            free_matrix(weighted_adjacency_matrix);
            return 1;
        }
        
        calc_lnorm_impl(weighted_adjacency_matrix, diagonal_degree_matrix);
        print_matrix(weighted_adjacency_matrix);
        free_matrix(diagonal_degree_matrix);
        free_matrix(weighted_adjacency_matrix);
    }

    free_matrix(input_data);
    return 0;
}
