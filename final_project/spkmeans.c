#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "matrix.h"

#ifndef INFINITY
#define INFINITY (1.0/0.0)
#endif
#define EPSILON 0.00001

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

size_t* find_the_largest_abs_value(matrix* A){
    size_t *arr;
    double val = 0;
    size_t a;
    size_t b;
    arr = calloc(2, sizeof(size_t));
    for(a = 0; a < A->rows; a++){
       for(b = 0; b < A->columns; b++){
        if(a!=b && fabs(A->content[a][b]) > val ){
            arr[0] = a;
            arr[1] = b;
            val = fabs(A->content[a][b]);
        }

       } 
    }
    return arr;
}
matrix *init_rotation_matrix(matrix *A, size_t i, size_t j,  size_t rows, size_t columns){
    matrix *P;
    double theta;
    double t;
    double c;
    double s;
    int sign_theta;
    size_t a;

    P = init_matrix(rows, columns);
    theta = (A->content[j][j] - A->content[i][i]) / (2*A->content[i][j]);
    sign_theta = theta >= 0 ? 1 : -1; 
    t = sign_theta / (fabs(theta) + sqrt(pow(theta, 2) + 1));
    c = 1 / (sqrt(pow(t, 2) + 1));
    s = t*c;
    for(a = 0; a <rows; a++){
        P->content[a][a] = 1;
    }
    P->content[i][i] = c;
    P->content[j][j] = c;
    P->content[i][j] = s;
    P->content[j][i] = -s;

    return P;

}
matrix *matrix_transpose(matrix *P ,size_t rows,size_t columns){
    matrix *P_trans;
    size_t a;
    size_t b;
    P_trans = init_matrix(rows, columns);
    
    for(a = 0 ; a<rows ; a++){
        for(b = 0; b<columns ; b++){
            P_trans->content[a][b] = P->content[b][a];

        }
    }
    return P_trans;
}


double off(matrix *mat, size_t rows , size_t columns){
    size_t a;
    size_t b;
    double sum = 0;

    for(a = 0 ; a<rows ; a++){
        for(b = 0; b<columns ; b++){
           if(a != b){
                sum += pow(mat->content[a][b],2);
            }
        }
    }
    return sum;
}

void jacobi_algo(matrix *A){
    size_t rows = A->rows;
    size_t columns = A->columns;
    matrix *P;
    matrix *P_trans;
    matrix *A_tag;
    matrix *V;
    size_t i;
    size_t r;
    size_t j;
    size_t a;
    size_t b;
    size_t k;
    size_t* arr; 
    double off_A;
    double off_A_tag;
    double c;
    double s;
    V = init_matrix(rows, columns);
    for(a = 0 ; a<rows ; a++){
        for(b = 0; b<columns ; b++){
           if(a == b){
            V->content[a][b] = 1;
            }
        }
    }
        

    for(k=0; k < 100; k++)
    {
        A_tag = init_matrix(rows, columns);
        arr = find_the_largest_abs_value(A);
        i = arr[0];
        j = arr[1];
        P = init_rotation_matrix(A, i, j, rows, columns);
        P_trans = matrix_transpose(P , rows, columns);
        c = P->content[i][i];
        s = P->content[i][j];
        for(a = 0 ; a<rows ; a++){
            for(b = 0; b<columns ; b++){
                A_tag->content[a][b] = A->content[a][b];
            }
        }
        for(r = 0 ; r<rows ; r++){
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
        V = multiply_matrices(V, P);
        off_A = off(A, rows, columns);
        off_A_tag = off(A_tag, rows, columns);
        if(fabs(off_A - off_A_tag) <= EPSILON){
            /* A = A_tag */
            for(a = 0 ; a<rows ; a++){
                for(b = 0; b<columns ; b++){
                    A->content[a][b] = A_tag->content[a][b];  
                }
            }
            free_matrix(A_tag);
            free_matrix(P);
            free_matrix(P_trans);
            free(arr);
            break;
        }

       /* A = A_tag */
       
        for(a = 0 ; a<rows ; a++){
            for(b = 0; b<columns ; b++){
                A->content[a][b] = A_tag->content[a][b];  
                }
            }
       
        free_matrix(A_tag);
        free_matrix(P);
        free_matrix(P_trans);
        free(arr);
    }
        
    for(a = 0; a < rows; a++ ){
        for(b = 0; b < columns; b++){
            if(a==b && b == columns -1){
                printf("%.4f\n" ,A->content[a][b]);
            }
            else if (a == b)
            {
                printf("%.4f," ,A->content[a][b]);
            }
            
        }
    }

    print_matrix(V);   
     
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
    
    if(strcmp(goal, "jacobi") == 0) {
        jacobi_algo(input_data);

    }

    free_matrix(input_data);
    return 0;
}
