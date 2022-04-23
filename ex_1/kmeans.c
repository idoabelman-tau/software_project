#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>

#define EPSILON 0.001
#define ITER_MAX 200
#define INFINITY (1.0/0.0)

size_t dimension(FILE *fp) {
    char c;
    size_t d = 1;
    while ((c = getc(fp)) != '\n') {
        if (c == ',') {
            d++;
        }
    }

    fseek(fp, 0, SEEK_SET);
    return d;
}


size_t points_count(FILE *fp){
    size_t count = 0;
    char c;
    for (c = getc(fp); c != EOF; c = getc(fp)){
        if (c == '\n') {
            count++;
        }
    }
    fseek(fp, 0, SEEK_SET);
    return count;
}

double euclidean_diff_norm(double *v1, double *v2, size_t d){
    size_t i;
    double distance_sum = 0;

    for (i = 0; i < d; ++i) {
        distance_sum += pow(v1[i] - v2[i], 2);
    }

    return sqrt(distance_sum);

}

void centroids_write(FILE *fp, double **centroids, size_t K ,size_t d ) {
    size_t i = 0;
    size_t j = 0;

    for (i = 0; i < K; ++i) {
        for (j = 0; j < d; ++j) {
            if (j == d - 1) {
                fprintf(fp, "%.4f", centroids[i][j]);
            } else {
                fprintf(fp, "%.4f,", centroids[i][j]);
            }
        }
        fputc('\n', fp);
    }
}

int find_k_means(double **x, size_t K, size_t d, size_t max_iter, size_t points, double **mu){
    size_t i;
    size_t j;
    size_t k;
    int *cluster_counts;
    double **cluster_sums;
    double *cluster_sums_buffer;
    int mu_converged;
    double **new_mu;
    double *new_mu_buffer;

    for(i = 0 ; i < K; i++){
        for(j = 0 ; j < d; j++){
            mu[i][j] = x[i][j];  
        }
    }

    cluster_sums_buffer = calloc(K*d, sizeof(double));
    if (cluster_sums_buffer == NULL)
    {
        return 1;
    }
    
    cluster_sums = calloc(K, sizeof(double *));
    if (cluster_sums == NULL)
    {
        free(cluster_sums_buffer);
        return 1;
    }

    for (i=0; i < K; i++ ){
        cluster_sums[i] = cluster_sums_buffer + (i * d) ;
    }
    
    cluster_counts = calloc(K, sizeof(size_t));
    if (cluster_counts == NULL)
    {
        free(cluster_sums_buffer);
        free(cluster_sums);
        return 1;
    }

 
    new_mu_buffer = calloc(K*d, sizeof(double));
    if (new_mu_buffer == NULL)
    {
        free(cluster_sums_buffer);
        free(cluster_sums);
        free(cluster_counts);
        return 1;
    }

    new_mu = calloc(K, sizeof(double*));
    if (new_mu == NULL)
    {
        free(cluster_sums_buffer);
        free(cluster_sums);
        free(cluster_counts);
        free(new_mu);
        return 1;
    }

    for (i=0; i < K; i++ ){
        new_mu[i] = new_mu_buffer + (i * d);
    }

    for(k = 0; k< max_iter; k++){
    
        for(i=0; i<points; i++){ 
            double distance;
            double min_distance =INFINITY;
            int min_j;
            size_t n;

            for(j=0; j<K; j++){ 
                distance = euclidean_diff_norm(mu[j], x[i], d);
                	if (distance < min_distance){
                        min_distance = distance;
					    min_j = j;
                    }
					
            }
            for(n = 0; n < d; n++){
                cluster_sums[min_j][n] += x[i][n];
            }
            cluster_counts[min_j] += 1;

        }

        mu_converged = 1;
        for(i = 0 ; i < K; i++){
            for(j = 0 ; j < d; j++){
                new_mu[i][j] = (cluster_sums[i][j])/(cluster_counts[i]); 
            }
            
            if (euclidean_diff_norm(mu[i], new_mu[i], d) >= EPSILON) {
                mu_converged = 0;
            }
        }
        for(i = 0 ; i < K; i++){
                for(j = 0 ; j < d; j++){
                    mu[i][j] = new_mu[i][j];
                }
        }
        if(mu_converged){
            break;
        }

        memset(new_mu_buffer, 0, sizeof(double)*K*d);
        memset(cluster_counts, 0, sizeof(size_t)*K);
        memset(cluster_sums_buffer, 0, sizeof(double)*K*d);      
    }

    free(cluster_counts);
    free(cluster_sums);
    free(cluster_sums_buffer);
    free(new_mu);
    free(new_mu_buffer);

    return 0;
}

int main(int argc, char *argv[]){
    size_t K = 0;
    size_t max_iter = ITER_MAX;
    FILE *inf = NULL;
    FILE *onf = NULL;
    char *input_file;
    char *output_file;
    size_t d = 0;
    size_t points = 0;
    double point;
    size_t i;
    size_t j;
    double *centroids_buffer;
    double **centroids;
    double *points_matrix_buffer;
    double **points_matrix;
    int error_value;
    

    if (argc == 4){
        K = atoi(argv[1]);
        if(K != 0){
            input_file = argv[2];
            output_file = argv[3];
            
        }
        else{  
            printf("Invalid Input!\n");
            return 1;
        }
        
    }
    else if (argc == 5){
        K = atoi(argv[1]);
        max_iter = atoi(argv[2]);
        if((K != 0)&&(max_iter!= 0)){
            input_file = argv[3];
            output_file = argv[4];

        }
        else{
            printf("Invalid Input!\n");
            return 1;
        }
    }
    else{
        printf("Invalid Input!\n");
        return 1;
    }

    inf = fopen(input_file, "r");
    if (inf == NULL)
    {
        printf("An Error Has Occurred\n");
        return 1;
    }
    
    d = dimension(inf);
    points = points_count(inf);

    points_matrix_buffer = calloc(points*d, sizeof(double));
    if (points_matrix_buffer == NULL)
    {
        printf("An Error Has Occurred\n");
        return 1;
    }
    
    points_matrix = calloc(points, sizeof(double *));
    if (points_matrix == NULL)
    {
        printf("An Error Has Occurred\n");
        free(points_matrix_buffer);
        return 1;
    }

    for (i=0; i < points; i++ ){
        points_matrix[i] = points_matrix_buffer + (i * d);
    }

    for(i = 0; i < points; i++){
        for(j = 0; j < d; j++) {
            if( j == d-1 ){
                fscanf(inf, "%lf\n", &point);
                points_matrix[i][j] = point; 
            }
            else{
                fscanf(inf, "%lf,", &point);
                points_matrix[i][j] = point;
            }
        }
    }

    fclose(inf);

    
    centroids_buffer = calloc(K*d, sizeof(double));
    if (centroids_buffer == NULL)
    {
        printf("An Error Has Occurred\n");
        free(points_matrix_buffer);
        free(points_matrix);
        return 1;
    }

    centroids = calloc(K, sizeof(double*));
    if (centroids == NULL)
    {
        printf("An Error Has Occurred\n");
        free(points_matrix_buffer);
        free(points_matrix);
        free(centroids_buffer);
        return 1;
    }

    for (i=0; i < K; i++ ){
        centroids[i] = centroids_buffer + (i * d);
    }

    error_value = find_k_means(points_matrix, K, d, max_iter, points, centroids);
    if (error_value != 0)
    {
        printf("An Error Has Occurred\n");
        free(points_matrix_buffer);
        free(points_matrix);
        free(centroids_buffer);
        free(centroids);
        return 1;
    }
    

    onf = fopen(output_file, "w");
    if (onf == NULL)
    {
        printf("An Error Has Occurred\n");
        free(points_matrix_buffer);
        free(points_matrix);
        free(centroids_buffer);
        free(centroids);
        return 1;
    }

    centroids_write(onf, centroids, K, d);

    fclose(onf);

    free(points_matrix);
    free(points_matrix_buffer);
    free(centroids);
    free(centroids_buffer);

    return 0;
}
    