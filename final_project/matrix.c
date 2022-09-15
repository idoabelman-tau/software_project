#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "matrix.h"

/* doc in header */
matrix *init_matrix(size_t rows, size_t columns) {
    size_t i;

    matrix *mat = (matrix *) malloc(sizeof(matrix));
    if (mat == NULL)
    {
        return NULL;
    }
    

    mat->buffer = calloc(rows*columns, sizeof(double));
    if (mat->buffer == NULL)
    {
        free(mat);
        return NULL;
    }
    
    mat->content = calloc(rows, sizeof(double *));
    if (mat->content == NULL)
    {
        free(mat->buffer);
        free(mat);
        return NULL;
    }
    for (i=0; i < rows; i++ ){
        mat->content[i] = mat->buffer + (i * columns);
    }

    mat->rows = rows;
    mat->columns = columns;
    return mat;
}

/* doc in header */
void free_matrix(matrix *mat) {
    free(mat->content);
    free(mat->buffer);
    free(mat);
}

/* doc in header */
double euclidean_diff_norm(double *v1, double *v2, size_t d){
    size_t i;
    double distance_sum = 0;

    for (i = 0; i < d; ++i) {
        distance_sum += pow(v1[i] - v2[i], 2);
    }

    return sqrt(distance_sum);
}

/* counts the amount of columns in a csv file representing a matrix by counting the
commas in the first line, then returning to the start. */
static size_t file_columns_count(FILE *fp) {
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

/* counts the amount of rows in a csv file representing a matrix by counting the
line feeds, then returning to the start. */
static size_t file_rows_count(FILE *fp){
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

/* doc in header */
matrix *read_file_to_matrix(char *file_name) {
    FILE *file;
    size_t i;
    size_t j;
    matrix *data;
    double point;

    file = fopen(file_name, "r");
    if (file == NULL)
    {
        return NULL;
    }
    
    data = init_matrix(file_rows_count(file), file_columns_count(file));
    if (data == NULL)
    {
        return NULL;
    }
    

    for(i = 0; i < data->rows; i++){
        for(j = 0; j < data->columns; j++) {
            if( j == data->columns - 1 ){
                fscanf(file, "%lf\n", &point);
                data->content[i][j] = point; 
            }
            else{
                fscanf(file, "%lf,", &point);
                data->content[i][j] = point;
            }
        }
    }

    fclose(file);
    return data;
}

/* doc in header */
void print_matrix(matrix *mat) {
    size_t i;
    size_t j;

    for (i = 0; i < mat->rows; ++i) {
        for (j = 0; j < mat->columns; ++j) {
            if (j == mat->columns - 1) {
                printf("%.4f\n", mat->content[i][j]);
            } else {
                printf("%.4f,", mat->content[i][j]);
            }
        }
    }
}

/* doc in header */
void zero_matrix(matrix *mat) {
    memset(mat->buffer, 0, sizeof(double) * mat->rows * mat->columns);
}

/* doc in header */
void matrix_copy(matrix *dest, matrix *source) {
    memcpy(dest->buffer, source->buffer, sizeof(double) * source->rows * source->columns);
}
