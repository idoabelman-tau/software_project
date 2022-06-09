#define PY_SSIZE_T_CLEAN
#include <Python.h>

#include <math.h>

static double euclidean_diff_norm(double *v1, double *v2, size_t d){
    size_t i;
    double distance_sum = 0;

    for (i = 0; i < d; ++i) {
        distance_sum += pow(v1[i] - v2[i], 2);
    }

    return sqrt(distance_sum);

}

static int fit_impl(double **x, size_t K, size_t d, size_t max_iter, size_t points, double **mu, double epsilon){
    size_t i;
    size_t j;
    size_t k;
    int *cluster_counts;
    double **cluster_sums;
    double *cluster_sums_buffer;
    int mu_converged;
    double **new_mu;
    double *new_mu_buffer;

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
            double min_distance = INFINITY;
            int min_j = 0;
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
            
            if (euclidean_diff_norm(mu[i], new_mu[i], d) >= epsilon) {
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

static PyObject* fit_api(PyObject *self, PyObject *args) {
    PyObject* datapoints_py;
    PyObject* datapoint_py;
    size_t point_count;
    size_t dimension;
    double* datapoints_buffer;
    double** datapoints;
    PyObject* datapoint_item_py;
    PyObject* initial_centroids_py;
    PyObject* centroid_py;
    PyObject* centroid_item_py;
    double* centroids_buffer;
    double** centroids;
    PyObject* result_centroids_py;
    PyObject* result_centroid_py;
    size_t k;
    size_t max_iter;
    double epsilon;
    int error_value;
    size_t i;
    size_t j;

    if(!PyArg_ParseTuple(args, "OOKKd:fit", &datapoints_py, &initial_centroids_py, &k, &max_iter, &epsilon)) {
        return NULL;
    }

    if (!PyList_Check(datapoints_py))
    {
        PyErr_SetString(PyExc_TypeError, "fit() argument 1 (datapoints) must be a list of lists of floats");
        return NULL;
    }
    
    if (!PyList_Check(initial_centroids_py))
    {
        PyErr_SetString(PyExc_TypeError, "fit() argument 2 (initial centroids) must be a list of lists of floats");
        return NULL;
    }

    point_count = (size_t) PyList_Size(datapoints_py);

    /* get dimension from first point and allocate 2d array accordingly */
    datapoint_py = PyList_GetItem(datapoints_py, 0);
    if (!PyList_Check(datapoint_py)){
        PyErr_BadArgument();
        return NULL;
    }
    
    dimension = (size_t) PyList_Size(datapoint_py);
    datapoints_buffer = calloc(point_count * dimension, sizeof(double));
    if (datapoints_buffer == NULL)
    {
        PyErr_NoMemory();
        return NULL;
    }

    datapoints = calloc(point_count, sizeof(double *));
    if (datapoints == NULL)
    {
        free(datapoints_buffer);
        PyErr_NoMemory();
        return NULL;
    }

    for (i=0; i < point_count; i++ ){
        datapoints[i] = datapoints_buffer + (i * dimension);
    }

    /* initialize datapoints 2d array based on python list of lists */
    for (i = 0; i < point_count; i++) {
        datapoint_py = PyList_GetItem(datapoints_py, i);
        if (!PyList_Check(datapoint_py)){
            free(datapoints);
            free(datapoints_buffer);
            PyErr_SetString(PyExc_TypeError, "fit() argument 1 (datapoints) must be a list of lists of floats");
            return NULL;
        }
        
        for (j = 0; j < dimension; j++)
        {
            datapoint_item_py = PyList_GetItem(datapoint_py, j);
            if(!PyFloat_Check(datapoint_item_py))
            {
                free(datapoints);
                free(datapoints_buffer);
                PyErr_SetString(PyExc_TypeError, "fit() argument 1 (datapoints) must be a list of lists of floats");
                return NULL;
            }

            datapoints[i][j] = PyFloat_AsDouble(datapoint_item_py);
        }
    }

    /* allocate centroids 2d array, assumed k centroids with length dimension */
    centroids_buffer = calloc(k * dimension, sizeof(double));
    if (centroids_buffer == NULL)
    {
        free(datapoints);
        free(datapoints_buffer);
        PyErr_NoMemory();
        return NULL;
    }

    centroids = calloc(k, sizeof(double *));
    if (centroids == NULL)
    {
        free(datapoints);
        free(datapoints_buffer);
        free(centroids_buffer);
        PyErr_NoMemory();
        return NULL;
    }

    for (i=0; i < k; i++ ){
        centroids[i] = centroids_buffer + (i * dimension);
    }

    /* initialize centroids based on python data */
    for (i = 0; i < k; i++) {
        centroid_py = PyList_GetItem(initial_centroids_py, i);
        if (!PyList_Check(centroid_py)){
            free(datapoints);
            free(datapoints_buffer);
            free(centroids_buffer);
            free(centroids);
            PyErr_SetString(PyExc_TypeError, "fit() argument 2 (initial centroids) must be a list of lists of floats");
            return NULL;
        }
        
        for (j = 0; j < dimension; j++)
        {
            centroid_item_py = PyList_GetItem(centroid_py, j);
            if(!PyFloat_Check(centroid_item_py))
            {
                free(datapoints);
                free(datapoints_buffer);
                free(centroids_buffer);
                free(centroids);
                PyErr_SetString(PyExc_TypeError, "fit() argument 2 (initial centroids) must be a list of lists of floats");
                return NULL;
            }

            centroids[i][j] = PyFloat_AsDouble(centroid_item_py);
        }
    }

    /* run implementation of kmeans fit with the data we have. return value is an error value,
    the actual output is the centroids array which is modified by the function */
    error_value = fit_impl(datapoints, k, dimension, max_iter, point_count, centroids, epsilon);
    if (error_value != 0)
    {
        free(datapoints);
        free(datapoints_buffer);
        free(centroids_buffer);
        free(centroids);
        PyErr_NoMemory(); /* all errors in fit are memory errors */
        return NULL;
    }
    
    /* construct python list of lists from centroids and return it */ 
    result_centroids_py = PyList_New(k);
    for (i = 0; i < k; i++)
    {
        result_centroid_py = PyList_New(dimension);
        for (j = 0; j < dimension; j++)
        {
            PyList_SetItem(result_centroid_py, j, PyFloat_FromDouble(centroids[i][j]));
        }
        
        PyList_SetItem(result_centroids_py, i, result_centroid_py);
    }

    return result_centroids_py;
    
}

static PyMethodDef capiMethods[] = {
    {"fit",                   
      (PyCFunction) fit_api, 
      METH_VARARGS,           
      PyDoc_STR("Fit data to k centroids using kmeans algorithm")}, 
    {NULL, NULL, 0, NULL}     
};


static struct PyModuleDef moduledef = {
    PyModuleDef_HEAD_INIT,
    "mykmeanssp", 
    NULL, 
    -1, 
    capiMethods 
};

PyMODINIT_FUNC
PyInit_mykmeanssp(void)
{
    PyObject *m;
    m = PyModule_Create(&moduledef);
    if (!m) {
        return NULL;
    }
    return m;
}