#define PY_SSIZE_T_CLEAN
#include <Python.h>
#include "matrix.h"
#include "spkmeans.h"

/* convert a python nested list of floats to a matrix.
 * It is the caller's responsibility to free the returned matrix by calling free_matrix().
 * If the argument is not a nested list of floats or there is not enough memory NULL is returned
 * and a matching python error is set. 
 */
static matrix* python_list_to_matrix(PyObject *mat_py) {
    size_t rows;
    size_t columns;
    matrix* mat;
    PyObject* row_py;
    PyObject* item_py;
    size_t i;
    size_t j;

    if (!PyList_Check(mat_py))
    {
        PyErr_SetString(PyExc_TypeError, "matrix argument must be a list of lists of floats");
        return NULL;
    }

    rows = (size_t) PyList_Size(mat_py);

    /* get columns count from first row and initialize matrix array accordingly */
    row_py = PyList_GetItem(mat_py, 0);
    if (!PyList_Check(row_py)){
        PyErr_SetString(PyExc_TypeError, "matrix argument must be a list of lists of floats");
        return NULL;
    }
    
    columns = (size_t) PyList_Size(row_py);

    mat = init_matrix(rows, columns);
    if (mat == NULL)
    {
        PyErr_NoMemory();
        return NULL;
    }

    /* initialize datapoints 2d array based on python list of lists */
    for (i = 0; i < rows; i++) {
        row_py = PyList_GetItem(mat_py, i);
        if (!PyList_Check(row_py)){
            free_matrix(mat);
            PyErr_SetString(PyExc_TypeError, "matrix argument must be a list of lists of floats");
            return NULL;
        }
        
        for (j = 0; j < columns; j++)
        {
            item_py = PyList_GetItem(row_py, j);
            if(!PyFloat_Check(item_py))
            {
                free_matrix(mat);
                PyErr_SetString(PyExc_TypeError, "matrix argument must be a list of lists of floats");
                return NULL;
            }

            mat->content[i][j] = PyFloat_AsDouble(item_py);
        }
    }

    return mat;
}

/* converts a matrix to a python nested list of floats */
static PyObject* matrix_to_python_list(matrix* mat) {
    PyObject* mat_py;
    PyObject* row_py;
    size_t i;
    size_t j;
    
    mat_py = PyList_New(mat->rows);

    for (i = 0; i < mat->rows; i++)
    {
        row_py = PyList_New(mat->columns);
        for (j = 0; j < mat->columns; j++)
        {
            PyList_SetItem(row_py, j, PyFloat_FromDouble(mat->content[i][j]));
        }
        
        PyList_SetItem(mat_py, i, row_py);
    }

    return mat_py;
}

/* api wrapper for implementations of calc_weighted_adjacency_matrix */
static PyObject* calc_weighted_adjacency_api(PyObject *self, PyObject *datapoints_py) {
    matrix* datapoints;
    matrix* weighted_adjacency_matrix;

    datapoints = python_list_to_matrix(datapoints_py);
    if (datapoints == NULL)
    {
        /* error set inside python_list_to_matrix */
        return NULL;
    }

    weighted_adjacency_matrix = calc_weighted_adjacency_impl(datapoints);
    if (weighted_adjacency_matrix == NULL)
    {
        PyErr_NoMemory(); /* all errors in impl function are memory errors */
        return NULL;
    }

    return matrix_to_python_list(weighted_adjacency_matrix);
}

/* api wrapper for implementations of calc_diagonal_degree_matrix */
static PyObject* calc_diagonal_degree_api(PyObject *self, PyObject *weighted_adjacency_matrix_py) {
    matrix* weighted_adjacency_matrix;
    matrix* diagonal_degree_matrix;

    weighted_adjacency_matrix = python_list_to_matrix(weighted_adjacency_matrix_py);
    if (weighted_adjacency_matrix == NULL)
    {
        /* error set inside python_list_to_matrix */
        return NULL;
    }

    diagonal_degree_matrix = calc_diagonal_degree_impl(weighted_adjacency_matrix);
    if (diagonal_degree_matrix == NULL)
    {
        PyErr_NoMemory(); /* all errors in impl function are memory errors */
        return NULL;
    }

    return matrix_to_python_list(diagonal_degree_matrix);
}

/* api wrapper for implementations of calc_lnorm. Note that unlike the implementation function,
 * the wrapper function actually returns lnorm as a python list of lists and does not modify
 * the python arguments it recieves.
 */
static PyObject* calc_lnorm_api(PyObject *self, PyObject *args) {
    PyObject* weighted_adjacency_matrix_py;
    PyObject* diagonal_degree_matrix_py;
    matrix* weighted_adjacency_matrix;
    matrix* diagonal_degree_matrix;

    if(!PyArg_ParseTuple(args, "OO:calc_lnorm", &weighted_adjacency_matrix_py, &diagonal_degree_matrix_py)) {
        return NULL;
    }

    weighted_adjacency_matrix = python_list_to_matrix(weighted_adjacency_matrix_py);
    if (weighted_adjacency_matrix == NULL)
    {
        /* error set inside python_list_to_matrix */
        return NULL;
    }

    diagonal_degree_matrix = python_list_to_matrix(diagonal_degree_matrix_py);
    if (diagonal_degree_matrix == NULL)
    {
        /* error set inside python_list_to_matrix */
        return NULL;
    }

    calc_lnorm_impl(weighted_adjacency_matrix, diagonal_degree_matrix);
    return matrix_to_python_list(weighted_adjacency_matrix);
}

/* api wrapper for implementations of fit */
static PyObject* fit_api(PyObject *self, PyObject *args) {
    PyObject* datapoints_py;
    PyObject* initial_centroids_py;
    matrix* datapoints;
    matrix* centroids;
    size_t k;
    size_t max_iter;
    double epsilon;
    int error_value;

    if(!PyArg_ParseTuple(args, "OOKKd:fit", &datapoints_py, &initial_centroids_py, &k, &max_iter, &epsilon)) {
        return NULL;
    }

    datapoints = python_list_to_matrix(datapoints_py);
    if (datapoints == NULL)
    {
        /* error set inside python_list_to_matrix */
        return NULL;
    }
    
    centroids = python_list_to_matrix(initial_centroids_py);
    if (centroids == NULL)
    {
        /* error set inside python_list_to_matrix */
        return NULL;
    }

    /* run implementation of kmeans fit with the data we have. return value is an error value,
    the actual output is the centroids array which is modified by the function */
    error_value = fit_impl(datapoints, k, max_iter, centroids, epsilon);
    if (error_value != 0)
    {
        free_matrix(datapoints);
        free_matrix(centroids);
        PyErr_NoMemory(); /* all errors in fit are memory errors */
        return NULL;
    }
    
    return matrix_to_python_list(centroids);
}

static PyObject* jacobi_api(PyObject *self, PyObject *A_py) {
    matrix* A;
    matrix* V;
    double* eigenvalues;
    PyObject* eigenvalues_py;
    PyObject* V_py;
    int err;
    size_t i;

    A = python_list_to_matrix(A_py);
    if(A == NULL) {
        /* error set inside python_list_to_matrix */
        return NULL;
    }

    V = init_matrix(A->rows, A->columns);
    if(V == NULL) {
        PyErr_NoMemory();
        return NULL;
    }

    eigenvalues = calloc(A->rows, sizeof(double));
    if(eigenvalues == NULL) {
        PyErr_NoMemory();
        return NULL;
    }

    err = jacobi_impl(A, eigenvalues, &V);
    if (err != 0)
    {
        PyErr_NoMemory();
        return NULL;
    }
    
    eigenvalues_py = PyList_New(A->rows);
    for (i = 0; i < A->rows; i++)
    {
        PyList_SetItem(eigenvalues_py, i, PyFloat_FromDouble(eigenvalues[i]));
    }
    

    V_py = matrix_to_python_list(V);
    return PyTuple_Pack(2, eigenvalues_py, V_py);
}

static PyMethodDef capiMethods[] = {
    {"calc_weighted_adjacency_matrix",                   
      (PyCFunction) calc_weighted_adjacency_api, 
      METH_O,           
      PyDoc_STR("Calculate the weighted adjacency matrix based on datapoints")}, 
    {"calc_diagonal_degree_matrix",                   
      (PyCFunction) calc_diagonal_degree_api, 
      METH_O,           
      PyDoc_STR("Calculate the diagonal degree matrix based on weighted adjacency matrix")}, 
    {"calc_lnorm",                   
      (PyCFunction) calc_lnorm_api, 
      METH_VARARGS,           
      PyDoc_STR("Calculate the lnorm based on weighted adjacency matrix and diagonal degree matrix")}, 
    {"fit",                   
      (PyCFunction) fit_api, 
      METH_VARARGS,           
      PyDoc_STR("Fit data to k centroids using kmeans algorithm")}, 
    {"jacobi",                   
      (PyCFunction) jacobi_api, 
      METH_O,           
      PyDoc_STR("Diagonalize matrix using jacobi, returning a tuple of the eigenvalues and the the diagonalization matrix (V)")}, 
    {NULL, NULL, 0, NULL}     
};


static struct PyModuleDef moduledef = {
    PyModuleDef_HEAD_INIT,
    "myspkmeans", 
    NULL, 
    -1, 
    capiMethods 
};

PyMODINIT_FUNC
PyInit_myspkmeans(void)
{
    PyObject *m;
    m = PyModule_Create(&moduledef);
    if (!m) {
        return NULL;
    }
    return m;
}