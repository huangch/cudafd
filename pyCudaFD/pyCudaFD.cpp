#define USE_PYTHON
#include <cudaCommon.h>
#include <cudaKernel.cuh>
#include <cudaFD.h>

static PyObject* pydbc(PyObject* self, PyObject* args)
{


	PyArrayObject *ImageObj;
	int s;
	int G;

	/* Parse tuples separately since args will differ between C fcns */
	if (PyArg_ParseTuple(args, "O!ii",
		&PyArray_Type, &ImageObj, &s, &G)
	) {
		PyObject2CArray<int, int> image(ImageObj);

		float result = 0;
		dbc(&result, image.getData(), image.getRowNum(), image.getColNum(), s, G);

		return PyFloat_FromDouble((double)result);
	}
	else {
		PyErr_SetString(PyExc_ValueError, "PyArg_ParseTuple error.");
		return NULL;

	}


}


static PyObject* pyebfd(PyObject* self, PyObject* args)
{


	PyArrayObject *ImageObj;
	int s;
	int G;

	/* Parse tuples separately since args will differ between C fcns */
	if (PyArg_ParseTuple(args, "O!ii",
		&PyArray_Type, &ImageObj, &s, &G)
		) {
		PyObject2CArray<int, int> image(ImageObj);

		float result = 0;
		ebfd(&result, image.getData(), image.getRowNum(), image.getColNum(), s, G);

		return PyFloat_FromDouble((double)result);
	}
	else {
		PyErr_SetString(PyExc_ValueError, "PyArg_ParseTuple error.");
		return NULL;

	}


}

/*  define functions in module */
static PyMethodDef pyFuncMethods[] =
{
	{ "dbc", pydbc, METH_VARARGS, "dbc." },
	{ "ebfd", pyebfd, METH_VARARGS, "ebfd." },
	{ NULL, NULL, 0, NULL }
};

/* module initialization */
PyMODINIT_FUNC initpyCudaFD(void)
{
	(void)Py_InitModule("pyCudaFD", pyFuncMethods);
	import_array();
}

