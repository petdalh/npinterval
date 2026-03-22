#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include <Python.h>
#include <numpy/arrayobject.h>
#include <numpy/npy_math.h>
#include <numpy/ufuncobject.h>
#include <stdio.h>
#include "structmember.h"

#include "interval.h"


typedef struct {
    PyObject_HEAD
    interval obval;
} PyInterval;

static PyTypeObject PyInterval_Type;

PyArray_Descr* interval_descr;


static inline int
PyInterval_Check(PyObject* object) {
  return PyObject_IsInstance(object,(PyObject*)&PyInterval_Type);
}

static PyObject*
PyInterval_FromInterval(interval q) {
  PyInterval* p = (PyInterval*)PyInterval_Type.tp_alloc(&PyInterval_Type,0);
  if (p) { p->obval = q; }
  return (PyObject*)p;
}


#define PyInterval_AsInterval(q, o)                                     \
  if(PyInterval_Check(o)) {                                             \
    q = ((PyInterval*)o)->obval;                                        \
  } else {                                                              \
    PyErr_SetString(PyExc_TypeError,                                    \
                    "Input object is not an interval.");                \
    return NULL;                                                        \
  }

#define PyInterval_AsIntervalPointer(q, o)                              \
  if(PyInterval_Check(o)) {                                             \
    q = &((PyInterval*)o)->obval;                                       \
  } else {                                                              \
    PyErr_SetString(PyExc_TypeError,                                    \
                    "Input object is not an interval.");                \
    return NULL;                                                        \
  }

static PyObject *
pyinterval_new(PyTypeObject *type, PyObject *NPY_UNUSED(args), PyObject *NPY_UNUSED(kwds))
{
  PyInterval* self;
  self = (PyInterval *)type->tp_alloc(type, 0);
  return (PyObject *)self;
}

static int
pyinterval_init(PyObject *self, PyObject *args, PyObject *kwds)
{
    Py_ssize_t size = PyTuple_Size(args);
    interval* i;
    PyObject* I = {0};
    i = &(((PyInterval*)self)->obval);

    if (kwds && PyDict_Size(kwds)) {
        PyErr_SetString(PyExc_TypeError,
                        "interval constructor takes no keyword arguments");
        return -1;
    }

    i->l = 0.0;
    i->u = 0.0;
    if(size == 0) {
        return 0;
    } else if(size == 1) {
        if(PyArg_ParseTuple(args, "O", &I) && PyInterval_Check(I)) {
            i->l = ((PyInterval*)I)->obval.l;
            i->u = ((PyInterval*)I)->obval.u;
            return 0;
        } else if(PyArg_ParseTuple(args, "d", &i->l)) {
            i->u = i->l;
            return 0;
        }
    } else if(size == 2 && PyArg_ParseTuple(args, "dd", &i->l, &i->u)) {
        return 0;
    }

    PyErr_SetString(PyExc_TypeError,
                    "interval constructor takes zero, one, or two arguments, or an interval");
    return -1;
}

#define UNARY_BOOL_RETURNER(name)                                       \
  static PyObject*                                                      \
  pyinterval_##name(PyObject* a, PyObject* NPY_UNUSED(b)) {             \
    interval i = {0.0, 0.0};                                            \
    PyInterval_AsInterval(i, a);                                        \
    return PyBool_FromLong(interval_##name(i));                         \
  }

UNARY_BOOL_RETURNER(nonzero)

#define BINARY_BOOL_RETURNER(name)                                      \
  static PyObject*                                                      \
  pyinterval_##name(PyObject* a, PyObject* b) {                          \
    interval i = {0.0, 0.0};                                  \
    interval j = {0.0, 0.0};                                  \
    PyInterval_AsInterval(i, a);                                        \
    PyInterval_AsInterval(j, b);                                        \
    return PyBool_FromLong(interval_##name(i,j));                       \
  }

BINARY_BOOL_RETURNER(equal)
BINARY_BOOL_RETURNER(not_equal)
BINARY_BOOL_RETURNER(subseteq)
BINARY_BOOL_RETURNER(supseteq)
BINARY_BOOL_RETURNER(subset)
BINARY_BOOL_RETURNER(supset)

#define UNARY_FLOAT_RETURNER(name)                                      \
  static PyObject*                                                      \
  pyinterval_##name(PyObject* a, PyObject* NPY_UNUSED(b)) {              \
    interval i = {0.0, 0.0};                                  \
    PyInterval_AsInterval(i, a);                                        \
    return PyFloat_FromDouble(interval_##name(i));                      \
  }
UNARY_FLOAT_RETURNER(norm)

#define UNARY_INTERVAL_RETURNER(name)                                   \
  static PyObject*                                                      \
  pyinterval_##name(PyObject* a, PyObject* NPY_UNUSED(b)) {             \
    interval i = {0.0, 0.0};                                            \
    PyInterval_AsInterval(i, a);                                        \
    return PyInterval_FromInterval(interval_##name(i));                 \
  }
UNARY_INTERVAL_RETURNER(negative)
UNARY_INTERVAL_RETURNER(inverse)
UNARY_INTERVAL_RETURNER(sin)
UNARY_INTERVAL_RETURNER(cos)
UNARY_INTERVAL_RETURNER(tan)
UNARY_INTERVAL_RETURNER(arctan)
UNARY_INTERVAL_RETURNER(tanh)
UNARY_INTERVAL_RETURNER(exp)
UNARY_INTERVAL_RETURNER(sqrt)
UNARY_INTERVAL_RETURNER(square)
UNARY_INTERVAL_RETURNER(abs)

static PyObject*
pyinterval_positive(PyObject* self, PyObject* NPY_UNUSED(b)) {
  Py_INCREF(self);
  return self;
}

#define II_BINARY_INTERVAL_RETURNER(name)                               \
  static PyObject*                                                      \
  pyinterval_##name(PyObject* a, PyObject* b) {                         \
    interval i = {0.0, 0.0};                                            \
    interval j = {0.0, 0.0};                                            \
    PyInterval_AsInterval(i, a);                                        \
    PyInterval_AsInterval(j, b);                                        \
    return PyInterval_FromInterval(interval_##name(i,j));             \
  }

II_BINARY_INTERVAL_RETURNER(union)
II_BINARY_INTERVAL_RETURNER(intersection)
II_BINARY_INTERVAL_RETURNER(maximum)
II_BINARY_INTERVAL_RETURNER(minimum)

// Read a single integer element of any numpy integer dtype as a double.
static inline double
read_integer_element(const char* ptr, int elsize, int is_unsigned) {
    if (is_unsigned) {
        switch(elsize) {
            case 1: return (double)*((const npy_uint8*)ptr);
            case 2: return (double)*((const npy_uint16*)ptr);
            case 4: return (double)*((const npy_uint32*)ptr);
            default: return (double)*((const npy_uint64*)ptr);
        }
    } else {
        switch(elsize) {
            case 1: return (double)*((const npy_int8*)ptr);
            case 2: return (double)*((const npy_int16*)ptr);
            case 4: return (double)*((const npy_int32*)ptr);
            default: return (double)*((const npy_int64*)ptr);
        }
    }
}

#define II_IS_SI_BINARY_INTERVAL_RETURNER_FULL(fake_name, name)         \
static PyObject*                                                        \
pyinterval_##fake_name##_array_operator(PyObject* a, PyObject* b) {     \
  NpyIter *iter;                                                        \
  NpyIter_IterNextFunc *iternext;                                       \
  PyArrayObject *op[2];                                                 \
  PyObject *ret;                                                        \
  npy_uint32 flags;                                                     \
  npy_uint32 op_flags[2];                                               \
  PyArray_Descr *op_dtypes[2];                                          \
  npy_intp itemsize, *innersizeptr, innerstride;                        \
  char **dataptrarray;                                                  \
  char *src, *dst;                                                      \
  interval p = {0.0, 0.0};                                              \
  PyInterval_AsInterval(p, a);                                          \
  flags = NPY_ITER_EXTERNAL_LOOP;                                       \
  op[0] = (PyArrayObject *) b;                                          \
  op[1] = NULL;                                                         \
  op_flags[0] = NPY_ITER_READONLY;                                      \
  op_flags[1] = NPY_ITER_WRITEONLY | NPY_ITER_ALLOCATE;                 \
  op_dtypes[0] = PyArray_DESCR((PyArrayObject*) b);                     \
  op_dtypes[1] = interval_descr;                                        \
  iter = NpyIter_MultiNew(2, op, flags, NPY_KEEPORDER, NPY_NO_CASTING, op_flags, op_dtypes); \
  if (iter == NULL) {                                                   \
    return NULL;                                                        \
  }                                                                     \
  iternext = NpyIter_GetIterNext(iter, NULL);                           \
  innerstride = NpyIter_GetInnerStrideArray(iter)[0];                   \
  itemsize = NpyIter_GetDescrArray(iter)[1]->elsize;                    \
  innersizeptr = NpyIter_GetInnerLoopSizePtr(iter);                     \
  dataptrarray = NpyIter_GetDataPtrArray(iter);                         \
  if(PyArray_EquivTypes(PyArray_DESCR((PyArrayObject*) b), interval_descr)) { \
    npy_intp i;                                                         \
    do {                                                                \
      npy_intp size = *innersizeptr;                                    \
      src = dataptrarray[0];                                            \
      dst = dataptrarray[1];                                            \
      for(i = 0; i < size; i++, src += innerstride, dst += itemsize) {  \
        *((interval *) dst) = interval_##name(p, *((interval *) src));  \
      }                                                                 \
    } while (iternext(iter));                                           \
  } else if(PyArray_ISFLOAT((PyArrayObject*) b)) {                      \
    npy_intp i;                                                         \
    do {                                                                \
      npy_intp size = *innersizeptr;                                    \
      src = dataptrarray[0];                                            \
      dst = dataptrarray[1];                                            \
      for(i = 0; i < size; i++, src += innerstride, dst += itemsize) {  \
        *(interval *) dst = interval_##name##_scalar(p, *((double *) src)); \
      }                                                                 \
    } while (iternext(iter));                                           \
  } else if(PyArray_ISINTEGER((PyArrayObject*) b)) {                    \
    npy_intp i;                                                         \
    int int_elsize = PyArray_DESCR((PyArrayObject*) b)->elsize;         \
    int int_is_unsigned = PyArray_ISUNSIGNED((PyArrayObject*) b);       \
    do {                                                                \
      npy_intp size = *innersizeptr;                                    \
      src = dataptrarray[0];                                            \
      dst = dataptrarray[1];                                            \
      for(i = 0; i < size; i++, src += innerstride, dst += itemsize) {  \
        double dval = read_integer_element(src, int_elsize, int_is_unsigned); \
        *((interval *) dst) = interval_##name##_scalar(p, dval);       \
      }                                                                 \
    } while (iternext(iter));                                           \
  } else {                                                              \
    NpyIter_Deallocate(iter);                                           \
    return NULL;                                                        \
  }                                                                     \
  ret = (PyObject *) NpyIter_GetOperandArray(iter)[1];                  \
  Py_INCREF(ret);                                                       \
  if (NpyIter_Deallocate(iter) != NPY_SUCCEED) {                        \
    Py_DECREF(ret);                                                     \
    return NULL;                                                        \
  }                                                                     \
  return ret;                                                           \
}                                                                       \
static PyObject*                                                        \
pyinterval_##fake_name(PyObject* a, PyObject* b) {                    \
  npy_int64 val64;                                                     \
  npy_int32 val32;                                                     \
  interval p = {0.0, 0.0};                                 \
  if(PyArray_Check(b)) { return pyinterval_##fake_name##_array_operator(a, b); } \
  if(PyFloat_Check(a) && PyInterval_Check(b)) {                      \
    return PyInterval_FromInterval(interval_scalar_##name(PyFloat_AsDouble(a), ((PyInterval*)b)->obval)); \
  }                                                                    \
  if(PyLong_Check(a) && PyInterval_Check(b)) {                        \
    return PyInterval_FromInterval(interval_scalar_##name(PyLong_AsLong(a), ((PyInterval*)b)->obval)); \
  }                                                                    \
  PyInterval_AsInterval(p, a);                                     \
  if(PyInterval_Check(b)) {                                          \
    return PyInterval_FromInterval(interval_##name(p,((PyInterval*)b)->obval)); \
  } else if(PyFloat_Check(b)) {                                        \
    return PyInterval_FromInterval(interval_##name##_scalar(p,PyFloat_AsDouble(b))); \
  } else if(PyLong_Check(b)) {                                          \
    return PyInterval_FromInterval(interval_##name##_scalar(p,PyLong_AsLong(b))); \
  } else if(PyObject_TypeCheck(b, &PyInt64ArrType_Type)) {             \
    PyArray_ScalarAsCtype(b, &val64);                                  \
    return PyInterval_FromInterval(interval_##name##_scalar(p, val64)); \
  } else if(PyObject_TypeCheck(b, &PyInt32ArrType_Type)) {             \
    PyArray_ScalarAsCtype(b, &val32);                                  \
    return PyInterval_FromInterval(interval_##name##_scalar(p, val32)); \
  }                                                                    \
  Py_RETURN_NOTIMPLEMENTED; \
}
#define II_IS_SI_BINARY_INTERVAL_RETURNER(name) II_IS_SI_BINARY_INTERVAL_RETURNER_FULL(name, name)
II_IS_SI_BINARY_INTERVAL_RETURNER(add)
II_IS_SI_BINARY_INTERVAL_RETURNER(subtract)
II_IS_SI_BINARY_INTERVAL_RETURNER(multiply)
II_IS_SI_BINARY_INTERVAL_RETURNER(divide)


#define II_IS_SI_BINARY_INTERVAL_INPLACE_FULL(fake_name, name)        \
  static PyObject*                                                      \
  pyinterval_inplace_##fake_name(PyObject* a, PyObject* b) {          \
    interval* p = {0};                                                \
    if(PyFloat_Check(a) || PyLong_Check(a)) {                            \
      PyErr_SetString(PyExc_TypeError, "Cannot in-place "#fake_name" a scalar by a interval; should be handled by python."); \
      return NULL;                                                      \
    }                                                                   \
    PyInterval_AsIntervalPointer(p, a);                             \
    if(PyInterval_Check(b)) {                                         \
      interval_inplace_##name(p,((PyInterval*)b)->obval);           \
      Py_INCREF(a);                                                     \
      return a;                                                         \
    } else if(PyFloat_Check(b)) {                                       \
      interval_inplace_##name##_scalar(p,PyFloat_AsDouble(b));        \
      Py_INCREF(a);                                                     \
      return a;                                                         \
    } else if(PyLong_Check(b)) {                                         \
      interval_inplace_##name##_scalar(p,PyLong_AsLong(b));            \
      Py_INCREF(a);                                                     \
      return a;                                                         \
    }                                                                   \
    Py_RETURN_NOTIMPLEMENTED; \
  }
#define II_IS_SI_BINARY_INTERVAL_INPLACE(name) II_IS_SI_BINARY_INTERVAL_INPLACE_FULL(name, name)
II_IS_SI_BINARY_INTERVAL_INPLACE(add)
II_IS_SI_BINARY_INTERVAL_INPLACE(subtract)
II_IS_SI_BINARY_INTERVAL_INPLACE(multiply)
II_IS_SI_BINARY_INTERVAL_INPLACE(divide)


#define IS_BINARY_INTERVAL_RETURNER(name)                               \
  static PyObject*                                                      \
  pyinterval_##name(PyObject* a, PyObject* b) {                         \
    interval i = {0.0, 0.0};                                            \
    PyInterval_AsInterval(i, a);                                        \
    if(PyFloat_Check(b)) {                                              \
      return PyInterval_FromInterval(interval_##name##_scalar(i, PyFloat_AsDouble(b))); \
    } else if (PyLong_Check(b)) {                                       \
      return PyInterval_FromInterval(interval_##name##_scalar(i, PyLong_AsLong(b)));\
    }                                                                   \
    Py_RETURN_NOTIMPLEMENTED;                                           \
  }

IS_BINARY_INTERVAL_RETURNER(power)

static PyObject*
pyinterval_inplace_power(PyObject* a, PyObject* b) {
  interval* p = {0};
  if(PyFloat_Check(a) || PyLong_Check(a)) {
    PyErr_SetString(PyExc_TypeError, "Cannot in-place power a scalar by a interval; should be handled by python.");
    return NULL;
  }
  PyInterval_AsIntervalPointer(p, a);
  if(PyInterval_Check(b)) {
    Py_RETURN_NOTIMPLEMENTED;
  } else if(PyFloat_Check(b)) {
    interval_inplace_power_scalar(p,PyFloat_AsDouble(b));
    Py_INCREF(a);
    return a;
  } else if(PyLong_Check(b)) {
    interval_inplace_power_scalar(p,PyLong_AsLong(b));
    Py_INCREF(a);
    return a;
  }
  Py_RETURN_NOTIMPLEMENTED;
}


static PyObject *
pyinterval__reduce(PyInterval* self)
{
  return Py_BuildValue("O(OO)", Py_TYPE(self),
                       PyFloat_FromDouble(self->obval.l), PyFloat_FromDouble(self->obval.u));
}

static PyObject *
pyinterval_getstate(PyInterval* self, PyObject* args)
{
  if (!PyArg_ParseTuple(args, ":getstate"))
    return NULL;
  return Py_BuildValue("OO",
                       PyFloat_FromDouble(self->obval.l), PyFloat_FromDouble(self->obval.u));
}

static PyObject *
pyinterval_setstate(PyInterval* self, PyObject* args)
{
  interval* q;
  q = &(self->obval);

  if (!PyArg_ParseTuple(args, "dd:setstate", &q->l, &q->u)) {
    return NULL;
  }
  Py_INCREF(Py_None);
  return Py_None;
}

PyMethodDef pyinterval_methods[] = {
  // Unary bool returners
  {"nonzero", pyinterval_nonzero, METH_O,
   "True if the interval is PRECISELY nonzero"},
  // Binary bool returners
  {"equal", pyinterval_equal, METH_O,
   "True if the intervals are PRECISELY equal"},
  {"not_equal", pyinterval_not_equal, METH_O,
   "True if the intervals are not PRECISELY equal"},
  {"subseteq", pyinterval_subseteq, METH_O,
   "True if i1 is a subset (inclusive) of i2"},
  {"supseteq", pyinterval_supseteq, METH_O,
   "True if i1 is a superset (inclusive) of i2"},
  {"subset", pyinterval_subset, METH_O,
   "True if i1 is a subset (strict) of i2"},
  {"supset", pyinterval_supset, METH_O,
   "True if i1 is a superset (strict) of i2"},
  // Unary float returners
  {"norm", pyinterval_norm, METH_NOARGS,
   "Measure of the interval"},
  // Unary interval returners
  {"inverse", pyinterval_inverse, METH_NOARGS,
   "Return the inverse of the interval"},
  {"reciprocal", pyinterval_inverse, METH_NOARGS,
   "Return the reciprocal of the interval"},
  {"sin", pyinterval_sin, METH_NOARGS,
   "Return the sine of the interval"},
  {"cos", pyinterval_cos, METH_NOARGS,
   "Return the cosine of the interval"},
  {"tan", pyinterval_tan, METH_NOARGS,
   "Return the tangent of the interval"},
  {"arctan", pyinterval_arctan, METH_NOARGS,
   "Return the inverse tangent of the interval"},
  {"tanh", pyinterval_tanh, METH_NOARGS,
   "Return the hyperbolic tangent of the interval"},
  {"exp", pyinterval_exp, METH_NOARGS,
   "Return the exponential of the interval"},
  {"sqrt", pyinterval_sqrt, METH_NOARGS,
   "Return the sqrt of the interval"},
  {"square", pyinterval_square, METH_NOARGS,
   "Return the square of the interval"},
  // Binary interval returners
  {"union", pyinterval_union, METH_O,
   "Return the union of two intervals"},
  {"intersection", pyinterval_intersection, METH_O,
   "Return the intersection of two intervals"},
  {"minimum", pyinterval_minimum, METH_O,
   "Return the minimum of two intervals"},
  {"maximum", pyinterval_maximum, METH_O,
   "Return the maximum of two intervals"},
  {"__reduce__", (PyCFunction)pyinterval__reduce, METH_NOARGS,
   "Return state information for pickling."},
  {"__getstate__", (PyCFunction)pyinterval_getstate, METH_VARARGS,
   "Return state information for pickling."},
  {"__setstate__", (PyCFunction)pyinterval_setstate, METH_VARARGS,
   "Reconstruct state information from pickle."},

  {NULL, NULL, 0, NULL}
};

static PyObject* pyinterval_num_power(PyObject* a, PyObject* b, PyObject *c) { (void) c; return pyinterval_power(a,b); }
static PyObject* pyinterval_num_inplace_power(PyObject* a, PyObject* b, PyObject *c) { (void) c; return pyinterval_inplace_power(a,b); }
static PyObject* pyinterval_num_negative(PyObject* a) { return pyinterval_negative(a,NULL); }
static PyObject* pyinterval_num_positive(PyObject* a) { return pyinterval_positive(a,NULL); }
static PyObject* pyinterval_num_absolute(PyObject* a) { return pyinterval_abs(a, NULL); }
static PyObject* pyinterval_num_inverse(PyObject* a) { return pyinterval_inverse(a,NULL); }
static int pyinterval_num_nonzero(PyObject* a) {
  interval q = ((PyInterval*)a)->obval;
  return interval_nonzero(q);
}
#define CANNOT_CONVERT(target)                                          \
  static PyObject* pyinterval_convert_##target(PyObject* a) {         \
    PyErr_SetString(PyExc_TypeError, "Cannot convert interval to " #target); \
    return NULL;                                                        \
  }
CANNOT_CONVERT(int)
CANNOT_CONVERT(float)


static PyNumberMethods pyinterval_as_number = {
  pyinterval_add,               // nb_add
  pyinterval_subtract,          // nb_subtract
  pyinterval_multiply,          // nb_multiply
  0,                              // nb_remainder
  0,                              // nb_divmod
  pyinterval_num_power,         // nb_power
  pyinterval_num_negative,      // nb_negative
  pyinterval_num_positive,      // nb_positive
  pyinterval_num_absolute,      // nb_absolute
  pyinterval_num_nonzero,       // nb_nonzero
  pyinterval_num_inverse,       // nb_invert
  0,                              // nb_lshift
  0,                              // nb_rshift
  0,                              // nb_and
  0,                              // nb_xor
  0,                              // nb_or
  pyinterval_convert_int,       // nb_int
  0,                              // nb_reserved
  pyinterval_convert_float,     // nb_float
  pyinterval_inplace_add,       // nb_inplace_add
  pyinterval_inplace_subtract,  // nb_inplace_subtract
  pyinterval_inplace_multiply,  // nb_inplace_multiply
  0,                              // nb_inplace_remainder
  pyinterval_num_inplace_power, // nb_inplace_power
  0,                              // nb_inplace_lshift
  0,                              // nb_inplace_rshift
  0,                              // nb_inplace_and
  0,                              // nb_inplace_xor
  0,                              // nb_inplace_or
  pyinterval_divide,            // nb_floor_divide
  pyinterval_divide,            // nb_true_divide
  pyinterval_inplace_divide,    // nb_inplace_floor_divide
  pyinterval_inplace_divide,    // nb_inplace_true_divide
  0,                              // nb_index
  0,                              // nb_matrix_multiply
  0,                              // nb_inplace_matrix_multiply
};

PyMemberDef pyinterval_members[] = {
  {"l", T_DOUBLE, offsetof(PyInterval, obval.l), 0,
   "The lower bound of the interval"},
  {"u", T_DOUBLE, offsetof(PyInterval, obval.u), 0,
   "The upper bound of the interval"},
  {NULL, 0, 0, 0, NULL}
};

static PyObject *
pyinterval_get_vec(PyObject *self, void *NPY_UNUSED(closure))
{
  interval *q = &((PyInterval *)self)->obval;
  int nd = 1;
  npy_intp dims[1] = { 2 };
  int typenum = NPY_DOUBLE;
  PyObject* components = PyArray_SimpleNewFromData(nd, dims, typenum, &(q->l));
  Py_INCREF(self);
  PyArray_SetBaseObject((PyArrayObject*)components, self);
  return components;
}

static int
pyinterval_set_vec(PyObject *self, PyObject *value, void *NPY_UNUSED(closure))
{
  PyObject *element;
  interval *q = &((PyInterval *)self)->obval;
  if (value == NULL) {
    PyErr_SetString(PyExc_TypeError, "Cannot set interval to empty value");
    return -1;
  }
  if (! (PySequence_Check(value) && PySequence_Size(value)==2) ) {
    PyErr_SetString(PyExc_TypeError,
                    "A interval's vector components must be set to something of length 2");
    return -1;
  }
  element = PySequence_GetItem(value, 0);
  if(element == NULL) { return -1; }
  q->l = PyFloat_AsDouble(element);
  Py_DECREF(element);
  element = PySequence_GetItem(value, 1);
  if(element == NULL) { return -1; }
  q->u = PyFloat_AsDouble(element);
  Py_DECREF(element);
  return 0;
}

PyGetSetDef pyinterval_getset[] = {
  {"vec", pyinterval_get_vec, pyinterval_set_vec,
   "The vector part (l,u) of the interval as a numpy array", NULL},
  {NULL, NULL, NULL, NULL, NULL}
};


static long
pyinterval_hash(PyObject *o)
{
  interval q = ((PyInterval *)o)->obval;
  long value = 0x456789;
  value = (10000004 * value) ^ _Py_HashDouble(o, q.l);
  value = (10000004 * value) ^ _Py_HashDouble(o, q.u);
  if (value == -1)
    value = -2;
  return value;
}

static PyObject *
pyinterval_repr(PyObject *o)
{
  char str[128];
  interval q = ((PyInterval *)o)->obval;
  sprintf(str, "([%.4g, %.4g])", q.l, q.u);
  return PyUnicode_FromString(str);
}

static PyObject *
pyinterval_str(PyObject *o)
{
  char str[128];
  interval q = ((PyInterval *)o)->obval;
  sprintf(str, "([%.4g, %.4g])", q.l, q.u);
  return PyUnicode_FromString(str);
}

static PyTypeObject PyInterval_Type = {
  PyVarObject_HEAD_INIT(NULL, 0)
  "interval.interval",                        // tp_name
  sizeof(PyInterval),                         // tp_basicsize
  0,                                          // tp_itemsize
  0,                                          // tp_dealloc
  0,                                          // tp_vectorcall_offset
  0,                                          // tp_getattr
  0,                                          // tp_setattr
  0,                                          // tp_as_async
  pyinterval_repr,                            // tp_repr
  &pyinterval_as_number,                      // tp_as_number
  0,                                          // tp_as_sequence
  0,                                          // tp_as_mapping
  pyinterval_hash,                            // tp_hash
  0,                                          // tp_call
  pyinterval_str,                             // tp_str
  0,                                          // tp_getattro
  0,                                          // tp_setattro
  0,                                          // tp_as_buffer
  Py_TPFLAGS_DEFAULT | Py_TPFLAGS_BASETYPE,   // tp_flags
  "Floating-point interval numbers",          // tp_doc
  0,                                          // tp_traverse
  0,                                          // tp_clear
  0,                                          // tp_richcompare
  0,                                          // tp_weaklistoffset
  0,                                          // tp_iter
  0,                                          // tp_iternext
  pyinterval_methods,                         // tp_methods
  pyinterval_members,                         // tp_members
  pyinterval_getset,                          // tp_getset
  0,                                          // tp_base; will be reset to &PyGenericArrType_Type after numpy import
  0,                                          // tp_dict
  0,                                          // tp_descr_get
  0,                                          // tp_descr_set
  0,                                          // tp_dictoffset
  pyinterval_init,                            // tp_init
  0,                                          // tp_alloc
  pyinterval_new,                             // tp_new
  0,                                          // tp_free
  0,                                          // tp_is_gc
  0,                                          // tp_bases
  0,                                          // tp_mro
  0,                                          // tp_cache
  0,                                          // tp_subclasses
  0,                                          // tp_weaklist
  0,                                          // tp_del
  0,                                          // tp_version_tag
  0,                                          // tp_finalize
};

// Functions implementing internal features. Not all of these function
// pointers must be defined for a given type. The required members are
// nonzero, copyswap, copyswapn, setitem, getitem, and cast.
static PyArray_ArrFuncs _PyInterval_ArrFuncs;

static npy_bool
INTERVAL_nonzero (char *ip, PyArrayObject *ap)
{
  interval q;
  if (ap == NULL || PyArray_ISBEHAVED_RO(ap)) {
    q = *(interval *)ip;
  }
  else {
    PyArray_Descr *descr;
    descr = PyArray_DescrFromType(NPY_DOUBLE);
    descr->f->copyswap(&q.l, ip, !PyArray_ISNOTSWAPPED(ap), NULL);
    descr->f->copyswap(&q.u, ip+8, !PyArray_ISNOTSWAPPED(ap), NULL);
    Py_DECREF(descr);
  }
  return (npy_bool) interval_nonzero(q);
}

static void
INTERVAL_copyswap(interval *dst, interval *src,
                    int swap, void *NPY_UNUSED(arr))
{
  PyArray_Descr *descr;
  descr = PyArray_DescrFromType(NPY_DOUBLE);
  descr->f->copyswapn(dst, sizeof(double), src, sizeof(double), 2, swap, NULL);
  Py_DECREF(descr);
}

static void
INTERVAL_copyswapn(interval *dst, npy_intp dstride,
                     interval *src, npy_intp sstride,
                     npy_intp n, int swap, void *NPY_UNUSED(arr))
{
  PyArray_Descr *descr;
  descr = PyArray_DescrFromType(NPY_DOUBLE);
  descr->f->copyswapn(&dst->l, dstride, &src->l, sstride, n, swap, NULL);
  descr->f->copyswapn(&dst->u, dstride, &src->u, sstride, n, swap, NULL);
  Py_DECREF(descr);
}

static int INTERVAL_setitem(PyObject* item, interval* qp, void* NPY_UNUSED(ap))
{
  PyObject *element;
  if(PyInterval_Check(item)) {
    memcpy(qp,&(((PyInterval *)item)->obval),sizeof(interval));
  } else if(PySequence_Check(item) && PySequence_Length(item)==2) {
    element = PySequence_GetItem(item, 0);
    if(element == NULL) { return -1; }
    qp->l = PyFloat_AsDouble(element);
    Py_DECREF(element);
    element = PySequence_GetItem(item, 1);
    if(element == NULL) { return -1; }
    qp->u = PyFloat_AsDouble(element);
    Py_DECREF(element);
  } else if(PyFloat_Check(item)) {
    qp->l = PyFloat_AS_DOUBLE(item);
    qp->u = qp->l;
  } else if(PyLong_Check(item)) {
    qp->l = PyLong_AsDouble(item);
    qp->u = qp->l;
  } else {
    PyErr_SetString(PyExc_TypeError,
                    "Unknown input to INTERVAL_setitem");
    return -1;
  }
  return 0;
}


static PyObject *
INTERVAL_getitem(void* data, void* NPY_UNUSED(arr))
{
  interval q;
  memcpy(&q,data,sizeof(interval));
  return PyInterval_FromInterval(q);
}

static void
INTERVAL_fillwithscalar(interval *buffer, npy_intp length, interval *value, void *NPY_UNUSED(ignored))
{
  npy_intp i;
  interval val = *value;

  for (i = 0; i < length; ++i) {
    buffer[i] = val;
  }
}

static void
INTERVAL_dot(void* ip0_, npy_intp is0, void* ip1_, npy_intp is1,
        void* op, npy_intp n, void* NPY_UNUSED(arr)) {
    interval r = {0};
    const char *ip0 = (char*)ip0_, *ip1 = (char*)ip1_;
    npy_intp i;
    for (i = 0; i < n; i++) {
        r = interval_add(r,interval_multiply(*(interval*)ip0,*(interval*)ip1));
        ip0 += is0;
        ip1 += is1;
    }
    *(interval*)op = r;
}

// This is a macro (followed by applications of the macro) that cast
// the input types to standard intervals with only a nonzero scalar
// part.
#define MAKE_T_TO_INTERVAL(TYPE, type)                                \
  static void                                                           \
  TYPE ## _to_interval(type *ip, interval *op, npy_intp n,          \
                         PyArrayObject *NPY_UNUSED(aip), PyArrayObject *NPY_UNUSED(aop)) \
  {                                                                     \
    while (n--) {                                                       \
      double d = (double) *ip++; \
      op->l = d;                                          \
      op->u = d;                                                        \
      op++;                                                             \
    }                                                                   \
  }
MAKE_T_TO_INTERVAL(FLOAT, npy_float);
MAKE_T_TO_INTERVAL(DOUBLE, npy_double);
MAKE_T_TO_INTERVAL(LONGDOUBLE, npy_longdouble);
MAKE_T_TO_INTERVAL(BOOL, npy_bool);
MAKE_T_TO_INTERVAL(BYTE, npy_byte);
MAKE_T_TO_INTERVAL(UBYTE, npy_ubyte);
MAKE_T_TO_INTERVAL(SHORT, npy_short);
MAKE_T_TO_INTERVAL(USHORT, npy_ushort);
MAKE_T_TO_INTERVAL(INT, npy_int);
MAKE_T_TO_INTERVAL(UINT, npy_uint);
MAKE_T_TO_INTERVAL(LONG, npy_long);
MAKE_T_TO_INTERVAL(ULONG, npy_ulong);
MAKE_T_TO_INTERVAL(LONGLONG, npy_longlong);
MAKE_T_TO_INTERVAL(ULONGLONG, npy_ulonglong);

static void register_cast_function(int sourceType, int destType, PyArray_VectorUnaryFunc *castfunc)
{
  PyArray_Descr *descr = PyArray_DescrFromType(sourceType);
  PyArray_RegisterCastFunc(descr, destType, castfunc);
  PyArray_RegisterCanCast(descr, destType, NPY_NOSCALAR);
  Py_DECREF(descr);
}


// Macro for basic unary interval ufuncs applied to numpy arrays of intervals.
#define UNARY_GEN_UFUNC(ufunc_name, func_name, ret_type)        \
  static void                                                           \
  interval_##ufunc_name##_ufunc(char** args, const npy_intp* dimensions,    \
                                  const npy_intp* steps, void* NPY_UNUSED(data)) { \
    char *ip1 = args[0], *op1 = args[1];                                \
    npy_intp is1 = steps[0], os1 = steps[1];                            \
    npy_intp n = dimensions[0];                                         \
    npy_intp i;                                                         \
    for(i = 0; i < n; i++, ip1 += is1, op1 += os1){                     \
      const interval in1 = *(interval *)ip1;                        \
      *((ret_type *)op1) = interval_##func_name(in1);};}
#define UNARY_UFUNC(name, ret_type) \
  UNARY_GEN_UFUNC(name, name, ret_type)
UNARY_UFUNC(norm, npy_double)
UNARY_UFUNC(sin, interval)
UNARY_UFUNC(cos, interval)
UNARY_UFUNC(tan, interval)
UNARY_UFUNC(arctan, interval)
UNARY_UFUNC(tanh, interval)
UNARY_UFUNC(exp, interval)
UNARY_UFUNC(sqrt, interval)
UNARY_UFUNC(square, interval)
UNARY_UFUNC(negative, interval)
static void
interval_positive_ufunc(char** args, const npy_intp* dimensions, const npy_intp* steps, void* NPY_UNUSED(data)) {
  char *ip1 = args[0], *op1 = args[1];
  npy_intp is1 = steps[0], os1 = steps[1];
  npy_intp n = dimensions[0];
  npy_intp i;
  for(i = 0; i < n; i++, ip1 += is1, op1 += os1) {
    const interval in1 = *(interval *)ip1;
    *((interval *)op1) = in1;
  }
}

// Macro for basic binary interval ufuncs applied to numpy arrays of intervals.
#define BINARY_GEN_UFUNC(ufunc_name, func_name, arg_type1, arg_type2, ret_type) \
  static void                                                           \
  interval_##ufunc_name##_ufunc(char** args, const npy_intp* dimensions,    \
                                  const npy_intp* steps, void* NPY_UNUSED(data)) { \
    char *ip1 = args[0], *ip2 = args[1], *op1 = args[2];                \
    npy_intp is1 = steps[0], is2 = steps[1], os1 = steps[2];            \
    npy_intp n = dimensions[0];                                         \
    npy_intp i;                                                         \
    for(i = 0; i < n; i++, ip1 += is1, ip2 += is2, op1 += os1) {        \
      const arg_type1 in1 = *(arg_type1 *)ip1;                          \
      const arg_type2 in2 = *(arg_type2 *)ip2;                          \
      *((ret_type *)op1) = interval_##func_name(in1, in2);            \
    };                                                                  \
  };
#define BINARY_UFUNC(name, ret_type)                    \
  BINARY_GEN_UFUNC(name, name, interval, interval, ret_type)
#define BINARY_SCALAR_UFUNC(name, ret_type)                             \
  BINARY_GEN_UFUNC(name##_scalar, name##_scalar, interval, npy_double, ret_type) \
  BINARY_GEN_UFUNC(scalar_##name, scalar_##name, npy_double, interval, ret_type)
BINARY_GEN_UFUNC(power_scalar, power_scalar, interval, npy_double, interval)
BINARY_UFUNC(add, interval)
BINARY_UFUNC(subtract, interval)
BINARY_UFUNC(multiply, interval)
BINARY_UFUNC(divide, interval)
BINARY_GEN_UFUNC(true_divide, divide, interval, interval, interval)
BINARY_GEN_UFUNC(floor_divide, divide, interval, interval, interval)
BINARY_UFUNC(equal, npy_bool)
BINARY_UFUNC(not_equal, npy_bool)
BINARY_UFUNC(subseteq, npy_bool)
BINARY_UFUNC(supseteq, npy_bool)
BINARY_UFUNC(subset, npy_bool)
BINARY_UFUNC(supset, npy_bool)
BINARY_SCALAR_UFUNC(add, interval)
BINARY_SCALAR_UFUNC(subtract, interval)
BINARY_SCALAR_UFUNC(multiply, interval)
BINARY_SCALAR_UFUNC(divide, interval)
BINARY_GEN_UFUNC(true_divide_scalar, divide_scalar, interval, npy_double, interval)
BINARY_GEN_UFUNC(floor_divide_scalar, divide_scalar, interval, npy_double, interval)
BINARY_GEN_UFUNC(scalar_true_divide, scalar_divide, npy_double, interval, interval)
BINARY_GEN_UFUNC(scalar_floor_divide, scalar_divide, npy_double, interval, interval)
BINARY_UFUNC(union, interval)
BINARY_UFUNC(intersection, interval)
BINARY_UFUNC(maximum, interval)
BINARY_UFUNC(minimum, interval)

static NPY_INLINE void
interval_matmul(char **args, const npy_intp *dimensions, const npy_intp *steps)
{
    char *ip1 = args[0];
    char *ip2 = args[1];
    char *op = args[2];

    npy_intp dm = dimensions[0];
    npy_intp dn = dimensions[1];
    npy_intp dp = dimensions[2];

    npy_intp is1_m = steps[0];
    npy_intp is1_n = steps[1];
    npy_intp is2_n = steps[2];
    npy_intp is2_p = steps[3];
    npy_intp os_m = steps[4];
    npy_intp os_p = steps[5];

    npy_intp m, p;

    for (m = 0; m < dm; m++) {
        for (p = 0; p < dp; p++) {
            INTERVAL_dot(ip1, is1_n, ip2, is2_n, op, dn, NULL);

            ip2 += is2_p;
            op  +=  os_p;
        }

        ip2 -= is2_p * p;
        op -= os_p * p;

        ip1 += is1_m;
        op += os_m;
    }
}

static void
interval_matmul_ufunc(char **args, const npy_intp *dimensions, const npy_intp *steps, void *NPY_UNUSED(func))
{
    npy_intp N_;
    npy_intp dN = dimensions[0];
    npy_intp s0 = steps[0];
    npy_intp s1 = steps[1];
    npy_intp s2 = steps[2];

    for (N_ = 0; N_ < dN; N_++, args[0] += s0, args[1] += s1, args[2] += s2) {
        interval_matmul(args, dimensions+1, steps+3);
    }
}

// Assorted other top-level methods for the module
static PyMethodDef IntervalMethods[] = {
  {NULL, NULL, 0, NULL}
};


int interval_elsize = sizeof(interval);

typedef struct { char c; interval q; } align_test;
int interval_alignment = offsetof(align_test, q);


/////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////
//                                                             //
//  Everything above was preparation for the following set up  //
//                                                             //
/////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////

static struct PyModuleDef moduledef = {
    PyModuleDef_HEAD_INIT,
    "npinterval",
    NULL,
    -1,
    IntervalMethods,
    NULL,
    NULL,
    NULL,
    NULL
};

PyMODINIT_FUNC PyInit_numpy_interval(void) {

  PyObject *module;
  PyObject *tmp_ufunc;
  int intervalNum;
  int arg_types[3];
  PyObject* numpy;
  PyObject* numpy_dict;

  module = PyModule_Create(&moduledef);

  if(module==NULL) {
    return NULL;
  }

  import_array();
  if (PyErr_Occurred()) {
    return NULL;
  }
  import_umath();
  if (PyErr_Occurred()) {
    return NULL;
  }
  numpy = PyImport_ImportModule("numpy");
  if (!numpy) {
    return NULL;
  }
  numpy_dict = PyModule_GetDict(numpy);
  if (!numpy_dict) {
    return NULL;
  }

  PyInterval_Type.tp_base = &PyGenericArrType_Type;
  if (PyType_Ready(&PyInterval_Type) < 0) {
    PyErr_Print();
    PyErr_SetString(PyExc_SystemError, "Could not initialize PyInterval_Type.");
    return NULL;
  }

  PyArray_InitArrFuncs(&_PyInterval_ArrFuncs);
  _PyInterval_ArrFuncs.nonzero = (PyArray_NonzeroFunc*)INTERVAL_nonzero;
  _PyInterval_ArrFuncs.copyswap = (PyArray_CopySwapFunc*)INTERVAL_copyswap;
  _PyInterval_ArrFuncs.copyswapn = (PyArray_CopySwapNFunc*)INTERVAL_copyswapn;
  _PyInterval_ArrFuncs.setitem = (PyArray_SetItemFunc*)INTERVAL_setitem;
  _PyInterval_ArrFuncs.getitem = (PyArray_GetItemFunc*)INTERVAL_getitem;
  _PyInterval_ArrFuncs.dotfunc = (PyArray_DotFunc*)INTERVAL_dot;
  _PyInterval_ArrFuncs.fillwithscalar = (PyArray_FillWithScalarFunc*)INTERVAL_fillwithscalar;

  interval_descr = PyObject_New(PyArray_Descr, &PyArrayDescr_Type);
  interval_descr->typeobj = &PyInterval_Type;
  interval_descr->kind = 'V';
  interval_descr->type = 'i';
  interval_descr->byteorder = '=';
  interval_descr->flags = NPY_NEEDS_PYAPI | NPY_USE_GETITEM | NPY_USE_SETITEM;
  interval_descr->type_num = 0; // assigned at registration
  interval_descr->elsize = interval_elsize;
  interval_descr->alignment = interval_alignment;
  interval_descr->subarray = NULL;
  interval_descr->fields = NULL;
  interval_descr->names = NULL;
  interval_descr->f = &_PyInterval_ArrFuncs;
  interval_descr->metadata = NULL;
  interval_descr->c_metadata = NULL;

  Py_INCREF(&PyInterval_Type);
  intervalNum = PyArray_RegisterDataType(interval_descr);

  if (intervalNum < 0) {
    return NULL;
  }

  register_cast_function(NPY_BOOL, intervalNum, (PyArray_VectorUnaryFunc*)BOOL_to_interval);
  register_cast_function(NPY_BYTE, intervalNum, (PyArray_VectorUnaryFunc*)BYTE_to_interval);
  register_cast_function(NPY_UBYTE, intervalNum, (PyArray_VectorUnaryFunc*)UBYTE_to_interval);
  register_cast_function(NPY_SHORT, intervalNum, (PyArray_VectorUnaryFunc*)SHORT_to_interval);
  register_cast_function(NPY_USHORT, intervalNum, (PyArray_VectorUnaryFunc*)USHORT_to_interval);
  register_cast_function(NPY_INT, intervalNum, (PyArray_VectorUnaryFunc*)INT_to_interval);
  register_cast_function(NPY_UINT, intervalNum, (PyArray_VectorUnaryFunc*)UINT_to_interval);
  register_cast_function(NPY_LONG, intervalNum, (PyArray_VectorUnaryFunc*)LONG_to_interval);
  register_cast_function(NPY_ULONG, intervalNum, (PyArray_VectorUnaryFunc*)ULONG_to_interval);
  register_cast_function(NPY_LONGLONG, intervalNum, (PyArray_VectorUnaryFunc*)LONGLONG_to_interval);
  register_cast_function(NPY_ULONGLONG, intervalNum, (PyArray_VectorUnaryFunc*)ULONGLONG_to_interval);
  register_cast_function(NPY_FLOAT, intervalNum, (PyArray_VectorUnaryFunc*)FLOAT_to_interval);
  register_cast_function(NPY_DOUBLE, intervalNum, (PyArray_VectorUnaryFunc*)DOUBLE_to_interval);
  register_cast_function(NPY_LONGDOUBLE, intervalNum, (PyArray_VectorUnaryFunc*)LONGDOUBLE_to_interval);

  #define REGISTER_UFUNC(name)                                          \
    PyUFunc_RegisterLoopForType((PyUFuncObject *)PyDict_GetItemString(numpy_dict, #name), \
                                interval_descr->type_num, interval_##name##_ufunc, arg_types, NULL)
  #define REGISTER_SCALAR_UFUNC(name)                                   \
    PyUFunc_RegisterLoopForType((PyUFuncObject *)PyDict_GetItemString(numpy_dict, #name), \
                                interval_descr->type_num, interval_scalar_##name##_ufunc, arg_types, NULL)
  #define REGISTER_UFUNC_SCALAR(name)                                   \
    PyUFunc_RegisterLoopForType((PyUFuncObject *)PyDict_GetItemString(numpy_dict, #name), \
                                interval_descr->type_num, interval_##name##_scalar_ufunc, arg_types, NULL)
  #define REGISTER_NEW_UFUNC_GENERAL(pyname, cname, nargin, nargout, doc) \
    tmp_ufunc = PyUFunc_FromFuncAndData(NULL, NULL, NULL, 0, nargin, nargout, \
                                        PyUFunc_None, #pyname, doc, 0); \
    PyUFunc_RegisterLoopForType((PyUFuncObject *)tmp_ufunc,             \
                                interval_descr->type_num, interval_##cname##_ufunc, arg_types, NULL); \
    PyDict_SetItemString(numpy_dict, #pyname, tmp_ufunc);               \
    Py_DECREF(tmp_ufunc)
  #define REGISTER_NEW_UFUNC(name, nargin, nargout, doc)                \
    REGISTER_NEW_UFUNC_GENERAL(name, name, nargin, nargout, doc)

  // interval -> double
  arg_types[0] = interval_descr->type_num;
  arg_types[1] = NPY_DOUBLE;
  REGISTER_NEW_UFUNC(norm, 1, 1,
                     "Return the measure of each interval.\n");

  // interval -> interval
  arg_types[0] = interval_descr->type_num;
  arg_types[1] = interval_descr->type_num;
  REGISTER_UFUNC(sin);
  REGISTER_UFUNC(cos);
  REGISTER_UFUNC(tan);
  REGISTER_UFUNC(arctan);
  REGISTER_UFUNC(tanh);
  REGISTER_UFUNC(exp);
  REGISTER_UFUNC(sqrt);
  REGISTER_UFUNC(square);
  REGISTER_UFUNC(negative);
  REGISTER_UFUNC(positive);

  // interval, interval -> bool
  arg_types[0] = interval_descr->type_num;
  arg_types[1] = interval_descr->type_num;
  arg_types[2] = NPY_BOOL;
  REGISTER_UFUNC(equal);
  REGISTER_UFUNC(not_equal);
  REGISTER_NEW_UFUNC(subseteq, 2, 1,
                     "Return true if i1 is a subset (inclusive) of i2");
  REGISTER_NEW_UFUNC(supseteq, 2, 1,
                     "Return true if i1 is a superset (inclusive) of i2");
  REGISTER_NEW_UFUNC(subset, 2, 1,
                     "Return true if i1 is a subset (strict) of i2");
  REGISTER_NEW_UFUNC(supset, 2, 1,
                     "Return true if i1 is a superset (strict) of i2");

  // interval, interval -> interval
  arg_types[0] = interval_descr->type_num;
  arg_types[1] = interval_descr->type_num;
  arg_types[2] = interval_descr->type_num;
  REGISTER_UFUNC(add);
  REGISTER_UFUNC(subtract);
  REGISTER_UFUNC(multiply);
  REGISTER_UFUNC(divide);
  REGISTER_UFUNC(true_divide);
  REGISTER_UFUNC(floor_divide);
  REGISTER_UFUNC(matmul);
  REGISTER_UFUNC(maximum);
  REGISTER_UFUNC(minimum);
  REGISTER_NEW_UFUNC(union, 2, 1,
                     "Return the union of intervals");
  REGISTER_NEW_UFUNC(intersection, 2, 1,
                     "Return the intersection of intervals");

  // double, interval -> interval
  arg_types[0] = NPY_DOUBLE;
  arg_types[1] = interval_descr->type_num;
  arg_types[2] = interval_descr->type_num;
  REGISTER_SCALAR_UFUNC(add);
  REGISTER_SCALAR_UFUNC(subtract);
  REGISTER_SCALAR_UFUNC(multiply);
  REGISTER_SCALAR_UFUNC(divide);
  REGISTER_SCALAR_UFUNC(true_divide);
  REGISTER_SCALAR_UFUNC(floor_divide);

  // interval, double -> interval
  arg_types[0] = interval_descr->type_num;
  arg_types[1] = NPY_DOUBLE;
  arg_types[2] = interval_descr->type_num;
  REGISTER_UFUNC_SCALAR(add);
  REGISTER_UFUNC_SCALAR(subtract);
  REGISTER_UFUNC_SCALAR(multiply);
  REGISTER_UFUNC_SCALAR(divide);
  REGISTER_UFUNC_SCALAR(power);
  REGISTER_UFUNC_SCALAR(true_divide);
  REGISTER_UFUNC_SCALAR(floor_divide);

  PyModule_AddObject(module, "interval", (PyObject *)&PyInterval_Type);

  return module;
}
