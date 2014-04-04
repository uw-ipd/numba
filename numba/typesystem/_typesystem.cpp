#include "_pymodule.h"
#include "capsulethunk.h"
#include "typesystem.hpp"


const char PYCAP_TYPECONTEXT[] = "numba::TypeContext";
const char PYCAP_TYPE[] = "numba::Type";


extern "C" {

PyObject* new_typecontext(PyObject *self, PyObject *args);
void delete_typecontext(PyObject *obj);
PyObject* fill_machine_types(PyObject *self, PyObject *args);

PyObject* get_type(PyObject *self, PyObject *args);
PyObject* get_type_name(PyObject *self, PyObject *args);
PyObject* get_type_rank(PyObject *self, PyObject *args);
PyObject* coerce(PyObject *self, PyObject *args);
PyObject* cast(PyObject *self, PyObject *args);
PyObject* select_overload(PyObject *self, PyObject *args);
PyObject* select_best_overload(PyObject *self, PyObject *args);

static PyMethodDef ext_methods[] = {
#define declmethod(func) { #func , ( PyCFunction )func , METH_VARARGS , NULL }
    declmethod(new_typecontext),
    declmethod(fill_machine_types),
    declmethod(get_type),
    declmethod(get_type_name),
    declmethod(get_type_rank),
    declmethod(coerce),
    declmethod(cast),
    declmethod(select_overload),
    declmethod(select_best_overload),
    { NULL },
#undef declmethod
};

MOD_INIT(_typesystem) {
    PyObject *m;
    MOD_DEF(m, "_typesystem", "No docs", ext_methods)
    if (m == NULL)
        return MOD_ERROR_VAL;

    return MOD_SUCCESS_VAL(m);
}

}   // extern C

using namespace numba;

TypeContext* unwrap_typecontext(PyObject *tm);
Type* unwrap_type(PyObject *ctx);
PyObject* wrap(Type* type);


///// Implementation //////


PyObject* new_typecontext(PyObject *self, PyObject *args) {
    if(!PyArg_ParseTuple(args, "")) {
        return NULL;
    }
    TypeContext *ctx = new TypeContext;
    void *p_ctx = reinterpret_cast<void*>(ctx);
    return PyCapsule_New(p_ctx, PYCAP_TYPECONTEXT, &delete_typecontext);
}

void delete_typecontext(PyObject *obj) {
    delete unwrap_typecontext(obj);
}

PyObject* fill_machine_types(PyObject *self, PyObject *args) {
    PyObject *ctx;
    if (!PyArg_ParseTuple(args, "O", &ctx)) {
        return NULL;
    }
    fillMachineTypes(*unwrap_typecontext(ctx));
    Py_RETURN_NONE;
}

TypeContext* unwrap_typecontext(PyObject *ctx) {
    void* p = PyCapsule_GetPointer(ctx, PYCAP_TYPECONTEXT);
    return reinterpret_cast<TypeContext*>(p);
}

PyObject* wrap(Type* type) {
    return PyCapsule_New(reinterpret_cast<void*>(type), PYCAP_TYPE, NULL);
}


PyObject* get_type(PyObject *self, PyObject *args) {
    PyObject *ctx;
    char *name;
    if (!PyArg_ParseTuple(args, "Os", &ctx, &name)) {
        return NULL;
    }
    Type *type = unwrap_typecontext(ctx)->types.get(name);
    return PyCapsule_New(reinterpret_cast<void*>(type), PYCAP_TYPE, NULL);
}

Type* unwrap_type(PyObject *ctx) {
    void* p = PyCapsule_GetPointer(ctx, PYCAP_TYPE);
    return reinterpret_cast<Type*>(p);
}

PyObject* get_type_name(PyObject *self, PyObject *args) {
    PyObject *type;
    if (!PyArg_ParseTuple(args, "O", &type)) {
        return NULL;
    }
    return PyString_FromString(unwrap_type(type)->name.c_str());
}

PyObject* get_type_rank(PyObject *self, PyObject *args) {
    PyObject *ctx;
    PyObject *type;
    if (!PyArg_ParseTuple(args, "OO", &ctx, &type)) {
        return NULL;
    }
    int rank = unwrap_typecontext(ctx)->getRank(unwrap_type(type));
    return PyLong_FromLong(rank);
}

PyObject* coerce(PyObject *self, PyObject *args) {
    PyObject *ctx;
    PyObject *types; // sequence of PyCapsule<Type*>
    if (!PyArg_ParseTuple(args, "OO", &ctx, &types)) {
        return NULL;
    }

    Py_ssize_t N = PySequence_Size(types);
    if (N == 0) return NULL;

    // Unpack the types
    Type** typeset = new Type*[N];
    for (Py_ssize_t i = 0; i < N; ++i) {
        typeset[i] = unwrap_type(PySequence_Fast_GET_ITEM(types, i));
    }

    CoerceDescriptor cd = coerce(*unwrap_typecontext(ctx), typeset, N);

    // Clean up
    delete [] typeset;

    // Prepare coerce descriptor as Python tuple
    if (!cd.okay) {
        Py_RETURN_NONE;
    }

    PyObject *cotype = wrap(cd.type);
    PyObject *safe = (cd.safe ? Py_True : Py_False);
    Py_INCREF(safe);

    return PyTuple_Pack(2, cotype, safe);
}

PyObject* cast(PyObject *self, PyObject *args) {
    PyObject *ctx;
    PyObject *fromtype, *totype;
    if(!PyArg_ParseTuple(args, "OOO", &ctx, &fromtype, &totype)) {
        return NULL;
    }

    TypeContext &ctxref = *unwrap_typecontext(ctx);
    CastDescriptor cd = ctxref.cast(unwrap_type(fromtype),
                                    unwrap_type(totype));

    const std::string tccstr = explainCompatibility(cd.tcc);
    return Py_BuildValue("(si)", tccstr.c_str(), cd.distance);
}

PyObject* select_overload(PyObject *self, PyObject *args) {
    PyObject *ctx, *sigs, *vers;

    // sigs and vers are sequence of types
    if(!PyArg_ParseTuple(args, "OOO", &ctx, &sigs, &vers)) {
        return NULL;
    }

    // Assuming the number of arguments matches
    Py_ssize_t nargs = PySequence_Size(sigs);
    Py_ssize_t len_vers = PySequence_Size(vers);
    Py_ssize_t nvers = len_vers / nargs;

    Type **arr_sigs = new Type*[nargs];
    Type **arr_vers = new Type*[len_vers];
    int *arr_sels = new int[nvers];

    // Unwrap sigs
    for (Py_ssize_t i=0; i < nargs; ++i) {
        arr_sigs[i] = unwrap_type(PySequence_Fast_GET_ITEM(sigs, i));
    }

    // Unwrap vers
    for (Py_ssize_t i=0; i < len_vers; ++i) {
        arr_vers[i] = unwrap_type(PySequence_Fast_GET_ITEM(vers, i));
    }


    // Run overload resolution
    int selct = selectOverload(*unwrap_typecontext(ctx), arr_sigs, arr_vers,
                               arr_sels, nvers, nargs);

    delete [] arr_sigs;
    delete [] arr_vers;
    delete [] arr_sels;

    if (selct < 0) {
        PyErr_SetString(PyExc_RuntimeError, "numba::selectOverload returned "
                                            "negative value");
        return NULL;
    }

    // Prepare output
    PyObject *retval = PyTuple_New(selct);
    for (int i=0; i < selct; ++i) {
        PyTuple_SET_ITEM(retval, i, PyLong_FromLong(arr_sels[i]));
    }

    return retval;
}


PyObject* select_best_overload(PyObject *self, PyObject *args) {
    PyObject *ctx, *sigs, *vers;

    // sigs and vers are sequence of types
    if(!PyArg_ParseTuple(args, "OOO", &ctx, &sigs, &vers)) {
        return NULL;
    }

    // Assuming the number of arguments matches
    Py_ssize_t nargs = PySequence_Size(sigs);
    Py_ssize_t len_vers = PySequence_Size(vers);
    Py_ssize_t nvers = len_vers / nargs;

    Type **arr_sigs = new Type*[nargs];
    Type **arr_vers = new Type*[len_vers];

    // Unwrap sigs
    for (Py_ssize_t i=0; i < nargs; ++i) {
        arr_sigs[i] = unwrap_type(PySequence_Fast_GET_ITEM(sigs, i));
    }

    // Unwrap vers
    for (Py_ssize_t i=0; i < len_vers; ++i) {
        arr_vers[i] = unwrap_type(PySequence_Fast_GET_ITEM(vers, i));
    }

    // Run overload resolution
    int selct = selectBestOverload(*unwrap_typecontext(ctx), arr_sigs,
                                   arr_vers, nvers, nargs);

    delete [] arr_sigs;
    delete [] arr_vers;

    // Prepare output
    if (selct < 0) {
        Py_RETURN_NONE;
    }
    return PyLong_FromLong(selct);
}
