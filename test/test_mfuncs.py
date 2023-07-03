import numpy
import pytest
import scipy.linalg
import scipy.sparse
from numpy.testing import (
    assert_array_almost_equal,
    assert_array_equal,
)
import krypy


def square_func(x):
    return x**2


def exp_func(x):
    return numpy.exp(x)


def inv_func(x):
    return 1/x


def identity_func(x):
    return x


def get_diagonal_mat(n):
    A = numpy.diag(
        numpy.linspace(0.1, 1, n)
        )
    return A


def get_tridiagonal_mat(n):
    A = numpy.ones((n, n))
    A = numpy.tril(A, 1) - numpy.tril(A, -2)
    A[0, 1] = 2
    return A


def get_hermitian_mat(n):
    numpy.random.seed(0)
    A = numpy.random.rand(n, n)
    A = A + A.conj().T
    A = A/2
    return A


def get_nonhermitian_mat(n):
    numpy.random.seed(0)
    A = numpy.random.rand(n, n)
    return A


def get_complex_nonhermitian_mat(n):
    A = get_nonhermitian_mat(n) + 1j * get_nonhermitian_mat(n)
    return A


def get_pos_semidef_herm_mat():
    A = get_diagonal_mat(5) - numpy.diag(0.095*numpy.ones(5))
    return A


def get_ones_vector(n):
    return numpy.ones((n, 1))


def get_random_vector(n):
    numpy.random.seed(0)
    return numpy.random.rand(n, 1)


def get_random_complex_vector(n):
    b = get_random_vector(n) + 1j * get_random_vector(n)
    return b


def convert_to_sparse(A):
    return scipy.sparse.csr_array(A)


@pytest.mark.parametrize("f, mat", [
    (square_func, get_diagonal_mat(5)),
    (identity_func, get_tridiagonal_mat(5)),
    (numpy.exp, get_hermitian_mat(5))
])
def test_mfunc_general(f, mat):
    from krypy.mfuncs import MatrixFunction
    mfunc = MatrixFunction(f)
    assert_array_equal(mfunc(mat), scipy.linalg.funm(mat, f))


@pytest.mark.parametrize("mat", [
    get_hermitian_mat(3),
    get_hermitian_mat(5),
    get_hermitian_mat(6)
])
def test_mfunc_hermitian(mat):
    from krypy.mfuncs import MatrixFunction
    mfunc = MatrixFunction(numpy.exp)
    assert_array_almost_equal(mfunc(mat, is_hermitian=True),
                              scipy.linalg.expm(mat))

@pytest.mark.parametrize("mat", [
    get_hermitian_mat(3) + numpy.eye(3),
    get_hermitian_mat(5) + numpy.eye(5),
    get_hermitian_mat(6) + numpy.eye(6),
    get_hermitian_mat(8) + numpy.eye(8),
    get_pos_semidef_herm_mat()
])
def test_mfunc_hermitian_pos_semidef(mat):
    from krypy.mfuncs import MatrixFunction
    mfunc = MatrixFunction(numpy.exp)
    assert_array_almost_equal(mfunc(mat,
                                    is_hermitian=True,
                                    is_pos_semidefinite=True),
                              scipy.linalg.expm(mat))
    
@pytest.mark.parametrize("mat", [
    get_diagonal_mat(5),
    get_hermitian_mat(5),
    get_nonhermitian_mat(5),
])
def test_expm_matfunc(mat):
    from krypy.mfuncs import MatrixExponential
    f = MatrixExponential()
    assert_array_equal(f(mat), scipy.linalg.expm(mat))

@pytest.mark.parametrize("mat", [
    get_diagonal_mat(5),
    get_hermitian_mat(5),
    get_nonhermitian_mat(5),
])
def test_inv_matfunc(mat):
    from krypy.mfuncs import MatrixInverse
    f = MatrixInverse()
    assert_array_equal(f(mat), scipy.linalg.inv(mat))

@pytest.mark.parametrize("mat", [
    get_diagonal_mat(5),
    get_hermitian_mat(5),
    get_nonhermitian_mat(5),
])
def test_expm_matfunc_sparse(mat):
    mat = scipy.sparse.csc_array(mat)
    from krypy.mfuncs import MatrixExponential
    f = MatrixExponential()
    assert_array_equal(f(mat).toarray(),
                       scipy.sparse.linalg.expm(mat).toarray())

@pytest.mark.parametrize("mat", [
    get_diagonal_mat(5),
    get_hermitian_mat(5),
    get_nonhermitian_mat(5),
])
def test_inv_matfunc_inv_sparse(mat):
    mat = scipy.sparse.csc_array(mat)
    from krypy.mfuncs import MatrixInverse
    f = MatrixInverse()
    assert_array_equal(f(mat).toarray(),
                       scipy.sparse.linalg.inv(mat).toarray())


_test_inputs = [
    (get_diagonal_mat(5),
     krypy.mfuncs.MatrixExponential(),
     scipy.linalg.expm(get_diagonal_mat(5)),
     True,
     True,
     True),
    (get_diagonal_mat(5),
     krypy.mfuncs.MatrixExponential(),
     None,
     True,
     True,
     True),
    (get_diagonal_mat(5),
     krypy.mfuncs.MatrixExponential(),
     None,
     False,
     False,
     False),
    (get_diagonal_mat(5),
     krypy.mfuncs.MatrixExponential(),
     scipy.linalg.expm(get_diagonal_mat(5)),
     False,
     False,
     False),
    (get_nonhermitian_mat(10),
     krypy.mfuncs.MatrixExponential(),
     scipy.linalg.expm(get_nonhermitian_mat(10)),
     True,
     True,
     True),
    (get_diagonal_mat(5),
     krypy.mfuncs.MatrixExponential(),
     None,
     True,
     True,
     True),
    (get_nonhermitian_mat(10),
     krypy.mfuncs.MatrixExponential(),
     None,
     False,
     False,
     False),
    (get_nonhermitian_mat(10),
     krypy.mfuncs.MatrixExponential(),
     scipy.linalg.expm(get_nonhermitian_mat(10)),
     False,
     False,
     False)
]
@pytest.mark.parametrize("mat, mfunc, fA, normal, "
                         + "hermitian, pos_semidefinite", 
                         _test_inputs)
def test_matfunc_sys(mat, mfunc, fA,
                     normal, hermitian, pos_semidefinite):
    from krypy.mfuncs import MatrixFunctionSystem
    from krypy.mfuncs import MatrixFunction
    mfunc_system = MatrixFunctionSystem(mat, mfunc, fA,
                                        False,
                                        normal, hermitian,
                                        pos_semidefinite)
    assert numpy.all(mfunc_system.get_mat() == mat)
    assert isinstance(mfunc_system.get_mfunc(), MatrixFunction)
    assert_array_almost_equal(mfunc_system.get_computed_mfunc(),
                              mfunc(mat, hermitian, pos_semidefinite))


_hermitian_example_additive = [
    get_hermitian_mat(20),
    krypy.mfuncs.MatrixExponential(),
    None,
    numpy.ones((20, 1))/20,
    False,
    True,
    {"c": None,
     "sign": 1,
     "maxiter": 17,
     "tol": 1e-12,
     "d": 2,
     "ortho": "dmgs",
     "explicit_update": None,
     "dropping_tol": None,
     "store_res": True}
]
_hermitian_example_additive_large = [
    get_hermitian_mat(100)/10,
    krypy.mfuncs.MatrixExponential(),
    None,
    numpy.ones((100, 1))/100,
    False,
    True,
    {"c": None,
     "sign": 1,
     "maxiter": 30,
     "tol": 1e-12,
     "d": 2,
     "ortho": "dmgs",
     "explicit_update": None,
     "dropping_tol": None,
     "store_res": True}
]
_hermitian_example_subtractive = [
    get_hermitian_mat(20),
    krypy.mfuncs.MatrixExponential(),
    None,
    numpy.ones((20, 1))/numpy.sqrt(20),
    False,
    True,
    {"c": None,
     "sign": -1,
     "maxiter": 17,
     "tol": 1e-12,
     "d": 2,
     "ortho": "dmgs",
     "explicit_update": None,
     "dropping_tol": None,
     "store_res": True}
]
_nonhermitian_example_additive = [
    get_nonhermitian_mat(20)/10,
    krypy.mfuncs.MatrixExponential(),
    None,
    numpy.ones((20, 1))/numpy.sqrt(20),
    False,
    False,
    {"c": get_random_vector(20),
     "sign": 1,
     "maxiter": 17,
     "tol": 1e-12,
     "d": 2,
     "ortho": "dmgs",
     "explicit_update": None,
     "dropping_tol": None,
     "store_res": True}
]
_complex_example = [
    get_complex_nonhermitian_mat(20)/10,
    krypy.mfuncs.MatrixExponential(),
    None,
    get_random_complex_vector(20)/10,
    False,
    False,
    {"c": get_ones_vector(20)/10,
     "sign": 1,
     "maxiter": 17,
     "tol": 1e-12,
     "d": 2,
     "ortho": "dmgs",
     "explicit_update": None,
     "dropping_tol": None,
     "store_res": True}
]

@pytest.mark.parametrize("A, f, fA, b, return_object, hermitian, "
                         + "kwargs", 
                         [_hermitian_example_additive,
                          _hermitian_example_additive_large,
                          _hermitian_example_subtractive,
                          _nonhermitian_example_additive,
                          _complex_example])
def test_rank_one_update(A, f, fA, b, return_object, hermitian, kwargs):
    from krypy.mfuncs import rank_one_update
    c = kwargs.get("c", b)
    c = b if c is None else c
    sign = kwargs.get("sign", 1)
    assert scipy.linalg.norm(rank_one_update(A, f, fA, b, return_object, 
                                             hermitian, **kwargs)
                             - (f(A + sign * b @ c.conj().T) - f(A))) < 1e-10


_large_sparse_nonhermitian_example = [
    convert_to_sparse(get_tridiagonal_mat(500)/10),
    krypy.mfuncs.MatrixExponential(),
    None,
    numpy.ones((500, 1))/500,
    False,
    False,
    {"c": get_random_vector(500),
     "sign": 1,
     "maxiter": 100,
     "tol": 1e-12,
     "d": 2,
     "ortho": "dmgs",
     "explicit_update": None,
     "dropping_tol": None,
     "store_res": True}
]
@pytest.mark.parametrize("A, f, fA, b, return_object, hermitian, "
                         + "kwargs", 
                         [_large_sparse_nonhermitian_example])
def test_rank_one_update_sparse(A, f, fA, b, return_object, hermitian, kwargs):
    from krypy.mfuncs import rank_one_update
    c = kwargs.get("c", b)
    c = b if c is None else c
    sign = kwargs.get("sign", 1)
    assert (scipy.linalg.norm(rank_one_update(A, f, fA, b, return_object,
                                              hermitian, **kwargs).toarray()
                              - (f(A + sign * b @ c.conj().T) - f(A)))
            < 1e-10)


_A = get_diagonal_mat(800)/10
_A[5, 1] = 1
_large_sparse_timed = [
    convert_to_sparse(_A),
    krypy.mfuncs.MatrixExponential(),
    None,
    numpy.ones((800, 1)),
    True,
    False,
    {"c": get_random_vector(800),
     "sign": 1,
     "maxiter": 100,
     "tol": 1e-12,
     "d": 2,
     "ortho": "dmgs",
     "explicit_update": None,
     "dropping_tol": None,
     "store_res": False}
]


@pytest.mark.parametrize("A, f, fA, b, return_object, hermitian, "
                         + "kwargs", 
                         [_large_sparse_timed])
def test_rank_one_update_sparse_timed(A, f, fA, b, return_object, hermitian, kwargs):
    from krypy.mfuncs import rank_one_update
    from krypy.utils import Timer
    c = kwargs.get("c", b)
    c = b if c is None else c
    sign = kwargs.get("sign", 1)
    t1 = Timer()
    fA = f(A)
    for i in range(5):
        with t1:
            rank_one_update(A, f, fA, b, return_object, hermitian,
                            **kwargs)
    B = A.toarray()
    fB = f(B)
    t2 = Timer()
    for i in range(5):
        with t2:
            rank_one_update(B, f, fB, b, return_object, hermitian,
                            **kwargs)
    assert sum(t1) < sum(t2)
