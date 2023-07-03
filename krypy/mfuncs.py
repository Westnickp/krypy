from .utils import (Arnoldi, ArgumentError,
                    arnoldi_res)
import numpy
import scipy.linalg
import scipy.sparse
from collections import deque


__all__ = [
    "MatrixFunction",
    "MatrixExponential",
    "MatrixInverse",
    "MatrixFunctionSystem",
    "RankOneUpdate",
    "rank_one_update"
]


class MatrixFunction:
    """Class wrapper for scipy matrix function methods.

    Attributes
    ----------
    f : function
        A scalar function to be applied to a matrix.
    f_description : str, optional
        A brief description of the function for display purposes.
    implementation : str, optional
        A keyword indicating the implementation to be used.
        Currently, only the 'scipy' implementation, which
        wraps the scipy.linalg matrix function methods,
        is in use.
    dropping_tol : float, optional
        A float-value smaller than which all entries
        should be dropped. This is currently unused,
        but is intended for future work where it may be
        interesting to sparsify the matrix when most of
        its entries are near zero.

    Examples
    --------
    >>> f = MatrixFunction(lambda x: x**2)
    >>> A = numpy.array([[1, 0], [0, 2]])
    >>> print(f(A))
    [[1 0]
     [0 4]]
    """
    def __init__(
            self,
            f,
            f_description=None,
            implementation="scipy",
            dropping_tol=1e-14
    ):
        self.f = numpy.vectorize(f)
        self.f_description = f_description
        self.implementation = implementation

    def _evaluate_general(self, A):
        """Wraps scipy matrix function method."""
        if self.implementation == "scipy":
            return scipy.linalg.funm(A, self.f)
        else:
            raise NotImplementedError("Unknown implementation:"
                                      + " {}".format(self.implementation))

    def _evaluate_hermitian(self, A, is_pos_semidefinite=False):
        """Compute matrix function for a Hermitian matrix.

        Parameters
        ----------
        A : 2D numpy array, Hermitian
            Input matrix which is being passed
            to the MatrixFunction for computation.
        is_pos_semidefinite : bool, optional
            Flag for guarding against precision errors by ensuring
            the computed eigenvalues are non-negative.

        Returns
        -------
        2D numpy array
            computed matrix function

        Notes
        _____
        This algorithm (or something similar) is described in [1]
        and also in scipy documentation for scipy.linalg.funm.

        References
        __________
        ..  [1] Higham, Nicholas. Functions of Matrices: Theory
            and Computation. Chapter 4.
        """
        w, v = scipy.linalg.eigh(A, check_finite=True)
        if is_pos_semidefinite:
            w = numpy.maximum(w, 0)
        w = self.f(w)
        return (v * w) @ v.conj().T

    def _evaluate_general_sparse(self, A_sp):
        raise NotImplementedError("Unknown sparse implementation:"
                                  + " {}".format(self.implementation))

    def _evaluate_hermitian_sparse(self, A_sp, is_pos_semidefinite=False):
        raise NotImplementedError("Unknown sparse implementation:"
                                  + " {}".format(self.implementation))

    def __call__(self, A, is_hermitian=False, is_pos_semidefinite=False):
        """Compute matrix function of given matrix.

        Parameters
        ----------
        A : 2D numpy array
        is_hermitian : bool
            Boolean representing if the matrix is Hermitian or not. If
            true and the MatrixFunction has special methods for
            Hermitian matrices, then those methods will be called.
        is_pos_semidefinite : bool
            Boolean representing if the matrix is positive semidefinite
            (has nonnegative eigenvalues). Will be passed to special
            methods in case it is useful information for matrix
            function computation.
        """
        is_sparse = scipy.sparse.issparse(A)
        if is_hermitian and is_sparse:
            return self._evaluate_hermitian_sparse(A, is_pos_semidefinite)
        elif is_hermitian and not is_sparse:
            return self._evaluate_hermitian(A, is_pos_semidefinite)
        elif not is_hermitian and is_sparse:
            return self._evaluate_general_sparse(A)
        else:
            return self._evaluate_general(A)

    def __str__(self):
        return self.__repr__()

    def __repr__(self):
        if self.f_description:
            return str(type(self).__name__) + "({})".format(self.f_description)
        else:
            return str(type(self).__name__) + "({})".format(str(self.f))


class MatrixExponential(MatrixFunction):
    """Specialized MatrixFunction based on exponential function.

    Serves as a wrapper for scipy.linalg.expm and
    scipy.sparse.linalg.expm functions.
    """
    f_description = "numpy.exp(x)"
    f = numpy.exp

    def __init__(self,
                 implementation="scipy"):
        self.implementation = implementation

    def _evaluate_general(self, A):
        if self.implementation == "scipy":
            return scipy.linalg.expm(A)
        else:
            raise NotImplementedError("Unknown evaluation algorithm:"
                                      + " {}".format(self.implementation))

    def _evaluate_hermitian(self, A, is_pos_semidefinite=False):
        if self.implementation == "scipy":
            return scipy.linalg.expm(A)
        elif self.implementation == "hermitian":
            super()._evaluate_hermitian(A, is_pos_semidefinite)
        else:
            raise NotImplementedError("Unknown evaluation algorithm:"
                                      + " {}".format(self.implementation))

    def _evaluate_general_sparse(self, A_sp):
        return scipy.sparse.linalg.expm(A_sp)

    def _evaluate_hermitian_sparse(self, A_sp, is_pos_semidefinite=False):
        return self._evaluate_general_sparse(A_sp)


class MatrixInverse(MatrixFunction):
    """Specialized MatrixFunction based on inverse function.

    Serves as a wrapper for scipy.linalg.inv and
    scipy.sparse.linalg.inv functions.
    """
    f_description = "1/x"
    def f(x: any) -> any: return 1/x

    def __init__(self,
                 implementation="scipy"):
        self.implementation = implementation

    def _evaluate_general(self, A):
        if self.implementation == "scipy":
            return scipy.linalg.inv(A)
        else:
            raise NotImplementedError("Unknown evaluation algorithm:"
                                      + " {}".format(self.implementation))

    def _evaluate_hermitian(self, A, is_pos_semidefinite=False):
        """From scipy notes*** NEED TO CITE"""
        if self.implementation == "scipy":
            return scipy.linalg.inv(A)
        elif self.implementation == "hermitian":
            super()._evaluate_hermitian(A, is_pos_semidefinite)
        else:
            raise NotImplementedError("Unknown evaluation algorithm:"
                                      + " {}".format(self.implementation))

    def _evaluate_general_sparse(self, A_sp):
        return scipy.sparse.linalg.inv(A_sp)

    def _evaluate_hermitian_sparse(self, A_sp, is_pos_semidefinite=False):
        return self._evaluate_general_sparse(A_sp)


class MatrixFunctionSystem:
    """Class container for matrix function systems.

    Attributes
    ----------
    A : 2D numpy array
    mfunc : MatrixFunction
    fA : None type or 2D numpy array, optional
        If None, then fA = mfunc(A) is computed on
        instantiation. Otherwise, fA is saved as the
        given argument.
    diag_only : bool, optional
        Currently unused. In future versions, hope
        to have option to save only the diagonal
        of the computed matrix function
    normal : bool, optional
        Currently unused. States if matrix is normal.
        If so, may be useful for passing to mfunc
        for computational strategy.
    hermitian : bool, optional
        Boolean passed to mfunc when computing matrix
        function.
    pos_semidefinite : bool, optional
        Boolean passed to mfunc when computing matrix
        function

    Examples
    --------
    >>> A = numpy.ones((2, 2))
    >>> mfunc = MatrixExponential()
    >>> print(MatrixFunctionSystem(A, mfunc))
    <class 'krypy.mfuncs.MatrixFunctionSystem'> object
        A: has dimensions (2, 2) and 4 non-zero elements
        type(A): <class 'numpy.ndarray'>
        f: numpy.exp(x)
        f(A) diagonal storage: False
        Normal: False
        Hermitian: False
        Positive-semidefinite: False
    """
    def __init__(self,
                 A,
                 mfunc,
                 fA=None,
                 diag_only=False,
                 normal=False,
                 hermitian=False,
                 pos_semidefinite=False):
        self.is_sparse = scipy.sparse.issparse(A)
        self.A = A
        self.mfunc = mfunc
        self.fA = (fA if fA is not None
                   else mfunc(A,
                              is_hermitian=hermitian,
                              is_pos_semidefinite=pos_semidefinite))
        self.diag_only = diag_only
        if diag_only and len(self.fA.shape) > 1:
            self.fA = (numpy.diag(self.fA) if (not self.is_sparse)
                       else numpy.diag(self.fA.toarray()))

        self.normal = True if hermitian or normal else False
        self.hermitian = hermitian
        self.pos_semidefinite = pos_semidefinite

    def get_mat(self):
        return self.A

    def get_mfunc(self):
        return self.mfunc

    def get_computed_mfunc(self):
        return self.fA

    def __repr__(self):
        s = str(type(self)) + " object\n"
        num_nonzero = (self.A.nnz if self.is_sparse
                       else sum(sum(self.A != numpy.zeros(self.A.shape))))
        s += "    A: has dimensions {} and {} non-zero elements\n".format(
            self.A.shape,
            num_nonzero
            )
        s += "    type(A): {}\n".format(type(self.A))
        s += "    f: {}\n".format(self.mfunc.f_description)
        s += "    f(A) diagonal storage: {}\n".format(str(self.diag_only))
        s += "    Normal: " + str(self.normal) + "\n"
        s += "    Hermitian: " + str(self.hermitian) + "\n"
        s += "    Positive-semidefinite: " + str(self.pos_semidefinite)
        return s


class RankOneUpdate:
    """Algorithms for rank-one update of MatrixFunctionSystem.

    Attributes
    ----------
    mfunc_system : MatrixFunctionSystem
    b : numpy array, size (n, 1)
        b is the vector used for the low-rank update. It
        should be stored as a column vector.
    c : numpy array, size (n, 1), optional
        If c is not given, it is assumed that the update
        is of the form b @ b.conj().T. Otherwise,
        the update is of the form b @ c.conj().T.
    sign : int, +/-1, optional
        sign representing whether the update is
        additive or subtractive for the system.
        That is, update is of the form
        A + sign * b @ c.conj().T
    maxiter : int, optional
        maximum number of iterations to be performed
    tol : float, optional
        target accuracy
    d : int, optional
        number of intermediate compressed matrix functions
        to store for estimating residuals. Larger d
        leads to more accurate residual estimate.
    orth : str, optional
        orthogonalization algorithm for Arnoldi iteration.
    explicit_update : 2D numpy array, optional
        If None, then does nothing. If a 2D array,
        then instead of estimating residuals the
        approximations are compared to this explicit
        update.
    dropping_tol : float, optional
        Currently unused. In the future may incorporate
        dropping of small elements in matrix function
        computation.
    store_res : bool, optional
        Indicates whether residuals should be stored
        at each iterate of algorithm.

    Notes
    -----
    For usage examples, see special topic report.
    The algorithms in this class, specifically those
    found in _compute_Xkf_hermitian and
    _compute_Xkf_nonhermitian, are adapted from [2].

    References
    __________
    .. [2] Beckermann et al. 2018. "Low-rank Updates of
       Matrix Functions." SIAM Matrix Analysis.

    """
    def __init__(self,
                 mfunc_system,
                 b,
                 c=None,
                 sign=1,
                 maxiter=None,
                 tol=1e-8,
                 d=2,
                 ortho="dmgs",
                 explicit_update=None,
                 dropping_tol=None,
                 store_res=True):
        # set attributes
        self.N = b.shape[0]
        self.mfunc_system = mfunc_system
        self.A = mfunc_system.get_mat()
        self.fA = mfunc_system.get_computed_mfunc()
        self.mfunc = mfunc_system.get_mfunc()
        self.diag_only = mfunc_system.diag_only
        self.is_sparse = mfunc_system.is_sparse

        self.b = b
        c_given = False if c is None or c.size == 0 else True
        self.c = c if c_given else b

        self.sign = sign

        self.maxiter = maxiter if maxiter is not None else min(20, self.N - 1)

        self.tol = tol

        self.d = d

        self.explicit_update = explicit_update

        self.dropping_tol = dropping_tol

        self.ortho = ortho

        self.hermitian = min([numpy.all(self.b == self.c),
                              mfunc_system.hermitian])

        # Construct left and right Arnoldi--left only if Hermitian.
        self.l_arnoldi = Arnoldi(self.A,
                                 self.b,
                                 self.maxiter,
                                 self.ortho)
        if not self.hermitian:
            self.r_arnoldi = Arnoldi(self.A.conj().T,
                                     self.sign * self.c,
                                     self.maxiter,
                                     self.ortho)
        else:
            self.r_arnoldi = self.l_arnoldi

        if store_res:
            self.res = numpy.zeros(self.N)
            self.res[0] = float('inf')
        else:
            self.res = None

        self.Xkf = None
        self.oldXkfs = deque()
        self.has_converged = False

    def advance(self):
        """Advance one iteration of Arnoldi."""
        self.l_arnoldi.advance()
        if not self.hermitian:
            self.r_arnoldi.advance()

        # If we are not storing residuals, only
        # need to advance Arnoldi bases
        if self.res is None:
            return

        # Else must keep track of intermediate Xkf and
        # compute residuals along the way.
        k = self.l_arnoldi.iter
        if k > 1 and len(self.oldXkfs) < self.d:
            self.oldXkfs.append(self.Xkf)
        elif k > 1 and len(self.oldXkfs) >= self.d:
            self.oldXkfs.popleft()
            self.oldXkfs.append(self.Xkf)

        if self.hermitian:
            self.Xkf = self._compute_Xkf_hermitian()
        else:
            self.Xkf = self._compute_Xkf_nonhermitian()

        if self.res is not None:
            self.res[k-1] = self._estimate_residual()
            if self.res[k-1] < self.tol:
                self.has_converged = True

    def _compute_Xkf_hermitian(self):
        """See [2] for algorithm details."""
        iter = self.get_iter()
        U, G, V, H = self.get_arnoldi()
        size_b = numpy.linalg.norm(self.b, 2)
        e1 = numpy.zeros((iter, 1))
        e1[0, 0] = 1
        M1 = self.mfunc(
            G[:iter, :iter] + (self.sign * size_b**2 * e1) @ e1.conj().T
            )
        M2 = self.mfunc(G[:iter, :iter])
        Xkf = M1 - M2
        return Xkf

    def _compute_Xkf_nonhermitian(self):
        """See [2] for algorithm details.."""
        iter = self.get_iter()
        U, G, V, H = self.get_arnoldi()
        size_b = numpy.linalg.norm(self.b, 2)
        size_c = numpy.linalg.norm(self.c, 2)
        e1 = numpy.zeros((iter, 1))
        e1[0, 0] = 1
        M1 = G[:iter, :iter]
        M2 = size_b * size_c * (e1 @ e1.conj().T)
        M3 = numpy.zeros((iter, iter))
        M4 = H[:iter, :iter].conj().T + (size_c * (V[:, :iter].conj().T @ self.b)) @ e1.conj().T
        T = numpy.block(
            [[M1, M2],
             [M3, M4]]
        )
        F = self.mfunc(T)
        Xkf = F[:iter, iter:2*iter]
        return Xkf

    def _estimate_residual(self):
        """See [2] for explanation of this residual estimate."""
        if self.explicit_update is not None:
            return self._compute_exact_residual()

        if self.Xkf is None:
            return float("inf")
        elif len(self.oldXkfs) < self.d:
            return numpy.linalg.norm(self.Xkf)
        else:
            k_plus_d = self.Xkf.shape[0]
            k = k_plus_d - self.d
            d = self.d
            R = self.Xkf
            R = R - numpy.block([[self.oldXkfs[0],     numpy.zeros((k, d))],
                                 [numpy.zeros((d, k)), numpy.zeros((d, d))]])
            return numpy.linalg.norm(R, 2)

    def _compute_exact_residual(self):
        if not self.is_sparse:
            return numpy.linalg.norm(self.get_update()
                                     - self.explicit_update, 2)
        else:
            return scipy.sparse.linalg.norm(self.get_update()
                                            - self.explicit_update)

    def _diagonal_update(self):
        raise NotImplementedError("No diagonal update defined yet.")

    def _full_update(self):
        iter = self.l_arnoldi.iter
        U, G, V, H = self.get_arnoldi()
        if self.Xkf is None and iter < 1:
            dim = (self.N, self.N)
            return (numpy.zeros(dim) if not self.mfunc_system.is_sparse
                    else scipy.sparse.csc_array(dim))
        elif self.Xkf is None and iter >= 1:
            Xkf = (self._compute_Xkf_hermitian() if self.hermitian
                   else self._compute_Xkf_nonhermitian())
        else:
            Xkf = self.Xkf

        if self.is_sparse:
            U = scipy.sparse.csc_array(U)
            Xkf = scipy.sparse.csc_array(Xkf)
            V = scipy.sparse.csc_array(V)

        return U[:, :iter] @ Xkf @ V[:, :iter].conj().T

    def get_update(self):
        if self.get_iter() < 1:
            raise ArgumentError("No iterations for"
                                + " low-rank update performed yet")

        if self.diag_only:
            return self._diagonal_update()
        else:
            return self._full_update()

    def get_updated_matrix(self):
        return self.A + (self.sign * self.b) @ self.c.conj().T

    def get_updated_matrix_function(self):
        return self.fA + self.get_update()

    def get_new_system(self, maintains_psd=False, maintains_normality=False):
        return MatrixFunctionSystem(self.get_updated_matrix(),
                                    self.mfunc,
                                    self.fA + self.get_update(),
                                    self.diag_only,
                                    maintains_normality,
                                    self.hermitian,
                                    maintains_psd)

    def get_iter(self):
        return self.l_arnoldi.iter

    def get_arnoldi(self):
        if self.hermitian:
            U, G = self.l_arnoldi.get()
            V, H = U, G
        else:
            U, G = self.l_arnoldi.get()
            V, H = self.r_arnoldi.get()
        return U, G, V, H

    def __repr__(self):
        s = str(type(self)) + " object\n"
        s += "\n".join(str(self.mfunc_system).split("\n")[1:-3]) + "\n"
        s += "    b: vector with {} elements and size {}\n".format(
            self.N, numpy.linalg.norm(self.b, 2))
        s += "    c: vector with {} elements and size {}\n".format(
            self.N, numpy.linalg.norm(self.c, 2))
        s += "    b == c: {}\n".format(str(numpy.all(self.b == self.c)))
        s += "    type of update: {}, {}\n".format(
            "additive" if self.sign == 1 else "subtractive",
            "hermitian" if self.hermitian else "nonhermitian"
        )
        s += "    current iteration: {}\n".format(self.get_iter())
        s += "    maximum iteration: {}\n".format(self.maxiter)
        s = (s + "    error estimate: {}\n".format(self.res[self.get_iter() - 1])
             if self.res is not None and self.get_iter() > 0 else s)
        s += "    target tolerance: {}\n".format(self.tol)
        if self.dropping_tol:
            s += "    dropping tolerance: {}\n".format(self.dropping_tol)
        U, G, V, H = self.get_arnoldi()
        if self.hermitian and G.size > 0:
            s += "    arnoldi residual: {}\n".format(arnoldi_res(self.A, U, G))
            s += "    arnoldi condition number: {}\n".format(
                numpy.linalg.cond(U) if U.size <= 1e8 else "N/A, size > 1e8"
            )
        elif G.size > 0:
            s += "    left arnoldi residual: {}\n".format(
                arnoldi_res(self.A, U, G)
                )
            s += "    left arnoldi condition number: {}\n".format(
                numpy.linalg.cond(U) if U.size <= 1e8 else "N/A, size > 1e8"
                )
            s += "    right arnoldi residual: {}\n".format(
                arnoldi_res(self.A.conj().T, V, H))
            s += "    right arnoldi condition number: {}\n".format(
                numpy.linalg.cond(V) if V.size <= 1e8 else "N/A, size > 1e8"
                )
        else:
            s += "    arnoldi: unavailable\n"
        return s


def rank_one_update(A, f, fA, b, return_object=False,
                    hermitian=False, **kwargs):
    """Wraps RankOneUpdate class for convenience.

    Instantiates RankOneUpate and runs until it converges,
    reaches maximum iterations, or becomes an invariant
    Arnoldi basis.

    Parameters
    ----------
    A : 2D numpy array, size (n, n)
    f : MatrixFunction
    fA : 2D numpy array
        pre-computed matrix function, or None
        and it will be computed necessarily
    b : 2D numpy array, size (n, 1)
    return_object : bool, optional
        If True, then the function returns a RankOneUpdate
        object which includes details about the convergence
        of the method and Arnoldi bases and more. Else,
        simply returns the relevant update.
    hermitian : bool, optional
        If True, will be passed to MatrixFunctionSystem
        for computation of fA and in order to keep track
        of whether the update should use the Hermitian
        algorithm.
    kwargs : dictionary, optional
        To be passed to RankOneUpdate object. See RankOneUpdate
        for more details.

    Returns
    -------
    2D numpy array if not return_object else RankOneUpdate

    Usage
    -----
    >>> A = np.random.rand(10, 10) #random
    >>> f = MatrixExponential()
    >>> fA = None
    >>> b = np.random.rand(10, 1) #random
    >>> scipy.linalg.norm(
    ...    rank_one_update(A, f, fA, b)
    ...     - (f(A + b @ b.T) - f(A))
    ... )
    0.003044040958273476 #random

    """
    _mfunc_system = MatrixFunctionSystem(A, f, fA, diag_only=False,
                                         normal=False,
                                         hermitian=hermitian,
                                         pos_semidefinite=False)
    _rank_one_update = RankOneUpdate(_mfunc_system,
                                     b,
                                     **kwargs)
    maxiter = _rank_one_update.maxiter
    while (_rank_one_update.get_iter() < maxiter
           and not _rank_one_update.has_converged
           and not _rank_one_update.l_arnoldi.invariant
           and not _rank_one_update.r_arnoldi.invariant):
        _rank_one_update.advance()

    if return_object:
        return _rank_one_update
    else:
        return _rank_one_update.get_update()
