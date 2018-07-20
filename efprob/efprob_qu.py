#
# Quantum probability library, prototype version
#
# Copyright: Bart Jacobs, Kenta Cho; 
# Radboud University Nijmegen
# efprob.cs.ru.nl
#
# Date: 2017-07-15
#
from functools import reduce
import functools
import itertools
import operator
import math
import cmath
import numbers
import numpy as np
import numpy.linalg
import numpy.random
import random
import scipy.linalg
import matplotlib.pyplot as plt


# http://qutip.org/docs/2.2.0/guide/guide-basics.html

# About arbitrary quantum channels:
# https://arxiv.org/abs/1611.03463
# https://arxiv.org/abs/1609.08103
#
# Security protocol with quantum channel:
# https://arxiv.org/pdf/1407.3886
#
# Explanations at:
# https://quantiki.org/wiki/basic-concepts-quantum-computation
#

float_format_spec = ".3g"

tolerance = 1e-8


########################################################################
# 
# Preliminary definitions
#
########################################################################


def approx_eq_num(r, s):
    return r.real - s.real <= tolerance and s.real - r.real <= tolerance and \
        r.imag - s.imag <= tolerance and s.imag - r.imag <= tolerance

def round(x):
    sign = 1 if x >= 0 else -1
    y = math.floor((sign * x + 0.5 * tolerance) / tolerance) 
    return sign * y * tolerance

def round_matrix(m):
    return np.vectorize(round)(m)

def approx_eq_mat(M, N):
    out = (M.shape == N.shape)
    n = M.shape[0]
    for i in range(n):
        for j in range(n):
            out = out and approx_eq_num(M[i][j], N[i][j])
    return out

def prod(iterable):
    """Returns the product of elements from iterable."""
    return reduce(operator.mul, iterable, 1)

def matrix_square_root(mat):
    E = np.linalg.eigh(mat)
    # rounding needed since eigenvalues are sometimes negative, by a
    # very small amount
    rounded_eigenvalues = [round(x) for x in E[0]]
    sq = np.dot(np.dot(E[1], 
                       np.sqrt(np.diag(rounded_eigenvalues))), 
                np.linalg.inv(E[1]))
    return sq

def matrix_absolute_value(mat):
    M = np.dot(mat, conjugate_transpose(mat))
    return matrix_square_root(M)

#
# Produce the list of eigenvectors vi, with (roots of )eigenvalues
# incorporated so that mat = sum |vi><vi|, given in python as:
#
# sum([ np.outer(v.conj(), v).T for v in spectral_decomposition(mat) ]) 
#
# see the nympy outer product description at:
# https://docs.scipy.org/doc/numpy/reference/generated/numpy.outer.html
#
def spectral_decomposition(mat):
    E = np.linalg.eigh( mat )
    EVs = [cmath.sqrt(x) * y for x,y in zip(list(E[0]), list(E[1].T))]
    return EVs

def kraus_decomposition(mat, n, m):
    EVs = spectral_decomposition(mat)
    # Turn each n*m vector e in EVs into an nxm matrix kraus_e
    out = []
    for e in EVs:
        kraus_e = np.zeros((n,m)) + 0j
        for i in range(n):
            for j in range(n):
                kraus_e[i][j] = e[j*n+i]
    out.append(kraus_e)
    return out

#
# Purification of a (square) matrix, which is assumed to be positive
#
def purify(mat):
    n = mat.shape[0]
    EV = spectral_decomposition(mat)
    return sum([np.kron(EV[i], vector_base(i,n)) for i in range(n)])

#
# We concentrate on square matrices/arrays
#

def is_square(mat):
    return len(mat.shape) == 2 and mat.shape[0] == mat.shape[1]

def is_zero(mat):
    return is_square(mat) \
        and np.all(mat <= tolerance) \
        and np.all(mat >= -tolerance)

def conjugate_transpose(mat):
    return mat.conj().T

def is_symmetric(mat):
    return is_square(mat) and (mat == mat.T).all()

def is_hermitian(mat):
    #print("hermitian\n", mat, "\n", conjugate_transpose(mat))
    return is_square(mat) and np.allclose(mat, conjugate_transpose(mat))

#
# Trick to get a random unitary nxn matrix U: 
# u = random_pred([n])
# U = scipy.linalg.expm(complex(0,-1) * u.array)
#
def is_unitary(mat):
    out = is_square(mat) 
    n = mat.shape[0]
    out = out and np.allclose(np.dot(mat, conjugate_transpose(mat)), 
                              np.eye(n))
    out = out and np.allclose(np.dot(conjugate_transpose(mat), mat), 
                              np.eye(n))
    return out

def is_isometry(mat):
    n = mat.shape[1]
    return np.allclose(np.dot(conjugate_transpose(mat), mat), 
                       np.eye(n))


def is_positive(mat):
    if not is_hermitian(mat):
        return False
    E = np.linalg.eigvals(mat)
    #print("eigenvalues", E)
    out = all([e.real > -tolerance for e in E])
    #print("is_positive", out)
    return out
    # if is_zero(mat):
    #     return True
    # try:
    #     # approach from http://stackoverflow.com/questions/16266720/find-out-if-matrix-is-positive-definite-with-numpy
    #     ch = np.linalg.cholesky(mat)
    #     return True
    # except:
    #     return False

def lowner_le(m1, m2):
    return is_positive(m2 - m1)

def is_effect(mat):
    if not is_positive(mat):
        print("not positive!")
        return False
    n = mat.shape[0]
    out = lowner_le(mat, np.eye(n))
    if not out:
        print("Not below the identity; difference is:\n",
              np.eye(n) - mat)
    return out

def is_state(mat):
    tr = np.trace(mat)
    print(round(tr.real), round(tr.imag))
    return is_positive(mat) and round(tr.real) == 1.0 and round(tr.imag) == 0.0


def entanglement_test(s):
    return np.allclose(s.array,
                       np.kron((s%[1,0]).array, (s%[0,1]).array))

#
# Standard base vector |i> in C^n
#
def vector_base(i,n):
    ls = np.zeros(n)
    ls[i] = 1
    return ls

#
# Standard base vector |i><j| in Mat_n
#
def matrix_base(i,j,n):
    mat = np.zeros((n,n))
    mat[i][j] = 1
    return mat



########################################################################
# 
# Classes
#
########################################################################


class Dom:
    """(Co)domains of states, predicates and channels"""
    def __init__(self, dims):
        # dims is list of numbers (dimensions of components)
        self.dims = dims
        self.size = prod(dims)

    def __repr__(self):
        return str(self.dims)

    def __eq__(self, other):
        return len(self.dims) == len(other.dims) and \
            all([self.dims[i] == other.dims[i] for i in range(len(self.dims))])

    def __ne__(self, other):
        return not (self == other)

    def __add__(self, other):
        """ concatenation of lists of dimensions """
        return Dom(self.dims + other.dims)

    def __mul__(self, n):
        if n == 0:
            raise Exception('Non-zero number required in Domain multiplication')
        if n == 1:
            return self
        return self + (self * (n-1))


class RandVar:
    """ Hermitian (self-adjoint) operators, forming the superclass for 
    both predicates (effects) and states (density matrices) """
    def __init__(self, ar, dom):
        self.array = ar
        self.dom = dom if isinstance(dom, Dom) else Dom(dom)
        if not is_hermitian(self.array):
            raise Exception('Random variable creation requires a hermitian matrix')
        if ar.shape[0] != self.dom.size:
            raise Exception('Non-matching matrix in random variable creation')

    def __repr__(self):
        return str(self.array)

    def __eq__(self, other):
        """ Equality test == """
        if not isinstance(other, RandVar):
            raise Exception('Equality of random variables requires a random variable argument')
        return self.dom == other.dom \
            and np.all(np.isclose(self.array, other.array))

    def __ne__(self, other):
        """ Non-equality test != """
        return not self == other

    def __neg__(self):
        """ Negation - """
        return RandVar(-self.array, self.dom)

    def __add__(self, other):
        """ Addition + """
        if not isinstance(other, RandVar):
            raise Exception('Addition of random variables requires a random variable argument')
        if not self.dom == other.dom:
            raise Exception('Addition of random variables requires equality of domains')
        return RandVar(self.array + other.array, self.dom)

    def __sub__(self, other):
        """ Subtraction - """
        if not isinstance(other, RandVar):
            raise Exception('Subtraction of random variables requires a random variable argument')
        if not self.dom == other.dom:
            raise Exception('Subtraction of random variables requires equality of domains')
        return RandVar(self.array - other.array, self.dom)

    def __mul__(self, scalar):
        """Multiplication * with a scalar. (Recall that Hermitian /
        self-adjoint matrices are not closed under Matrix
        multiplication, so this is not defined.) """
        if not isinstance(scalar, numbers.Real):
            raise Exception('Scalar multiplication for random variables is only defined for real numbers')
        return RandVar(scalar * self.array, self.dom)

    def __rmul__(self, scalar):
        """ Scalar multiplication * with scalar written first """
        return self * scalar

    def __matmul__(self, other):
        """ Parallel product @ """
        if not isinstance(other, RandVar):
            raise Exception('Parallel composition of random variables requires a random variable argument')
        return type(self)(np.kron(self.array, other.array), self.dom + other.dom)

    def __pow__(self, n):
        """ Iterated parallel product """
        if n == 0:
            raise Exception('Power of a random variable must be at least 1')
        return reduce(lambda r1, r2: r1 @ r2, [self] * n)

    def transpose(self):
        """Experimental: take transpose = (conjugate) of the matrix 
        of a random variable"""
        return RandVar(self.array.transpose(), self.dom)

    def evolution(self, state):
        if not isinstance(state, State):
            raise Exception('Evolution of a random variable requires a state as first argument')
        if self.dom != state.dom:
            raise Exception('Domain mismatch in the evolution of a random variable')
        def U(t):
            return scipy.linalg.expm(complex(0,-1) * t * self.array)
        def ch(t):
            return channel_from_isometry(U(t), self.dom, self.dom)
        return lambda t: ch(t) >> state

    def plot_evolution(self, state, randvar, lower_bound, upper_bound, steps=100):
        if not isinstance(state, State) or not isinstance(randvar, RandVar):
            raise Exception('Type mismatch in the plotting of a random variable')
        if self.dom != state.dom or self.dom != randvar.dom:
            raise Exception('Domain mismatch in the plotting of a random variable')
        plot( lambda t: self.evolution(state)(t) >= randvar, 
              lower_bound,
              upper_bound,
              steps )



class Predicate(RandVar):
    def __init__(self, ar, dom):
        #if not is_effect(ar):
        #    raise Exception('Predicate creation requires a effect matrix')
        super().__init__(ar, dom)

    # The selection is a dim-length list of 0's and 1's, where 0
    # corresponds to marginalisation of the corresponding component.
    # This copied from the same operations for states.  operation is
    # not guaranteed to produce a predicate. Hence it is a "partial"
    # partial trace!
    #
    # def __mod__(self, selection):
    #     n = len(selection)
    #     if n != len(self.dims):
    #         raise Exception('Wrong length of marginal selection')
    #     dims = [n for (n,i) in zip(self.dims, selection) if i == 1]
    #     mat = self.array.reshape(tuple(self.dims + self.dims)) + 0.j
    #     #print(mat.shape)
    #     marg_pos = 0
    #     marg_dist = n
    #     for i in range(n):
    #         if selection[i] == 1:
    #             marg_pos = marg_pos+1
    #             continue
    #         #print(i, marg_pos, marg_dist)
    #         mat = mat.trace(axis1 = marg_pos, axis2 = marg_pos + marg_dist)
    #         marg_dist = marg_dist - 1
    #     p = prod(dims)
    #     mat.resize(p,p)
    #     return Predicate(mat, dims)

    def __invert__(self):
        """" Orthocomplement ~ """
        return Predicate(np.eye(self.dom.size) - self.array, self.dom)

    def __mul__(self, scalar):
        if not isinstance(scalar, numbers.Real):
            raise Exception('Scalar multiplication for predicates is only defined for real numbers')
        if scalar < 0.0 or scalar > 1.0:
            return super().__mul__(scalar)
        return Predicate(scalar * self.array, self.dom)

    def __rmul__(self, r):
        return self * r

    def __add__(self, pred):
        """ Partial addition + """
        if not isinstance(pred, Predicate) and isinstance(pred, RandVar):
            return super().__add__(pred)
        if not isinstance(pred, Predicate):
            raise Exception('Sum of predicates requires a predicate argument')
        if self.dom != pred.dom:
            raise Exception('Mismatch of dimensions in sum of predicates')
        mat = self.array + pred.array
        if not lowner_le(mat, np.eye(self.dom.size)):
            return super().__add__(pred)
        return Predicate(mat, self.dom)

    def __and__(self, pred):
        """ Sequential conjunction & """
        if not isinstance(pred, Predicate):
            raise Exception('Sequential conjunction of predicates requires a predicate argument')
        sq = matrix_square_root(self.array)
        return Predicate(np.dot(sq, np.dot(pred.array, sq)), self.dom)

    def __or__(self, p):
        """De Morgan dual of sequential conjunction."""
        return ~(~self & ~p)

    def as_subchan(self):
        """Turn a predicate on n into a channel n -> 0. The validity s >= p is
        the same as p.as_subchan() >> s, except that the latter is a 1x1
        matrix, from which the validity can be extracted via indices
        [0][0]. """
        return Channel(self.array.reshape(1,1, self.dom.size, self.dom.size), 
                       self.dom, [])

    def transpose(self):
        """ Experimental: take transpose (= conjugate) of the matrix of 
        a predicate """
        return Predicate(self.array.transpose(), self.dom)



class State:
    def __init__(self, ar, dom):
        self.array = ar
        self.dom = dom if isinstance(dom, Dom) else Dom(dom)
        if not is_positive(ar):
            print(ar)
            raise Exception('State creation requires a positive matrix')
        if round(np.trace(ar).real) != 1.0 or round(np.trace(ar).imag) != 0.0:
            print("  --> Warning: trace is not 1.0 in state creation, but:", np.trace(ar))

    # unfinished
    def __repr__(self):
        return str(self.array)
    # np.array2string(self.array, 
    #                            separator=',  ',
    #                            formatter={'complexfloat':lambda x: '%3g + %3gi' 
#                                          % (x.real, x.imag)})

    def __eq__(self, other):
        """ Equality test == """
        if not isinstance(other, State):
            raise Exception('Equality of state requires a state argument')
        return self.dom == other.dom \
            and np.all(np.isclose(self.array, other.array))

    def __ne__(self, other):
        """ Non-equality test != """
        return not self == other

    def __mod__(self, selection):
        """Marginalisation. The selection is a dim-length list of 0's and
        1's, where 0 corresponds to marginalisation of the
        corresponding component.

        How this works in a state with 3 components
        third marginal
        mat.trace(axis1 = 0, axis2 = 3).trace(axis1 = 0, axis2 = 2))
        second marginal
        mat.trace(axis1 = 0, axis2 = 3).trace(axis1 = 1, axis2 = 3))
        first marginal
        mat.trace(axis1 = 1, axis2 = 4).trace(axis1 = 1, axis2 = 3))
        """
        n = len(selection)
        if n != len(self.dom.dims):
            raise Exception('Wrong length of marginal selection')
        dims = [n for (n,i) in zip(self.dom.dims, selection) if i == 1]
        mat = self.array.reshape(tuple(self.dom.dims + self.dom.dims))
        #print(mat.shape)
        marg_pos = 0
        marg_dist = n
        for i in range(n):
            if selection[i] == 1:
                marg_pos = marg_pos+1
                continue
            #print(i, marg_pos, marg_dist)
            mat = mat.trace(axis1 = marg_pos, axis2 = marg_pos + marg_dist)
            marg_dist = marg_dist - 1
        p = prod(dims)
        mat.resize(p,p)
        return State(mat, Dom(dims))

    def __matmul__(self, other):
        """ Parallel product @ """
        if not isinstance(other, State):
            raise Exception('Parallel compositon of states requires a state argument')
        return State(np.kron(self.array, other.array), self.dom + other.dom)

    def __pow__(self, n):
        """ Iterated parallel product """
        if n == 0:
            raise Exception('Power of a random variable must be at least 1')
        return reduce(lambda r1, r2: r1 @ r2, [self] * n)

    def __ge__(self, rv):
        """ Validity """
        if not isinstance(rv, RandVar):
            raise Exception('Validity requires a random variable')
        if self.dom != rv.dom:
            raise Exception('State and random variable must have equal domains in validity')
        return np.trace(np.dot(self.array, rv.array)).real

    def __truediv__(self, pred):
        """ Conditioning """
        if not isinstance(pred, Predicate):
            raise Exception('Non-predicate used in conditioning')
        if self.dom != pred.dom:
            raise Exception('State and predicate with different domains in conditioning')
        v = self >= pred
        if v == 0:
            raise Exception('Zero-validity excludes conditioning')
        sq = matrix_square_root(pred.array)
        #print("square root", p.array, sq, np.dot(sq, np.dot(self.array, sq)))
        return State( np.dot(sq, np.dot(self.array, sq)) / v, self.dom)
        
    # # convex sum of two states
    # def __add__(self, stat):
    #     return lambda r: convex_state_sum(*[(r, self), (1-r, stat)])

    def as_pred(self):
        """ Turn a state into a predicate """
        return Predicate(self.array, self.dom)

    def transpose(self):
        """ Experimental: take transpose (= conjugate) of the matrix of 
        a state """
        return State(self.array.transpose(), self.dom)

    def __xor__(self, pred):
        """ Experimental: sequential conjunction & """
        if not isinstance(pred, Predicate):
            raise Exception('Conditioning requires a predicate argument')
        sq = matrix_square_root(self.array)
        conj = 1/(self >= pred) * np.dot(sq, np.dot(pred.array, sq))
        return State(conj, self.dom)

    def as_chan(self):
        """Turn a state on dom into a channel 0 -> dom. This is useful for
        bringing in extra "ancillary" bits into the system.  Not that
        the empty list [] is the appropriate domain type, where the
        product over this list is 1, as used in the dimension of the
        matrix. """
        n = self.dom.size
        mat = np.zeros((n,n,1,1)) + 0j
        mat[...,0,0] = self.array.conjugate()
        return Channel(mat, [], self.dom)

    def variance(self, randvar):
        """ Variance """
        if not isinstance(randvar, RandVar):
            raise Exception('Variance of a state requires a random variable argument')
        v = self >= randvar
        w = self >= RandVar(np.dot(randvar.array, randvar.array), self.dom)
        return w - (v ** 2)

    # def purify(self):
    #     v = purify(self.array)
    #     mat = np.outer(v, np.conjugate(v))
    #     return State(mat, [self.dom.size, self.dom.size])


        

# A channel A -> B
#
# Schroedinger picture: CP-map C : L(H_A) -> L(H_B) preserving traces 
#
# Heisenberg picture: unital CP-map D : L(H_B) -> L(H_A)
#
# This D = C*, satisfying <A, C(rho)> = <D(A), rho> which is the
# transformation validity rule, using that < , > is the
# Hilbert-Schmidt inner product, given by tr(-.-).
#
# Here we follow the Heisenberg picture, as used in von Neumann algebras.
#
class Channel:
    def __init__(self, ar, dom, cod):
        self.array = ar
        self.dom = dom if isinstance(dom, Dom) else Dom(dom)
        self.cod = cod if isinstance(cod, Dom) else Dom(cod)
        if ar.shape[0] != self.cod.size or ar.shape[1] != self.cod.size \
           or ar.shape[2] != self.dom.size or ar.shape[3] != self.dom.size:
            raise Exception('Non-matching matrix in channel creation')
        # mat is a (cod.size x cod.size) matrix of (dom.size x
        # dom.size) matrices Hence its shape is (cod.size, cod.size,
        # dom.size, dom.size) This a map dom -> cod in vNA^op

    def __repr__(self):
        return str(self.array)

        # "channel from" + str(self.dom_dims) + "to" + str(self.cod_dims)

    def __eq__(self, other):
        return self.dom == other.dom and self.cod == other.cod \
            and np.all(np.isclose(self.array, other.array))

    def __ne__(self, other):
        return not self == other

    # backward predicate transformation
    def __lshift__(self, p):
        #print(p.dims, self.dom_dims, self.cod_dims)
        if p.dom != self.cod:
            raise Exception('Non-match in predicate transformation')
        m = p.dom.size # = self.cod.size
        n = self.dom.size
        mat = np.zeros((n,n)) + 0j
        # Perform a linear extension of the channel, encoding its
        # behaviour on basisvectors in a matrix to arbitrary matrices.
        for k in range(m):
            for l in range(m):
                mat = mat + p.array[k][l] * self.array[k][l]
        return Predicate(mat, self.dom)

    # forward state transformation
    def __rshift__(self, s):
        #print("rshift dims", s.dims, self.dom_dims)
        if s.dom != self.dom:
            raise Exception('Non-match in state transformation')
        n = s.dom.size # = self.dom.size
        m = self.cod.size
        mat = np.zeros((m, m)) + 0j
        for k in range(m):
            for l in range(m):
                # NB: the order of k,l must be different on the left
                # and right hand side, because in the Hilbert-Schmidt
                # inner product a transpose is used: <A,B> = tr(A*B).
                mat[k][l] = np.trace(np.dot(s.array, 
                                            self.array[l][k]))
        #print("rshift out", is_positive(mat), np.trace(mat), "\n", mat)
        return State(mat, self.cod)

    # parallel compositon
    def __matmul__(self, c):
        return Channel(np.kron(self.array, c.array), 
                       self.dom + c.dom,
                       self.cod + c.cod)

    # sequential composition
    def __mul__(self, c):
        if self.dom != c.cod:
            raise Exception('Non-matching dimensions in channel composition')
        #print(self.array, "\n", c.array)
        n = c.dom.size
        m = self.cod.size
        p = self.dom.size # = c.cod.size
        #print("channel composition dimensions, ( n =", n, ") -> ( p =", p, ") -> ( m =", m, ")")
        #
        # c.array is pxp of nxn, self.array = mxm of pxp
        # output mat must be mxm of nxn
        # 
        # these numbers must be double checked; everything is square so far
        mat = np.zeros((m,m,n,n)) + 0j
        #print("shapes", c.array.shape, self.array.shape, mat.shape)
        for i in range(m):
            for j in range(m):
                mat[i][j] = sum([self.array[i][j][k][l] * c.array[k][l]
                                 for k in range(p) for l in range(p)])
        return Channel(mat, c.dom, self.cod)

    # experimental, based on Leifer-Spekkens
    def inversion(self, state):
        if len(self.dom.dims) > 1 or len(self.cod.dims) > 1:
            raise Exception('Inversion is defined only for channels with domain and codomain of dimension one')
        n = self.dom.dims[0]
        m = self.cod.dims[0]
        pair = graph_pair(state, self)
        return pair_extract(swaps(n,m) >> pair)[1]

    # turn c << (-) into c >> (-), as Hilbert-Schmidt dagger; the
    # result may not be a unitary operation. We do have:
    # c == c.dagger().dagger() 
    # p == (p.as_subchan().dagger() >> init_state).as_pred() 
    # s.as_pred() == s.as_chan().dagger() << truth([]) 

    def dagger(self):
        n = self.dom.size
        m = self.cod.size
        mat = np.zeros((n,n,m,m)) + 0j
        for i in range(n):
            for j in range(n):
                # essentially, self >> matrix_base(i,j,n)
                for k in range(m):
                    for l in range(m):
                        mat[i][j][k][l] = np.trace(np.dot(matrix_base(i,j,n), 
                                                          self.array[l][k]))
        return Channel(mat, self.cod, self.dom)

    def as_operator(self):
        """ Operator from Channel """
        n = self.dom.size
        m = self.cod.size
        mat = np.zeros((n*m,n*m)) + 0j
        for i in range(n):
            for j in range(n):
                for k in range(m):
                    for l in range(m):
                        mat[m*i+k][m*j+l] = self.array[k][l][i][j]
        #print("positive operator", is_positive(mat))
        return Operator(mat, self.dom, self.cod)

    def as_kraus(self):
        return self.as_operator().as_kraus()

    def purify(self):
        """ to be finished """
        array_list = self.as_operator().as_kraus().array_list
        print(len(array_list))
        print(array_list[0].shape)
        return None


#
# Alternative representation of a channel n -> m, namely as a
# square n*m x n*m matrix, called (transition) operator
#
class Operator:
    def __init__(self, mat, dom, cod):
        #print(mat.shape, dom_dims, cod_dims)
        self.array = mat
        self.dom = dom
        self.cod = cod
        if mat.shape[0] != self.dom.size * self.cod.size \
           or mat.shape[1] != self.dom.size * self.cod.size:
            raise Exception('Non-matching matrix in channel creation')

    def __str__(self):
        return str(self.array)

    def __eq__(self, other):
        return np.all(np.isclose(self.array, other.array))

    def __ne__(self, other):
        return not (self == other)

    def __lshift__(self, p):
        """ backward predicate transformation """
        if p.dom != self.cod:
            raise Exception('Non-match in predicate transformation')
        n = self.dom.size
        out = tr1( np.dot(np.kron(np.eye(n), p.array.T), self.array), n )
        return Predicate(out, self.dom)
    
    def __rshift__(self, s):
        """ forward state transformation """
        if s.dom != self.dom:
            raise Exception('Non-match in state transformation')
        n = s.dom.size # = self.dom.size
        m = self.cod.size
        out = tr2( np.dot(np.kron(s.array, np.eye(m)), self.array), m ).T
        return State(out, self.cod)

    def as_channel(self):
        """ turn operator into channel """
        n = self.dom.size
        m = self.cod.size
        mat = np.zeros((m,m,n,n)) + 0j
        for i in range(n):
            for j in range(n):
                for k in range(m):
                    for l in range(m):
                        mat[k][l][i][j] = self.array[i*m+k][j*m+l]
        return Channel(mat, self.dom, self.cod)

    def as_kraus(self):
        """ produce Kraus operators associated with operator """
        out = kraus_decomposition(self.array, self.dom.size, self.cod.size)
        return Kraus(out, self.dom, self.cod)



#
# Kraus operators associated with a channel
#
# Kraus representation of a channel c : n -> m as a list of nxm
# matrices. The key propertie are:
#
#  c << p   equals
#     sum([np.dot(np.dot(e.T, p.array), e) for e in kraus(c)])
#
#  c >> s   equals
#     sum([np.dot(np.dot(e, s.array), e.T) for e in kraus(c)])
#
# Notice the occurrences of transpose .T in different places.
#
class Kraus:
    def __init__(self, mat_list, dom, cod):
        #print(mat.shape, dom_dims, cod_dims)
        self.array_list = mat_list
        self.dom = dom
        self.cod = cod

    # backward predicate transformation
    def __lshift__(self, p):
        if p.dom != self.cod:
            raise Exception('Non-match in predicate transformation')
        out = sum([np.dot(np.dot(e.T, p.array), e) 
                   for e in self.array_list])
        return Predicate(out, self.dom)

    # forward state transformation
    def __rshift__(self, s):
        if s.dom != self.dom:
            raise Exception('Non-match in state transformation')
        out = sum([np.dot(np.dot(e, s.array), e.T) 
                   for e in self.array_list])
        return State(out, self.cod)

    def as_channel(self):
        n = self.dom.size
        m = self.cod.size
        mat = np.zeros((m,m,n,n)) + 0j
        for k in range(m):
            for l in range(m):
                mat[k][l] = sum([np.dot(np.dot(conjugate_transpose(e), 
                                               matrix_base(k,l,m)), e) 
                                 for e in self.array_list])
        return Channel(mat, self.dom, self.cod)

########################################################################
# 
# Functions for state, predicate, and channel
#
########################################################################

#
# Trivial state of type []
#
init_state = State(np.ones((1,1)), [])

#
# Pure state from vector v via outer product |v><v|
#
def vector_state(*ls):
    if len(ls) == 0:
        raise Exception('Vector state creation requires a non-empty lsit')
    v = np.array(ls)
    s = np.linalg.norm(v)
    v = v / s
    mat = np.outer(v, np.conjugate(v))
    return State(mat, [len(v)])

#
# Predicate from vector v via outer product |v><v|
#
def vector_pred(*ls):
    if len(ls) == 0:
        raise Exception('Vector state creation requires a non-empty lsit')
    v = np.array(ls)
    s = np.linalg.norm(v)
    v = v / s
    mat = np.outer(v, np.conjugate(v))
    return Predicate(mat, [len(v)])

#
# Computational unit state |i><i| of dimension n
#
def point_state(i, n):
    if i < 0 or i >= n:
        raise Exception('Index out-of-range in unit state creation')
    return vector_state(*vector_base(i,n))

#
# ket state, taking 0's and 1's as input, as in ket(0,1,1) for |011>
#
def ket(*ls):
    n = len(ls)
    if n == 0:
        raise Exception('Empty ket is impossible')
    if n == 1:
        return point_state(ls[0], 2)
    return point_state(ls[0], 2) @ ket(*ls[1:n])

#
# A probabilistic state constructed from an n-tuple of positive
# numbers. These numbers are normalised and put on the diagonal in the
# resulting density matrix.
#
def probabilistic_state(*ls):
    n = len(ls)
    s = sum(ls)
    mat = np.zeros((n,n))
    for i in range(n):
        mat[i,i] = ls[i]/s
    return State(mat, [n])

#
# The uniform probabilistic state of size n, with probability 1/n on
# the diagonal in the resulting density matrix.
#
def uniform_probabilistic_state(dom):
    dom = dom if isinstance(dom, Dom) else Dom(dom)
    n = dom.size
    mat = np.zeros((n,n))
    for i in range(n):
        mat[i,i] = 1/n
    return State(mat, dom)

#
# A random vector state of size n, using Python's random number
# generator. These states are useful for testing.
#
def random_vector_state(n):
    return vector_state(*[complex(random.uniform(-10.0, 10.0),
                                  random.uniform(-10.0, 10.0))
                          for i in range(n)])

#
# A random state of size n, using Python's random number
# generator. These states are useful for testing.
#
def random_state(dom):
    # alternative use numpy.random and A*.A/trace(A*.A)
    # for predicates use A*.A / max (eigenvalue (A*.A))
    # A = np.random.rand(n,n)
    # B = np.random.rand(n,n)
    # C = np.zeros((n,n)) + 0j
    # for i in range(n):
    #     for j in range(n):
    #         C[i,j] = complex(A[i,j], B[i,j])
    # D = np.dot(C, conjugate_transpose(C))
    # D = (1/np.trace(D).real) * D
    # print(np.trace(D), is_positive(D))
    dom = dom if isinstance(dom, Dom) else Dom(dom)
    n = dom.size
    ls = [vector_state(*[complex(random.uniform(-10.0, 10.0),
                                 random.uniform(-10.0, 10.0))
                         for i in range(n)]) 
          for j in range(n)]
    amps = [random.uniform(0.0, 1.0) for i in range(n)]
    s = sum(amps)
    mat = sum([amps[i]/s * ls[i].array for i in range(n)])
    return State(mat, dom)

#
# A random probabilistic state with domain dom, given by n = dom.size
# probabilities that add up to one on the diagonal of the resulting
# density matrix.
#
def random_probabilistic_state(dom):
    dom = dom if isinstance(dom, Dom) else Dom(dom)
    n = dom.size
    ls = [random.uniform(0.0, 1.0) for i in range(n)]
    s = sum(ls)
    mat = np.zeros((n,n))
    for i in range(n):
        mat[i,i] = ls[i]/s
    return State(mat, dom)

#
# A random probabilistic channel dom --> cod
#
def random_probabilistic_channel(dom, cod):
    dom = dom if isinstance(dom, Dom) else Dom(dom)
    cod = cod if isinstance(cod, Dom) else Dom(cod)
    n = dom.size
    m = cod.size
    rand = np.zeros((n,m))
    for i in range(n):
        ls = [random.uniform(0.0, 1.0) for j in range(m)]
        s = sum(ls)
        rand[i,...] = 1/s * np.array(ls)
    mat = np.zeros((m,m,n,n))
    for j in range(m):
        for i in range(n):
            mat[j][j][i][i] = rand[i][j]
    return Channel(mat, dom, cod)

#
# 0 <= theta <= pi, 0 <= phi <= 2*pi
#
def bloch_vector(theta, phi):
    return np.array([math.cos(theta/2), 
                     math.sin(theta/2) * math.e ** (phi * complex(0,1))])

def bloch_state(theta, phi):
    return vector_state(*bloch_vector(theta, phi))

#
# Truth predicate, for arbitrary dims
#
def truth(dom):
    dom = dom if isinstance(dom, Dom) else Dom(dom)
    return Predicate(np.eye(dom.size), dom)

def falsity(dom):
    return ~truth(dom)

def probabilistic_pred(*ls):
    n = len(ls)
    if n == 0:
        raise Exception('A non-empty list of numbers is required for a probabilistic predicate')
    if any([r < 0 or r > 1 for r in ls]):
        raise Exception('Probabilities cannot exceed 1 for a probabilistic predicate')
    return Predicate(np.diag(ls), [n])

def point_pred(i, n):
    return probabilistic_pred(*vector_base(i,n))

yes_pred = point_pred(0,2)
no_pred = point_pred(1,2)

#
# A random probabilitisc predicate of dimension n
#
def random_probabilistic_pred(dom):
    dom = dom if isinstance(dom, Dom) else Dom(dom)
    n = dom.size
    mat = np.zeros((n,n))
    for i in range(n):
        mat[i,i] = random.uniform(0.0, 1.0)
    return Predicate(mat, dom)

def random_pred(dom):
    dom = dom if isinstance(dom, Dom) else Dom(dom)
    n = dom.size
    ls = [vector_state(*[complex(random.uniform(-10.0, 10.0),
                                 random.uniform(-10.0, 10.0))
                         for i in range(n)])
          for j in range(n)]
    amps = [random.uniform(0.0, 1.0) for i in range(n)]
    mat = sum([amps[i] * ls[i].array for i in range(n)])
    E = np.linalg.eigvals(mat)
    m = max([x.real for x in E])
    return Predicate(mat/m, dom)

def random_randvar(n):
    ar = complex(20, 0) * (np.random.rand(n,n) - 0.5)
    ai = complex(0, 20) * (np.random.rand(n,n) - 0.5)
    a = ar + ai
    return RandVar(a + conjugate_transpose(a), [n])

#
# Choi matrix n x n of n x n matrices, obtained from m x n matrix u,
# by forming putting u * E_ij * u^*, where E_ij is |i><j|, at position
# (i,j). It satisfies choi(A @ B) = choi(A) @ choi(B), where @ is
# tensor
#
# http://mattleifer.info/2011/08/01/the-choi-jamiolkowski-isomorphism-youre-doing-it-wrong/
#
# Note: np.dot(U,V) is first V then U, so equals UV = U*V as matrix product
#
def choi(u):
    m = u.shape[0] # columns
    n = u.shape[1] # rows, so u is m x n matrix
    mat = np.zeros((m,m,n,n)) + 0j
    for i in range(m):
        for j in range(m):
            out = np.dot(conjugate_transpose(u), np.dot(matrix_base(i,j,m), u))
            mat[i,j] = out
    return mat

    
#
# Channel obtained from an isometry u. The key properties are:
# 
#   chan(u) >> s  =  conj_trans(u) * s.array * u
#
#   chan(u) << p  =  u * p.array * conj_trans(u)
#
# The domain and codomain are given as parameters, since they may
# be of the form [2,2] instead of [4], when u has 4 rows or columns,
# see for instance the cnot or swap channel below.
#
def channel_from_isometry(u, dom, cod):
    if not is_isometry(u):
        raise Exception('Isometry required for channel construction')
    return Channel(choi(u), dom, cod)



########################################################################
# 
# Concrete states
#
########################################################################

#
# Classical coin flip, for probability r in unit interval [0,1]
#
def cflip(r):
    if r < 0 or r > 1:
        raise Exception('Coin flip requires a number in the unit interval')
    return State(np.array([[r, 0],
                           [0, 1 - r]]), [2])

cfflip = cflip(0.5)

#
# Convex sum of states: the input list contains pairs (ri, si) where
# the ri are in [0,1] and add up to 1, and the si are states
#
def convex_state_sum(*ls):
    if len(ls) == 0:
        raise Exception('Convex sum cannot be empty')
    dom = ls[0][1].dom
    if any([s.dom != dom for r,s in ls]):
        raise Exception('Convex sum requires that states have the same dimensions')
    if any([r < 0 or r > 1 for (r,s) in ls]):
        raise Exception('Convex sum requires numbers in the unit interval')
    r = sum([r for r,s in ls])
    if not approx_eq_num(r, 1):
        raise Exception('Scalars must add up to 1 in convex sum')
    return State(sum([r * s.array for r,s in ls]), dom)


#
# identity channel dom -> dom
#
def idn(dom):
    dom = dom if isinstance(dom, Dom) else Dom(dom)
    n = dom.size
    return channel_from_isometry(np.eye(n), dom, dom)

#
# unique channel discard : dom -> []
#
def discard(dom):
    dom = dom if isinstance(dom, Dom) else Dom(dom)
    n = dom.size
    mat = np.eye(n)
    mat.resize((1,1,n,n))
    return Channel(mat, dom, [])

#
# Channel dims -> dims that only keeps classical part, by measuring in
# the standard basis
#
def classic(dom):
    dom = dom if isinstance(dom, Dom) else Dom(dom)
    n = dom.size
    mat = np.zeros((n,n,n,n))
    for i in range(n):
        tmp = np.zeros((n,n))
        tmp[i][i] = 1
        mat[i][i] = tmp
    return Channel(mat, dom, dom)

# def classic(*dims):
#     if len(dims) == 0:
#         raise Exception('Classic channel requires non-empty list of dimensions')
#     n = dims[0]
#     mat = np.zeros((n,n,n,n))
#     for i in range(n):
#         tmp = np.zeros((n,n))
#         tmp[i][i] = 1
#         mat[i][i] = tmp
#     ch = Channel(mat, [n], [n])
#     if len(dims) == 1:
#         return ch
#     return ch @ classic(*dims[1:])

#
# Injection channel kappa(m,k,n) : n -> m @ n for k below m with main
# property:
#
#   kappa(m,k,n) >> s  =  point_state(k,m) @ s
#
def kappa(m,k,n):
    mat = np.zeros((n*m,n*m,n,n)) + 0j
    ar = idn([n]).array
    for i in range(n):
        for j in range(n):
            mat[k*n+i][k*n+j] = ar[i][j]
    return Channel(mat, [n], [m,n])

#
# copy : n -> m @ n channel with as main property:
#
#   copy(m,n) >> s  =  uniform_probabilistic_state(m) @ s
#
#   copy(m,n) << truth(m) @ p  =  p )
#
def copy(m,n):
    mat = np.zeros((n*m,n*m,n,n)) + 0j
    ar = 1.0/m * idn([n]).array
    for k in range(m):
        for i in range(n):
            for j in range(n):
                mat[k*n+i][k*n+j] = ar[i][j]
    return Channel(mat, [n], [m,n])


#
# swap channel 2 @ 2 -> 2 @ 2
#
swap = channel_from_isometry(np.array([ [1, 0, 0, 0],
                                       [0, 0, 1, 0],
                                       [0, 1, 0, 0],
                                       [0, 0, 0, 1] ]), 
                            [2,2], [2,2])

#
# swap channel 2 @ 3 -> 3 @ 2
#
swap23 = channel_from_isometry(np.array([ [1, 0, 0, 0, 0, 0],
                                         [0, 0, 1, 0, 0, 0],
                                         [0, 0, 0, 0, 1, 0],
                                         [0, 1, 0, 0, 0, 0],
                                         [0, 0, 0, 1, 0, 0],
                                         [0, 0, 0, 0, 0, 1] ]), 
                            [2,3], [3,2])
#
# swap channel 3 @ 2 -> 2 @ 3
#
swap32 = channel_from_isometry(np.array([ [1, 0, 0, 0, 0, 0],
                                         [0, 0, 0, 1, 0, 0],
                                         [0, 1, 0, 0, 0, 0],
                                         [0, 0, 0, 0, 1, 0],
                                         [0, 0, 1, 0, 0, 0],
                                         [0, 0, 0, 0, 0, 1] ]), 
                            [3,2], [2,3])

def swaps(n,m):
    mat = np.zeros((n*m, n*m))
    for i in range(n):
        for j in range(m):
            ar1 = np.zeros(n)
            ar2 = np.zeros(m)
            ar1[i] = 1
            ar2[j] = 1
            mat[i*m+j] = np.kron(ar2, ar1)
    return channel_from_isometry(mat, [n,m], [m,n])

# #
# # first projection channel 2 @ 2 -> 2
# #
# proj1 = Channel(np.array([[np.array([[1,0,0,0], 
#                                      [0,1,0,0],
#                                      [0,0,0,0],
#                                      [0,0,0,0]]), np.array([[0,0,0,0], 
#                                                             [0,0,0,0],
#                                                             [1,0,0,0],
#                                                             [0,1,0,0]])], 
#                           [np.array([[0,0,1,0], 
#                                      [0,0,0,1],
#                                      [0,0,0,0],
#                                      [0,0,0,0]]), np.array([[0,0,0,0], 
#                                                             [0,0,0,0],
#                                                             [0,0,1,0],
#                                                             [0,0,0,1]])]]),
#                Dom([2, 2]), Dom([2]))

# #
# # second projection channel 2 @ 2 -> 2
# #
# proj2 = Channel(np.array([[np.array([[1,0,0,0], 
#                                      [0,0,0,0],
#                                      [0,0,1,0],
#                                      [0,0,0,0]]), np.array([[0,0,0,0], 
#                                                             [1,0,0,0],
#                                                             [0,0,0,0],
#                                                             [0,0,1,0]])], 
#                           [np.array([[0,1,0,0], 
#                                      [0,0,0,0],
#                                      [0,0,0,1],
#                                      [0,0,0,0]]), np.array([[0,0,0,0], 
#                                                             [0,1,0,0],
#                                                             [0,0,0,0],
#                                                             [0,0,0,1]])]]),
#                Dom([2, 2]), Dom([2]))


#
# Kronecker channel n @ m -> n*m
#
def kron(n,m):
    return Channel(choi(np.eye(n*m)), [n, m], [n*m])

#
# Kronecker inverse channel n*m -> n @ m
#
def kron_inv(n,m):
    return Channel(choi(np.eye(n*m)), [n*m], [n, m])

#
# Auxiliary function, placing a matrix in the lower-right corner of a
# new matrix that is twice as big, with zeros everywhere else.
#
# This is: |1><1| x mat,  where x is Kronecker
#
def lower_right(mat):
    # mat is assumed to be square
    return np.kron(np.array([[0,0],[0,1]]), mat)

#
# Same as before, except that ones are put on the upper left diagonal;
# this is used for conditional channels from gates.
#
# This is: |0><0| x id + |1><1| x mat,  where x is Kronecker
#
def lower_right_one(mat):
    # mat is assumed to be square
    n = mat.shape[0]
    return np.kron(np.array([[1,0],[0,0]]), np.eye(n)) \
        + np.kron(np.array([[0,0],[0,1]]), mat)

x_matrix = np.array([[0,1],
                     [1,0]])

#
# Pauli-X channel 2 -> 2
#
x_chan = channel_from_isometry(x_matrix, [2], [2])

#
# Eigen vectors of x_matrix and test
#
x_plus = vector_state(-1/math.sqrt(2), 1/math.sqrt(2))
x_min = vector_state(1/math.sqrt(2), 1/math.sqrt(2))
x_pred = x_plus.as_pred()
x_test = [x_pred, ~x_pred]

#
# cnot channel 2 @ 2 -> 2 @ 2
#
# This should be the same as quantum-control(x_chan)
#
cnot = channel_from_isometry(lower_right_one(x_matrix), [2, 2], [2, 2])


y_matrix = np.array([[0,-complex(0, 1)],
                     [complex(0,1),0]])

#
# Pauli-Y channel 2 -> 2
#
y_chan = channel_from_isometry(y_matrix, [2], [2])


#
# Test given by eigen vectors of y_matrix
#
y_plus = vector_state(1/math.sqrt(2), complex(0,-1)/math.sqrt(2))
y_min = vector_state(1/math.sqrt(2), complex(0,1)/math.sqrt(2))
y_pred = y_plus.as_pred()
y_test = [ y_pred, ~y_pred ]


z_matrix = np.array([[1,0],
                     [0,-1]])

#
# Pauli-Z channel 2 -> 2
#
z_chan = channel_from_isometry(z_matrix, [2], [2])


#
# Test given by eigen vectors of z_matrix
#
z_pred = point_pred(0,2)
z_test = [z_pred, ~z_pred]


hadamard_matrix = (1/math.sqrt(2)) * np.array([ [1, 1],
                                                [1, -1] ])

#
# Hadamard channel 2 -> 2
#
hadamard = channel_from_isometry(hadamard_matrix, [2], [2])

#
# Basic states, commonly written as |+> and |->
#
plus = hadamard >> ket(0)
minus = hadamard >> ket(1)

hadamard_test = [plus.as_pred(), minus.as_pred()]

#
# Controlled Hadamard 2 @ 2 -> 2 @ 2
#
chadamard = channel_from_isometry(lower_right_one(hadamard_matrix), 
                                 [2, 2], [2, 2])

#
# channel 2 @ 2 -> 2 @ 2 for producing Bell states
#
bell_chan = cnot * (hadamard @ idn([2]))
# bell00 = bell_chan >> ket(0,0)
# bell01 = bell_chan >> ket(0,1)
# bell10 = bell_chan >> ket(1,0)
# bell11 = bell_chan >> ket(1,1)

bell00_vect = np.array([1,0,0,1])
bell01_vect = np.array([0,1,1,0])
bell10_vect = np.array([1,0,0,-1])
bell11_vect = np.array([0,1,-1,0])

bell00 = State(0.5 * np.outer(bell00_vect, bell00_vect), [2,2])
bell01 = State(0.5 * np.outer(bell01_vect, bell01_vect), [2,2])
bell10 = State(0.5 * np.outer(bell10_vect, bell10_vect), [2,2])
bell11 = State(0.5 * np.outer(bell11_vect, bell11_vect), [2,2])

bell_test = [bell00.as_pred(),
             bell01.as_pred(),
             bell10.as_pred(),
             bell11.as_pred()]

#
# Greenberger-Horne-Zeilinger states
#
ghz = (idn([2]) @ cnot) >> ((bell_chan @ idn([2])) >> ket(0,0,0))

# The ghz states one by one

ghz_vect1 = np.array([1,0,0,0,0,0,0,1])
ghz_vect2 = np.array([1,0,0,0,0,0,0,-1])
ghz_vect3 = np.array([0,0,0,1,1,0,0,0])
ghz_vect4 = np.array([0,0,0,1,-1,0,0,0])
ghz_vect5 = np.array([0,0,1,0,0,1,0,0])
ghz_vect6 = np.array([0,0,1,0,0,-1,0,0])
ghz_vect7 = np.array([0,1,0,0,0,0,1,0])
ghz_vect8 = np.array([0,1,0,0,0,0,-1,0])

ghz1 = State(0.5 * np.outer(ghz_vect1, ghz_vect1), [2,2,2])
ghz2 = State(0.5 * np.outer(ghz_vect2, ghz_vect2), [2,2,2])
ghz3 = State(0.5 * np.outer(ghz_vect3, ghz_vect3), [2,2,2])
ghz4 = State(0.5 * np.outer(ghz_vect4, ghz_vect4), [2,2,2])
ghz5 = State(0.5 * np.outer(ghz_vect5, ghz_vect5), [2,2,2])
ghz6 = State(0.5 * np.outer(ghz_vect6, ghz_vect6), [2,2,2])
ghz7 = State(0.5 * np.outer(ghz_vect7, ghz_vect7), [2,2,2])
ghz8 = State(0.5 * np.outer(ghz_vect8, ghz_vect8), [2,2,2])

#
# The associated test
#
ghz_test = [ghz1.as_pred(),
            ghz2.as_pred(),
            ghz3.as_pred(),
            ghz4.as_pred(),
            ghz5.as_pred(),
            ghz6.as_pred(),
            ghz7.as_pred(),
            ghz8.as_pred()]

#
# W3 state
#
w3_vect = np.array([0,1,1,0,1,0,0,0])
w3 = State(1/3 * np.outer(w3_vect, w3_vect), [2,2,2])
           
def phase_shift_matrix(angle):
    return np.array([[1, 0],
                     [0, complex(math.cos(angle), math.sin(angle))]])

#
# Phase shift channel 2 -> 2, for angle between 0 and 2 pi
#
def phase_shift(angle):
    return channel_from_isometry(phase_shift_matrix(angle), [2], [2])

#
# Controlled phase shift channel 2 @ 2 -> 2 @ 2, for angle between 0 and 2 pi
#
def cphase_shift(angle):
    return channel_from_isometry(lower_right_one(phase_shift_matrix(angle)),
                                [2, 2], [2, 2])



#
# toffoli channel 2 @ 2 @ 2 -> 2 @ 2 @ 2
#
toffoli = channel_from_isometry(np.array([ [1, 0, 0, 0, 0, 0, 0, 0],
                                           [0, 1, 0, 0, 0, 0, 0, 0],
                                           [0, 0, 1, 0, 0, 0, 0, 0],
                                           [0, 0, 0, 1, 0, 0, 0, 0],
                                           [0, 0, 0, 0, 1, 0, 0, 0],
                                           [0, 0, 0, 0, 0, 1, 0, 0],
                                           [0, 0, 0, 0, 0, 0, 0, 1],
                                           [0, 0, 0, 0, 0, 0, 1, 0] ]),
                                [2, 2, 2], [2, 2, 2])



#
# Convex sum of channels: the input list contains pairs (ri, ci) where
# the ri are in [0,1] and add up to 1, and the ci are channels
#
def convex_channel_sum(*ls):
    if len(ls) == 0:
        raise Exception('Convex sum cannot be empty')
    dom_dims = ls[0][1].dom_dims
    cod_dims = ls[0][1].cod_dims
    if any([c.dom_dims != dom_dims or c.cod_dims != cod_dims for r,c in ls]):
        raise Exception('Convex sum requires parallel channels')
    if any([r < 0 or r > 1 for (r,c) in ls]):
        raise Exception('Convex sum requires numbers in the unit interval')
    r = sum([r for r,c in ls])
    if not approx_eq_num(r, 1):
        raise Exception('Scalars must add up to 1 in convex sum')
    return Channel(sum([r * c.array for r,c in ls]), dom_dims, cod_dims)

#
# Channel constructed from a list of states.
#
# Let c = channel_from_states(s1, ..., sn). Then c >> t equals the
# convex sum of states, given by the pairs 
#
#        ( t >= point_pred(i,n), si )
#
# This means that much information about t is lost.
#
# How this channel works on predicates is not clear.
#
def channel_from_states(*ls):
    n = len(ls)
    if n == 0:
        raise Exception('Non-empty list of state is need to form a channel')
    cod = ls[0].dom
    if any([s.dom != cod for s in ls]):
        raise Exception('States must all have the same type to form a channel')
    mat = np.zeros((cod.size, cod.size, n, n)) + 0j
    for j1 in range(cod.size):
        for j2 in range(cod.size):
            for i in range(n):
                mat[j1][j2][i][i] = ls[i].array[j1][j2]
    return Channel(mat, [n], cod)


#
# Classical conditional probability table converted into a
# channel. The input is a list of probabilities, of length 2^n, where
# n is the number of predecessor nodes.
#
def ccpt(*ls):
    n = len(ls)
    if n == 0:
        raise Exception('Conditional probability table must have non-empty list of probabilities')
    log = math.log(n, 2)
    if log != math.floor(log):
        raise Exception('Conditional probability table must have 2^n elements')
    log = int(log)
    mat = np.zeros((2,2,n,n))
    for i in range(n):
        mat[0][0][i][i] = ls[i]
        mat[1][1][i][i] = 1-ls[i]
    return Channel(mat, [2] * log, [2])




########################################################################
# 
# Measurement and control
#
########################################################################


#
# Measurement channel dom -> 2, for a predicate p of type dom. This
# channel does not keep a record of the updated state.
#
# The key property is, for a state s of type dom:
#
#   meas_pred(p) >> s  =  s >= p
#
# where the right-hand-side must be interpreted as a classic state
# with domain [2].
#
def meas_pred(p):
    n = p.dom.size
    mat = np.zeros((2,2,n,n)) + 0j
    mat[0][0] = p.array
    mat[1][1] = (~p).array
    return Channel(mat, p.dom, [2])

meas0 = meas_pred(point_pred(0,2))
meas1 = meas_pred(point_pred(1,2))

#
# Measurement generalised from a predicate to a test, that is to a
# list of predicates that add up to truth. Measurement wrt. a
# predicate p is the same as measurement wrt. the test [p, ~p]
#
# The l predicates in the test must all have the same domain dom. The
# measurement channel then has time dom -> l
#
def meas_test(ts):
    l = len(ts)
    if l == 0:
        raise Exception('Test must have non-zero length in measurement')
    dom = ts[0].dom
    if any([t.dom != dom for t in ts]):
        raise Exception('Tests must have the same domain in measurement')
    t = ts[0]
    for i in range(l-1):
        t = t + ts[i+1]
    if not np.all(np.isclose(t.array, truth(dom).array)):
        raise Exception('The predicates in a test must add up to truth')
    mat = np.zeros((l,l,dom.size,dom.size)) + 0j
    for i in range(l):
        mat[i][i] = ts[i].array
    return Channel(mat, dom, [l])

#
# Measurements in some standard bases.
#
meas_hadamard = meas_test(hadamard_test)
meas_bell = meas_test(bell_test)
meas_ghz = meas_test(ghz_test)

#
# Instrument dom -> [2] @ dom , for a predicate p of type dom.
#
# The main properties are:
#
#   (instr(p) >> s) % [1,0]  =  meas_pred(p) >> s,
#
#   (instr(p) >> s) % [0,1]  =  convex_state_sum( (s >= p, s/p), 
#                                                 (s >= ~p, s/~p) )
#
#   instr(p) << truth(2) @ q  =  (p & q) + (~p & q) 
#   instr(p) << point_pred(0,2) @ q  =  p & q
#   instr(p) << point_pred(1,2) @ q  =  ~p & q 
#
def instr(p):
    n = p.dom.size
    mat = np.zeros((2*n,2*n,n,n)) + 0j
    sqp = matrix_square_root(p.array)
    sqnp = matrix_square_root((~p).array)
    for i in range(n):
        for j in range(n):
            arg = matrix_base(i,j,n)
            out1 = np.dot(sqp, np.dot(arg, sqp))
            out2 = np.dot(sqnp, np.dot(arg, sqnp))
            mat[i][j] = out1
            mat[n+i][n+j] = out2
    return Channel(mat, p.dom, Dom([2]) + p.dom)


def pcase(p):
    def fun(*chan_pair):
        c = chan_pair[0]
        d = chan_pair[1]
        if c.dom != d.dom or c.cod != d.cod:
            raise Exception('channels must have equal domain and codomain in predicate case channel')
        return (discard([2]) @ idn(c.dom)) * ccase(c,d) * instr(p)
    return fun


#
# classical control of a channel c : dom -> cod, giving a channel
# ccontrol(c) : [2] + dom -> [2] + cod
#
def ccontrol(c):
    cd = c.dom.size
    cc = c.cod.size
    shape = [2 + cc, 2 + cc, 2 + cd, 2 + cd]
    mat = np.zeros(shape) + 0j
    #print(cd, cc, c.array.shape, mat.shape)
    for i in range(cd):
        for j in range(cd):
            mat[i][j][i][j] = 1
            mat[cd+i][cd+j] = lower_right(c.array[i][j])
            #print("control submatrix", i, j)
            #print(c.array[i][j])
            #for k in range(cd):
                #print(i, j, k)
                # the next formulation is inexplicably close for hadamard
                # mat[i][cd+j][i][cd+k] = cmath.sqrt(c.array[i][j][i][j]) 
                # mat[cd+i][j][cd+k][j] = cmath.sqrt(c.array[i][j][i][j]) 
                ## mat[i][cd+j][i][cd+k] = 1-k
                # this one has no effect for cnot
                ## mat[cd+i][j][cd+k][j] = 1-k
            # this one works for cnot: 
            # mat[i][cd+j][i][2*cd - 1 - j] = 1
            # mat[cd+i][j][2*cd - 1 - i][j] = 1
    return Channel(mat, Dom([2]) + c.dom, Dom([2]) + c.cod)

#
# A list of channels c1, ..., ca all with the same domain dom and
# codomain cod gives a classical case channel a @ dom -> a @ cod
#
def ccase(*chans):
    a = len(chans)
    if a == 0:
        raise Exception('Non-empty channel list is required in control')
    dom = chans[0].dom
    cod = chans[0].cod
    if any([c.dom != dom or c.cod != cod for c in chans]):
        raise Exception('Channels all need to have the same (co)domain in control')
    mat = np.zeros((a*cod.size, a*cod.size, a*dom.size, a*dom.size)) + 0j
    for b in range(a):
        for k in range(cod.size):
            for l in range(cod.size):
                for i in range(dom.size):
                    for j in range(dom.size):
                        mat[b*cod.size + k][b*cod.size + l] \
                            [b*dom.size + i][b*dom.size + j] \
                            = chans[b].array[k][l][i][j]
    return Channel(mat, Dom([a]) + dom, Dom([a]) + cod)



########################################################################
# 
# Distance between states, entropy, mutual information
#
########################################################################

#
# Trace distance
#
def trdist(s,t):
    diff = s.array - t.array
    prod = np.dot(diff, conjugate_transpose(diff))
    return 0.5 * np.trace(matrix_square_root(prod))

def shannon_entropy(s):
    """Shannon entropy"""
    eigen_values = np.linalg.eigh(s.array)[0]
    def f(x): return -math.log2(x)*x if x != 0 else 0
    return sum(np.vectorize(f)(eigen_values))

def mutual_information(js):
    n = len(js.dom.dims)
    if n < 2:
        raise Exception('Mutual information is defined only for joint states')
    selectors = []
    for i in range(n):
        ls = [0] * n
        ls[i] = 1
        selectors = selectors + [ls]
    marginals = [ js % sel for sel in selectors ]
    return sum(np.vectorize(shannon_entropy)(marginals)) - shannon_entropy(js)


########################################################################
# 
# Cup-cap channels; experimental
#
########################################################################


#
# unitary
#
def cup_chan(dom):
    dom = dom if isinstance(dom, Dom) else Dom(dom)
    n = dom.size
    ls = [1]
    for i in range(n-1):
        ls = ls + n*[0] + [1]
    v = np.array(ls)
    mat = np.zeros((n*n,n*n,1,1))
    mat[...,0,0] = np.outer(v.transpose(), v)
    return Channel(1/n * mat, [], dom + dom)

#
# non-unitary version
#
def cup_map(dom):
    dom = dom if isinstance(dom, Dom) else Dom(dom)
    n = dom.size
    return Channel(n * cup_chan(dom).array, [], dom + dom)

def cup_state(dom):
    return cup_chan(dom) >> init_state

def cup(dom):
    return cup_map(dom) >> init_state

#
# |v> = sum_{i} |ii>, giving matrix |v><v|, as in Leifer-Spekkens
#
def cap_chan(dom):
    dom = dom if isinstance(dom, Dom) else Dom(dom)
    n = dom.size
    ls = [1]
    for i in range(n-1):
        ls = ls + n*[0] + [1]
    v = np.array(ls)
    mat = np.zeros((1,1,n*n,n*n))
    mat[0][0] = np.outer(v.transpose(), v)
    return Channel(n * mat, dom * 2, [])

def cap_map(dom):
    dom = dom if isinstance(dom, Dom) else Dom(dom)
    n = dom.size
    return Channel(1/n * cap_chan(dom).array, dom * 2, [])

def cap_pred(dom):
    return cap_chan(dom) << truth([])

def cap(dom):
    return cap_map(dom) << truth([])


def channel_to_state(chan):
    return (idn(chan.dom) @ chan) * cup_chan(chan.dom) >> init_state

def state_to_channel(stat):
    n = stat.dom.dims[0]
    m = stat.dom.dims[1]
    return (cap_chan([n]) @ idn([m])) * (idn([n]) @ stat.as_chan())


def sqr_modifier(array, dom):
    n = array.shape[0]
    s = matrix_square_root(array)
    mat = np.zeros((n,n,n,n)) + 0j
    for i in range(n):
        for j in range(n):
            mat[i][j] = np.dot(s, np.dot(matrix_base(i,j,n), s))
    return Channel(mat, dom, dom)

def asrt(p):
    return sqr_modifier(p.array, p.dom)

def extract(stat):
    sqr_chan = sqr_modifier(1/(stat % [1,0]).dom.size * 
                            # adding transpose here yields a unital map
                            np.linalg.inv((stat % [1,0]).array).transpose(), 
                            (stat % [1,0]).dom)
    return state_to_channel(stat) * sqr_chan


########################################################################
# 
# Leifer-Spekkens style operations; experimental
#
########################################################################

#
# Turn channel n -> m into n -> n @ m
#
# Note: the definition is precisely the same as in the discrete case,
# but now we don't copy the state into the first coordinate, but
# measure it!
#
# In the second coordinate we keep the original channel:
#
#    (graph(c) >> t) % [0, 1]  =  c >> t
#
# But in the first coordinate:
#
#    (graph(c) >> t) % [1, 0]  = classic >> t
#
# Hence we have a probabilistic state with entries given by the validities:
#
#    t >= point_pred(i,n)
#
def graph(c):
    if len(c.dom.dims) != 1:
        raise Exception('Tupling not defined for product input ')
    n = c.dom.dims[0]
    m = c.cod.size
    mat = np.zeros((n*m,n*m,n,n)) + 0j
    for i in range(n):
        for k in range(m):
            for l in range(m):
                mat[i*m+k][i*m+l] = 1.0/n * c.array[k][l]
    return Channel(mat, c.dom, c.cod * n)


#
# Turn channel and state into joint state, whose first marginal is the
# original state
#
def graph_pair(s, c):
    a = np.kron(matrix_square_root(s.array), np.eye(c.cod.size))
    b = np.dot(a, np.dot(c.as_operator().array, a))
    return State(b, c.dom+c.cod)

#
# Turn joint state of type n @ m into state of type n and channel n -> m
#
def pair_extract(w):
    n = w.dom.dims[0]
    m = w.dom.dims[1]
    w1 = w % [1,0]
    w2 = np.kron(np.linalg.inv(matrix_square_root(w1.array)), np.eye(m))
    oper = Operator(np.dot(w2, np.dot(w.array, w2)), Dom([n]), Dom([m]))
    return (w1, oper.as_channel())


########################################################################
# 
# Transition operator related stuff
#
########################################################################


#
# Transition operator n*n x n*n associated with unitary u of n x n.
#
# The main property is UAU^* = tr2((A@id)transition_operator(U), n).T
#
def transition_from_unitary(u):
    n = u.shape[0]
    mat = np.ndarray((n*n,n*n)) + 0j
    for i in range(n):
        for j in range(n):
            for k in range(n):
                for l in range(n):
                    mat[n*i+k][n*j+l] = u[i][k] * u[j][l].conjugate()
    return mat


def operator_from_unitary(u, dom):
     if not is_unitary(u):
         raise Exception('Unitary matrix required for channel construction')
     return Operator(transition_from_unitary(u), dom, dom)


#
# Own partial trace implementations tr1 and tr2
#
# mat has shape (n*m, n*m); tr1 goes to (n,n) and tr2 to (m,m)
#
def tr1(mat,n):
    k = mat.shape[0]
    m = int(k/n) # remove this many
    out = np.zeros((n,n)) + 0j
    for j in range(m):
        v = np.array([0]*m)
        v[j] = 1
        w = np.kron(np.eye(n), v)
        out = out + np.dot(np.dot(w, mat), w.T)
    return out

def tr2(mat,m):
    k = mat.shape[0]
    n = int(k/m)  # remove this many
    out = np.zeros((m,m)) + 0j
    for i in range(n):
        v = np.array([0]*n)
        v[i] = 1
        w = np.kron(v, np.eye(m))
        out = out + np.dot(np.dot(w, mat), w.T)
    return out
    

# # tests at: http://www.thphy.uni-duesseldorf.de/~ls3/teaching/1515-QOQI/Additional/partial_trace.pdf
# M = np.array([[1, 2, complex(0,3), 4],
#               [5, 6, 7, 8],
#               [9, 10,11,12],
#               [13,complex(0,-14),15,16]])
# print( tr1(M, 2) )
# print( M.reshape(2,2,2,2).trace(axis1 = 1, axis2 = 3)  )
# print( tr2(M, 2) )
# print( M.reshape(2,2,2,2).trace(axis1 = 0, axis2 = 2)  )



#
# Pauli-X channel 2 -> 2
#
x_oper = operator_from_unitary(np.array([[0,1],
                                         [1,0]]), Dom([2]))

#
# Pauli-Y channel 2 -> 2
#
y_oper = operator_from_unitary(np.array([[0,-complex(0, 1)],
                                         [complex(0,1),0]]), Dom([2]))


#
# Pauli-Z channel 2 -> 2
#
z_oper = operator_from_unitary(np.array([[1,0],
                                         [0,-1]]), Dom([2]))


#
# Hadamard channel 2 -> 2
#
hadamard_oper = operator_from_unitary((1/math.sqrt(2)) * np.array([ [1, 1],
                                                                  [1, -1] ]),
                                      Dom([2]))

#
# Basic states, commonly written as |+> and |->
#
plus_oper = hadamard_oper >> ket(0)
minus_oper = hadamard_oper >> ket(1)


########################################################################
# 
# Generic plot function
#
########################################################################


def plot(f, lb, ub, steps = 100):
    fig, (ax) = plt.subplots(1, 1, figsize=(10,5))
    xs = np.linspace(lb, ub, steps, endpoint=True)
    ys = [f(x) for x in xs]
    plt.interactive(True)
    ax.plot(xs, ys, color="blue", linewidth=2.0, linestyle="-")
    plt.draw()
    plt.pause(0.001)
    input("Press [enter] to continue.")
    return None


########################################################################
# 
# Test functions
#
########################################################################


def validity():
    print("\nValidity tests")
    s1 = random_state(2)
    s2 = random_state(5)
    s3 = random_state(2)
    p1 = random_pred(2) 
    p2 = random_pred(5) 
    p3 = random_pred(2)
    print("* validity product difference test:", 
          (s1 @ s2 >= ~p1 @ (0.1 * p2)) - ((s1 >= 0.5 * ~p1) * (s2 >= 0.2 * p2)) )
    print("* transformation-validty difference test:", 
          (s1 @ s3 >= (chadamard << (p1 @ truth(2)))) \
          - ((chadamard >> s1 @ s3) >= (p1 @ truth(2))) )
    print("* weakening is the same as predicate transformation by a projection:", 
          p3 @ truth(2) == (idn([2]) @ discard([2])) << p3 )
    

def marginals():
    print("\nMarginal tests")
    print("* third marginal of |000> is:\n", ket(0,0,0) % [0,0,1])
    a = random_state(2)
    b = random_state(2)
    print("* a random product state, and then several projection operations on that state:", 
          a == (a @ b) % [1,0],
          a == idn([2]) >> a,
          a == idn([2]) @ discard([2]) >> (a @ b),
          a == discard([2]) @ idn([2]) >> (swap >> (a @ b)) )

def measurement():
    print("\nTests of predicates")
    print( ~x_pred == x_min.as_pred(),
           ~y_pred == y_min.as_pred() )
    print("\nXYZ non-commutation of tests")
    print( x_pred & y_pred != y_pred & x_pred,
           x_pred & z_pred != z_pred & x_pred,
           y_pred & z_pred != z_pred & y_pred )
    print("\nMeasurement and control tests")
    s = random_state(2)
    p = random_pred(2)
    q = random_pred(2)
    r = random_probabilistic_pred(2)
    print("* measurement channel applied to a state, with validity", 
           s >= p, "\n", meas_pred(p) >> s )
    print("cnot predicate transformation")
    print( (cnot << point_pred(0,2) @ q) == point_pred(0,2) @ q,
           (cnot << point_pred(1,2) @ r) == (point_pred(1,2) @ ~r),
           (cnot << point_pred(1,2) @ q) == (point_pred(1,2) @ ~q) )
    r = random.uniform(0,1)
    print("* Classical control with classical control bit:")
    print( (ccontrol(x_chan) >> cflip(r) @ s) % [1,0] == 
           probabilistic_state(r, 1-r),
           (ccontrol(x_chan) >> cflip(r) @ s) % [0,1] == 
           convex_state_sum( (r, s), (1-r, x_chan >> s) ) )
    print("* Classical case with classical control bit:")
    print( (ccase(y_chan,x_chan) >> cflip(r) @ s) % [1,0] == 
           probabilistic_state(r, 1-r),
           (ccase(y_chan,x_chan) >> cflip(r) @ s) % [0,1] == 
           convex_state_sum( (r, y_chan >> s), (1-r, x_chan >> s) ) )

def instrument():
    print("\nInstrument tests")
    p = random_pred(2)
    q = random_pred(2)
    s = random_state(2)
    print( (instr(p) >> s) % [1,0] == meas_pred(p) >> s,
           (instr(p) >> s) % [0,1] == convex_state_sum( (s >= p, s/p), 
                                                        (s >= ~p, s/~p) ),
           instr(p) << truth(2) @ q == (p & q) + (~p & q),
           instr(p) << point_pred(0,2) @ q == (p & q),
           instr(p) << point_pred(1,2) @ q == (~p & q) )
    print("channel equalities")
    print( (idn([2]) @ discard([2])) * instr(p) == meas_pred(p) )


def conditioning():
    print("\nConditioning tests")
    s = random_state(2)
    t = random_state(2)
    p = random_pred(2)
    q = random_pred(2)
    r = random_probabilistic_pred(2)
    t = random_probabilistic_pred(2)
    print("* Bayes difference of probabilities:",
          (s/p >= q) - ((s >= p & q) / (s >= p)) )
    print("* Conditioning is not an action:",
          s / truth(2) == s, 
          ((s / p) / q) == s / (p & q) )


def channel():
    print("\nChannel tests")
    s1 = random_state(3)
    s2 = random_state(3)
    s3 = random_state(3)
    t = random_state(3)
    c = channel_from_states(s1, s2, s3)
    print("* Swap tests")
    i = 4
    j = 2
    si = random_state([i])
    sj = random_state([j])
    print( swaps(i,j) * swaps(j,i) == idn(j,i),
           swaps(i,j) >> si @ sj == sj @ si )
    print( swap * swap == idn(2,2) )
    print("* channel from state state transformation as convex sum")
    print( convex_state_sum((t >= point_pred(0,3), s1),
                            (t >= point_pred(1,3), s2),
                            (t >= point_pred(2,3), s3)) )
    print( c >> t )
    print("* predicate as channel")
    p = random_pred(2)
    v = random_state(2)
    print( v >= p )
    print( p.as_chan() >> v )
    print("* discard channel; outcome is the identity matrix")
    print( discard([2]) * hadamard )
    print("* from product to channel")
    w1 = random_probabilistic_state(2)
    w2 = random_state(2)
    w = chadamard >> (w1 @ w2)
    print( w1 ==  w % [1, 0] )
    dc = productstate2channel(w)
    print( dc >> ket(0) )
    print( w2 )
    print( w2 == (chadamard >> (ket(0) @ w2)) % [0,1] )
    # the next two states are also equal
    print( dc >> ket(1) )
    print( (chadamard >> (ket(1) @ w2)) % [0,1] )

def bayesian_probability():
    print("\nBayesian probability")
    sens = ccpt(0.9, 0.05)
    prior = probabilistic_state(0.01, 0.99)
    print( sens >> prior )
    sense_inv = sens.inversion(prior)
    print( prior / (sens << point_pred(0,2)) == sense_inv >> ket(0),
           prior / (sens << point_pred(1,2)) == sense_inv >> ket(1) )
    print( sense_inv << truth(2) == truth(2),
           is_positive(sense_inv.as_operator().array) )

def kappa_copy():
    print("\nCoprojection and copy tests")
    s = random_state(4)
    p = random_pred(4)
    q = random_pred(2)
    print( kappa(3,2,4) >> s == point_state(2,3) @ s,
           kappa(3,1,4) << probabilistic_pred(0.3, 0.2, 0.5) @ p == 0.2 * p )
    c = hadamard * x_chan * z_chan
    d = y_chan * phase_shift(math.pi/3) * x_chan
    e = c * d
    print( ((discard([3]) @ idn(2)) * ccase(c, d, e) * kappa(3,0,2)) == c,
           ((discard([3]) @ idn(2)) * ccase(c, d, e) * kappa(3,1,2)) == d,
           ((discard([3]) @ idn(2)) * ccase(c, d, e) * kappa(3,2,2)) == e )
    # print ( np.isclose(((discard([2]) @ idn([2,2])) \
    #                     * ccase(ket(0).as_chan() @ idn([2]), 
    #                             ket(1).as_chan() @ x_chan)).array,
    #                    cnot.array) )
    print("* copy")
    print( copy(10,4) >> s == uniform_probabilistic_state(10) @ s,
           copy(6,5) << truth(6,5) == truth(5),
           copy(3, 2) << truth(3) @ q == q )
    print( copy(4,7) << truth(4, 7) == truth(7), 
           kappa(5,2,4) << truth(5,4) == truth(4) )
    print( ((discard([3]) @ idn(2)) * copy(3, 2)) == idn(2),
           ((idn(3) @ discard([2])) * copy(3, 2)) == 
           channel_from_states(uniform_probabilistic_state(3), 
                               uniform_probabilistic_state(3)) ) 
    t = random_state(2)
    print( (copy(3, 2) >> t) % [1,0] == uniform_probabilistic_state(3),
           (copy(3, 2) >> t) % [0,1] == t )

def graphs():
    print("\nGraph tests")
    c = x_chan * hadamard * phase_shift(math.pi/3)
    gr = (idn(2) @ c) * instr(point_pred(0,2))
    print("* truth preservation by channels")
    print( truth(2) == c << truth(2), truth(2) == gr << truth(2, 2) )
    # print("* graph properties")
    # print( (gr >> s) % [1, 0] )
    # print( classic(2) >> s )
    # print( classic(2) >> s == (gr >> s) % [1, 0] )
    # print( c >> s )
    # print( gr >> s )
    # print( (gr >> s) % [0, 1] )
    # print("* first component of tuple is go-classic:",
    #       (idn(2) @ discard(2)) * graph(c) == classic(2) )
    # print("* second component of tuple is the channel itself",
    #       (discard(2) @ idn(2)) * graph(c) == c )
    # print( np.isclose((((discard(2) @ idn(2)) * graph(c)) >> t).array,
    #                   (c >> t).array.T) )
    # dct = graph(productstate2channel(w))
    # print("* product from channel form product: recover the original:", 
    #       np.allclose(w.array, (dct >> w1).array))
    # u = random_state(2)
    # print("* channel from product from channel")
    # print( productstate2channel(graph(c) >> u) >> t )
    # print( c >> t )
    # print( np.allclose(c.array,
    #                    productstate2channel(graph(c) >> u).array) )


def transition():
    print("\nTransition tests")
    s = random_state(2)
    w = chadamard >> (s @ random_state(2))
    print("* transition state transformation is channel state transformation:")
    print( hadamard >> s == hadamard_oper >> s,
           x_chan >> s == x_oper >> s,
           y_chan >> s == y_oper >> s,
           z_chan >> s == z_oper >> s,
           (hadamard @ x_chan) >> w == (hadamard @ x_chan).as_operator() >> w )
    print("* transition predicate transformation is channel predicate transformation:")
    p = random_pred(2)
    q = kron(2,2) << random_pred(4)
    print( hadamard << p == hadamard_oper << p,
           x_chan << p == x_oper << p,
           y_chan << p == y_oper << p,
           z_chan << p == z_oper << p,
           (hadamard @ x_chan) << q == (hadamard @ x_chan).as_operator() << q )
    print("* Channel-to-operator conversions test")
    opr = hadamard_oper
    c = x_chan * hadamard * y_chan * z_chan
    print( c.as_operator().as_channel() == c,
           opr.as_channel().as_operator() == opr )
    print("* Leifer Spekkens")
    ls = graph_pair(s, c)
    print("graph_pair positive with trace: ", 
          is_positive(ls.array), np.trace(ls.array) )
    print( np.all(np.isclose((ls % [0,1]).array, (c >> s).array.T)) )
    print( pair_extract(ls)[0] == s, pair_extract(ls)[1] == c )
    w = chadamard >> (random_state(2) @ random_state(2))
    sp = pair_extract(w)
    print("pair extract equations: ",
          np.allclose((sp[1] >> sp[0]).array, (w % [0,1]).array.T), 
          graph_pair(sp[0], sp[1]) == w)
    c = chadamard * swap * cnot * swap
    p = cnot << (random_pred(2) @ random_pred(2))
    s = cnot >> (random_state(2) @ random_state(2))
    print("* Multidimensional Leifer Spekkes")
    print( graph_pair(s, c).dom )
    a = kron(2,2) << random_pred(4)
    b = kron(2,2) << random_pred(4)
    print( graph_pair(s,c) >= a @ b, s >= (c << a) & b )
    print("* Kraus test:",
          c << p == c.as_kraus() << p, c >> s == c.as_kraus() >> s )
    #print( tr1(c.as_operator().array, 4) )
    #print( tr2(c.as_operator().array, 4) )
    d = hadamard * x_chan
    print( np.allclose(d.array,
                       d.as_kraus().as_channel().array) )
    

def experiment():
    print("Experiments")
    # c = (chadamard * swap * cnot * swap) @ ket(0).as_chan()
    # print( c.dom, c.cod )
    # print( c.purify() )


def main():
    validity()
    # marginals()
    #measurement()
    # instrument()
    # conditioning()
    # channel()
    # bayesian_probability()
    # kappa_copy()
    # graphs()
    #transition()
    #experiment()


if __name__ == "__main__":
    main()


