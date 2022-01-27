
import numpy as np
import cirq
from typing import cast, Any
from cirq import linalg, protocols
from cirq._compat import proper_repr
from cirq.ops import gate_features

# Defining the Three Qubit Matrix Gate


def _phase_matrix(turns: float) -> np.ndarray:
    return np.diag([1, np.exp(2j * np.pi * turns)])


class SingleQubitMatrixGate(gate_features.SingleQubitGate):
    """A 1-qubit gate defined by its matrix.

    More general than specialized classes like `ZPowGate`, but more expensive
    and more float-error sensitive to work with (due to using
    eigendecompositions).
    """

    def __init__(self, matrix: np.ndarray) -> None:
        """
        Initializes the 2-qubit matrix gate.

        Args:
            matrix: The matrix that defines the gate.
        """
        if matrix.shape != (2, 2) or not linalg.is_unitary(matrix):
            raise ValueError('Not a 2x2 unitary matrix: {}'.format(matrix))
        self._matrix = matrix

    def validate_args(self, qubits):
        if len(qubits) != 1:
            raise ValueError(
                'Single-qubit gate applied to multiple qubits: {}({})'.format(
                    self, qubits))

    def __pow__(self, exponent: Any) -> 'SingleQubitMatrixGate':
        if not isinstance(exponent, (int, float)):
            return NotImplemented
        e = cast(float, exponent)
        new_mat = linalg.map_eigenvalues(self._matrix, lambda b: b**e)
        return SingleQubitMatrixGate(new_mat)

    def _trace_distance_bound_(self):
        vals = np.linalg.eigvals(self._matrix)
        rotation_angle = abs(np.angle(vals[0] / vals[1]))
        return rotation_angle * 1.2

    def _phase_by_(self, phase_turns: float, qubit_index: int):
        z = _phase_matrix(phase_turns)
        phased_matrix = z.dot(self._matrix).dot(np.conj(z.T))
        return SingleQubitMatrixGate(phased_matrix)

    def _has_unitary_(self) -> bool:
        return True

    def _unitary_(self) -> np.ndarray:
        return np.array(self._matrix)

    def _circuit_diagram_info_(self, args: protocols.CircuitDiagramInfoArgs
                               ) -> protocols.CircuitDiagramInfo:
        return protocols.CircuitDiagramInfo(
            wire_symbols=(_matrix_to_diagram_symbol(self._matrix, args),))

    def __hash__(self):
        vals = tuple(v for _, v in np.ndenumerate(self._matrix))
        return hash((SingleQubitMatrixGate, vals))

    def _approx_eq_(self, other: Any, atol) -> bool:
        if not isinstance(other, type(self)):
            return NotImplemented
        return np.allclose(self._matrix, other._matrix, rtol=0, atol=atol)

    def __eq__(self, other):
        if not isinstance(other, type(self)):
            return NotImplemented
        return np.alltrue(self._matrix == other._matrix)

    def __ne__(self, other):
        return not self == other

    def __repr__(self):
        return 'cirq.SingleQubitMatrixGate({})'.format(
            proper_repr(self._matrix))

    def __str__(self):
        return str(self._matrix.round(3))


class TwoQubitMatrixGate(gate_features.TwoQubitGate):
    """A 2-qubit gate defined only by its matrix.

    More general than specialized classes like `CZPowGate`, but more expensive
    and more float-error sensitive to work with (due to using
    eigendecompositions).
    """

    def __init__(self, matrix: np.ndarray) -> None:
        """
        Initializes the 2-qubit matrix gate.

        Args:
            matrix: The matrix that defines the gate.
        """

        if matrix.shape != (4, 4) or not linalg.is_unitary(matrix):
            raise ValueError('Not a 4x4 unitary matrix: {}'.format(matrix))
        self._matrix = matrix


    def validate_args(self, qubits):
        if len(qubits) != 2:
            raise ValueError(
                'Two-qubit gate not applied to two qubits: {}({})'.format(
                    self, qubits))


    def __pow__(self, exponent: Any) -> 'TwoQubitMatrixGate':
        if not isinstance(exponent, (int, float)):
            return NotImplemented
        e = cast(float, exponent)
        new_mat = linalg.map_eigenvalues(self._matrix, lambda b: b**e)
        return TwoQubitMatrixGate(new_mat)

    def _phase_by_(self, phase_turns: float, qubit_index: int):
        i = np.eye(2)
        z = _phase_matrix(phase_turns)
        z2 = np.kron(i, z) if qubit_index else np.kron(z, i)
        phased_matrix = z2.dot(self._matrix).dot(np.conj(z2.T))
        return TwoQubitMatrixGate(phased_matrix)

    def _approx_eq_(self, other: Any, atol) -> bool:
        if not isinstance(other, type(self)):
            return NotImplemented
        return np.allclose(self._matrix, other._matrix, rtol=0, atol=atol)

    def _unitary_(self) -> np.ndarray:
        return np.array(self._matrix)

    def _circuit_diagram_info_(self, args: protocols.CircuitDiagramInfoArgs
                               ) -> protocols.CircuitDiagramInfo:
        return protocols.CircuitDiagramInfo(
            wire_symbols=(_matrix_to_diagram_symbol(self._matrix, args), '#2'))

    def __hash__(self):
        vals = tuple(v for _, v in np.ndenumerate(self._matrix))
        return hash((SingleQubitMatrixGate, vals))

    def __eq__(self, other):
        if not isinstance(other, type(self)):
            return NotImplemented
        return np.alltrue(self._matrix == other._matrix)

    def __ne__(self, other):
        return not self == other

    def __repr__(self):
        return 'cirq.TwoQubitMatrixGate({})'.format(
                proper_repr(self._matrix))

    def __str__(self):
        return str(self._matrix.round(3))


class ThreeQubitMatrixGate(gate_features.ThreeQubitGate):
    """A 3-qubit gate defined only by its matrix.
    More general than specialized classes like `CZPowGate`, but more expensive
    and more float-error sensitive to work with (due to using
    eigendecompositions).
    """
    def __init__(self, matrix: np.ndarray) -> None:
        """
        Initializes the 2-qubit matrix gate.

        Args:
            matrix: The matrix that defines the gate.
        """

        if matrix.shape != (6, 6) or not linalg.is_unitary(matrix):
            raise ValueError('Not a 6x6 unitary matrix: {}'.format(matrix))
        self._matrix = matrix

    def validate_args(self, qubits):
        if len(qubits) != 3:
            raise ValueError('Three-qubit gate not applied to two qubits: {}({})'.format(self, qubits))

    def __pow__(self, exponent: Any) -> 'ThreeQubitMatrixGate':
        if not isinstance(exponent, (int, float)):
            return NotImplemented
        e = cast(float, exponent)
        new_mat = linalg.map_eigenvalues(self._matrix, lambda b: b**e)
        return ThreeQubitMatrixGate(new_mat)

    def _approx_eq_(self, other: Any, atol) -> bool:
        if not isinstance(other, type(self)):
            return NotImplemented
        return np.allclose(self._matrix, other._matrix, rtol=0, atol=atol)

    def _unitary_(self) -> np.ndarray:
        return np.array(self._matrix)

    def _apply_unitary_(self, args: protocols.ApplyUnitaryArgs) -> np.ndarray:
        if protocols.is_parameterized(self):
            return NotImplemented
        return args.target_tensor

    def _circuit_diagram_info_(self, args: protocols.CircuitDiagramInfoArgs) -> protocols.CircuitDiagramInfo:
        return protocols.CircuitDiagramInfo(
            ('3', '3', '3'),
            exponent=1)

    def __hash__(self):
        vals = tuple(v for _, v in np.ndenumerate(self._matrix))
        return hash((SingleQubitMatrixGate, vals))

    def __eq__(self, other):
        if not isinstance(other, type(self)):
            return NotImplemented
        return np.alltrue(self._matrix == other._matrix)

    def __ne__(self, other):
        return not self == other

    def __repr__(self):
        return 'cirq.ThreeQubitMatrixGate({})'.format(
                proper_repr(self._matrix))

    def __str__(self):
        return str(self._matrix.round(3))


def _matrix_to_diagram_symbol(matrix: np.ndarray,
                              args: protocols.CircuitDiagramInfoArgs) -> str:
    if args.precision is not None:
        matrix = matrix.round(args.precision)
    result = str(matrix)
    if args.use_unicode_characters:
        lines = result.split('\n')
        for i in range(len(lines)):
            lines[i] = lines[i].replace('[[', '')
            lines[i] = lines[i].replace(' [', '')
            lines[i] = lines[i].replace(']', '')
        w = max(len(line) for line in lines)
        for i in range(len(lines)):
            lines[i] = '│' + lines[i].ljust(w) + '│'
        lines.insert(0, '┌' + ' ' * w + '┐')
        lines.append('└' + ' ' * w + '┘')
        result = '\n'.join(lines)
    return result


def modifiedGramSchmidt(A):
    """
    Gives a orthonormal matrix, using modified Gram Schmidt Procedure
    :param A: a matrix of column vectors
    :return: a matrix of orthonormal column vectors
    """
    # assuming A is a square matrix
    dim = A.shape[0]
    Q = np.zeros(A.shape, dtype=A.dtype)
    for j in range(0, dim):
        q = A[:,j]
        for i in range(0, j):
            rij = np.vdot(Q[:, i], q)
            q = q - rij*Q[:, i]
        rjj = np.linalg.norm(q, ord=2)
        if np.isclose(rjj, 0.0):
            raise ValueError("invalid input matrix")
        else:
            Q[:, j] = q/rjj
    return Q


# Defining our 7 Qubits

qubits = [cirq.GridQubit(x, 0) for x in range(7)]


# Moment 0

h1 = cirq.H(qubits[1])
h2 = cirq.H(qubits[2])
h5 = cirq.H(qubits[5])
moment0 = cirq.Moment([h1, h2, h5])

# Moment 1

cn23 = cirq.CNOT(qubits[2], qubits[3])
cn14 = cirq.CNOT(qubits[1], qubits[4])
cn56 = cirq.CNOT(qubits[5], qubits[6])
moment1 = cirq.Moment([cn23, cn14, cn56])

# Moment 2

d = 3
n = np.power(2, d)
comp_seed = np.random.rand(n, n) + np.random.rand(n, n)*1j

# unitary = np.identity(n)
# unitary_star = np.identity(n)
unitary = modifiedGramSchmidt(comp_seed)
unitary_star = np.conjugate(np.transpose(unitary))

u = ThreeQubitMatrixGate(unitary)
u_star = ThreeQubitMatrixGate(unitary_star)

# u123 = u.on(qubits[0], qubits[1], qubits[2])
# u456 = u_star.on(qubits[3], qubits[4], qubits[5])
CP = cirq.CCZPowGate()
u123 = CP.on(qubits[0], qubits[1], qubits[2])
u456 = cirq.CSWAP.on(qubits[3], qubits[4], qubits[5])


moment2 = cirq.Moment([u123, u456])

# Moment 3

m23 = cirq.measure(qubits[2], qubits[3])
moment3 = cirq.Moment([m23])


# Assemble Moments

circuit = cirq.Circuit((moment0, moment1, moment2))
print(circuit)


# Creating Simulator

simulator = cirq.Simulator()
results = simulator.run(circuit, repetitions=10)
print("Results:")
print(results)





