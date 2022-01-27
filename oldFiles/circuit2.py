
import numpy as np
import cirq
import collections
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


class ThreeQubitMatrixGate(gate_features.TwoQubitGate):
    """A 3-qubit gate defined only by its matrix.

    More general than specialized classes like `CCZPowGate`, but more expensive
    and more float-error sensitive to work with (due to using
    eigendecompositions).
    """

    def __init__(self, matrix: np.ndarray) -> None:
        """
        Initializes the 3-qubit matrix gate.

        Args:
            matrix: The matrix that defines the gate.
        """

        if matrix.shape != (8, 8) or not linalg.is_unitary(matrix):
            raise ValueError('Not a 8x8 unitary matrix: {}'.format(matrix))
        self._matrix = matrix

    def validate_args(self, qubits):
        if len(qubits) != 3:
            raise ValueError(
                'Three-qubit gate not applied to threee qubits: {}({})'.format(
                    self, qubits))

    def __pow__(self, exponent: Any) -> 'ThreeQubitMatrixGate':
        if not isinstance(exponent, (int, float)):
            return NotImplemented
        e = cast(float, exponent)
        new_mat = linalg.map_eigenvalues(self._matrix, lambda b: b**e)
        return ThreeQubitMatrixGate(new_mat)

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

    def _circuit_diagram_info_(self, args: protocols.CircuitDiagramInfoArgs) -> protocols.CircuitDiagramInfo:
        return protocols.CircuitDiagramInfo(
            # wire_symbols=(_matrix_to_diagram_symbol(self._matrix, args), '#3', '#3'))
            wire_symbols=('#3', '#3', '#3'))

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


class NQubitMatrixGate(gate_features.TwoQubitGate):
    """An N-qubit gate defined only by its matrix.

    More general than specialized classes like `CCZPowGate`, but more expensive
    and more float-error sensitive to work with (due to using
    eigendecompositions).
    """

    def __init__(self, matrix: np.ndarray) -> None:
        """
        Initializes the N-qubit matrix gate.

        Args:
            matrix: The matrix that defines the gate.
        """

        nDim = np.log(np.shape(matrix)[1])/np.log(2)

        if matrix.shape != (2**nDim, 2**nDim) or not linalg.is_unitary(matrix):
            raise ValueError('Not a 2**nx2**n unitary matrix: {}'.format(matrix))
        self._matrix = matrix
        self._dim = nDim

    def validate_args(self, qubits):
        if len(qubits) != self._dim:
            raise ValueError(
                'N-qubit gate not applied to n qubits: {}({})'.format(
                    self, qubits))

    def __pow__(self, exponent: Any) -> 'NQubitMatrixGate':
        if not isinstance(exponent, (int, float)):
            return NotImplemented
        e = cast(float, exponent)
        new_mat = linalg.map_eigenvalues(self._matrix, lambda b: b**e)
        return ThreeQubitMatrixGate(new_mat)

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

    def _circuit_diagram_info_(self, args: protocols.CircuitDiagramInfoArgs) -> protocols.CircuitDiagramInfo:
        return protocols.CircuitDiagramInfo(
            # wire_symbols=(_matrix_to_diagram_symbol(self._matrix, args), '#3', '#3'))
            wire_symbols=('#N', '#N', '#N'))

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


def modifiedGramSchmidt(a):
    """
    Gives a orthonormal matrix, using modified Gram Schmidt Procedure
    :param a: a matrix of column vectors
    :return: a matrix of orthonormal column vectors
    """
    # assuming a is a square matrix
    dim = a.shape[0]
    Q = np.zeros(a.shape, dtype=a.dtype)
    for j in range(0, dim):
        q = a[:, j]
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
circuit = cirq.Circuit()

# ** Moment 0
# psi = cirq.H(qubits[0])
# circuit.append(cirq.Moment([psi]))


# ** Moment 1
h1 = cirq.H(qubits[1])
h2 = cirq.H(qubits[2])
h5 = cirq.H(qubits[5])
circuit.append(cirq.Moment([h1, h2, h5]))

# ** Moment 2
cn14 = cirq.CNOT(qubits[1], qubits[4])
cn23 = cirq.CNOT(qubits[2], qubits[3])
cn56 = cirq.CNOT(qubits[5], qubits[6])
circuit.append(cirq.Moment([cn56, cn23, cn14]))


# ** Moment 3

# creating  8 x 8 matrix (2^3 = 8)
n = 8
comp_seed = (1/np.sqrt(2)) * (np.random.rand(n, n) + np.random.rand(n, n)*1j)

unitary = modifiedGramSchmidt(comp_seed)
unitary_star = np.conjugate(unitary)

u = NQubitMatrixGate(unitary)
u_star = NQubitMatrixGate(unitary_star)
u123 = u.on(qubits[0], qubits[1], qubits[2])
u456 = u_star.on(qubits[3], qubits[4], qubits[5])

circuit.append(cirq.Moment([u123, u456]))


# ** Moment 4
cnp1 = cirq.CNOT(qubits[5], qubits[0])
cnp2 = cirq.CNOT(qubits[1], qubits[4])
cnp3 = cirq.CNOT(qubits[2], qubits[3])
# circuit.append(cirq.Moment([cnp3]))
circuit.append(cirq.Moment([cnp1, cnp2, cnp3]))


# ** Moment 5
hp1 = cirq.H(qubits[5])
hp2 = cirq.H(qubits[1])
hp3 = cirq.H(qubits[2])
# circuit.append(cirq.Moment([hp2]))
circuit.append(cirq.Moment([hp1, hp2, hp3]))


# ** Moment 6
m03 = cirq.measure(qubits[1], qubits[4], key='M0')
m14 = cirq.measure(qubits[2], qubits[3], key='M1')
# m25 = cirq.measure(qubits[2], qubits[3], key='M2')

circuit.append(cirq.Moment([m03, m14]))
# circuit.append(cirq.Moment([m03, m25]))

# ** Moment 7
# hl = cirq.H(qubits[6])
# circuit.append(cirq.Moment([hl]))

# ** Moment 8
ml = cirq.measure(qubits[6], key='M6')
circuit.append(cirq.Moment([ml]))

print(circuit)


# Creating Simulator
reps = 10000
simulator = cirq.Simulator()
results = simulator.run(circuit, repetitions=reps)
r = results.multi_measurement_histogram(keys=['M0', 'M6'])
# r = results.multi_measurement_histogram(keys=['M0', 'M1', 'M2', 'M6'])
col = collections.Counter(r)

print(r)
cor = []
# print(dif0, dif1, dif2, dif3)
# for i in range(4):
#     for j in range(4):
#         print(i, j)
#         cor.append((col[i, j, 0]-col[i, j, 1])/(col[i, j, 0]+col[i, j, 1]))


# for i in range(4):
#     for j in range(4):
#         for k in range(4):
#             print(i, j, k)
#             cor.append((col[i, j, k, 0]-col[i, j, k, 1])/(col[i, j, k, 0]+col[i, j, k, 1]))

print(np.sort(cor))


