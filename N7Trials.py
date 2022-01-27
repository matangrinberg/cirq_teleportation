
import numpy as np
import cirq
import time
from datetime import datetime
import collections
import csv
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
        self._dim = int(nDim)

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
        return NQubitMatrixGate(new_mat)

    def _phase_by_(self, phase_turns: float, qubit_index: int):
        i = np.eye(2)
        z = _phase_matrix(phase_turns)
        z2 = np.kron(i, z) if qubit_index else np.kron(z, i)
        phased_matrix = z2.dot(self._matrix).dot(np.conj(z2.T))
        return NQubitMatrixGate(phased_matrix)

    def _approx_eq_(self, other: Any, atol) -> bool:
        if not isinstance(other, type(self)):
            return NotImplemented
        return np.allclose(self._matrix, other._matrix, rtol=0, atol=atol)

    def _unitary_(self) -> np.ndarray:
        return np.array(self._matrix)

    def _circuit_diagram_info_(self, args: protocols.CircuitDiagramInfoArgs) -> protocols.CircuitDiagramInfo:
        return protocols.CircuitDiagramInfo(wire_symbols=('#N',)*self._dim)

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
        return 'cirq.NQubitMatrixGate({})'.format(
                proper_repr(self._matrix))

    def __str__(self):
        return str(self._matrix.round(3))


def _matrix_to_diagram_symbol(matrix: np.ndarray, args: protocols.CircuitDiagramInfoArgs) -> str:
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
    print(result)
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


N = 7
n = 2**N

successRates = []
trials = 4000
for it in range(trials):

    # Defining our Qubits
    qubits = [cirq.GridQubit(x, 0) for x in range(2*N+1)]
    circuit = cirq.Circuit()

    # Moment H
    momentH1 = []
    momentH1.append(cirq.H(qubits[0]))
    momentH1.append(cirq.H(qubits[1]))
    momentH1.append(cirq.H(qubits[2]))
    momentH1.append(cirq.H(qubits[3]))
    momentH1.append(cirq.H(qubits[4]))
    momentH1.append(cirq.H(qubits[5]))
    circuit.append(cirq.Moment(momentH1))

    # Moment CNOT
    momentCN1 = []
    momentCN1.append(cirq.CNOT(qubits[0], qubits[7]))
    momentCN1.append(cirq.CNOT(qubits[1], qubits[8]))
    momentCN1.append(cirq.CNOT(qubits[2], qubits[9]))
    momentCN1.append(cirq.CNOT(qubits[3], qubits[10]))
    momentCN1.append(cirq.CNOT(qubits[4], qubits[11]))
    momentCN1.append(cirq.CNOT(qubits[5], qubits[12]))
    circuit.append(cirq.Moment(momentCN1))

    # Creating  2^N x 2^N matrix

    comp_seed = (1/np.sqrt(2)) * (np.random.rand(n, n) + np.random.rand(n, n)*1j)
    unitary = modifiedGramSchmidt(comp_seed)
    unitary_star = np.conjugate(unitary)

    u = NQubitMatrixGate(unitary)
    u_star = NQubitMatrixGate(unitary_star)
    u_tr = NQubitMatrixGate(np.transpose(unitary))
    u_dag = NQubitMatrixGate(np.transpose(unitary_star))
    u_dagg = NQubitMatrixGate(np.conjugate(np.transpose(unitary)))

    momentUni = []
    momentUni.append(u.on(qubits[0], qubits[1], qubits[2], qubits[3], qubits[4], qubits[5], qubits[6]))
    momentUni.append(u_star.on(qubits[7], qubits[8], qubits[9], qubits[10], qubits[11], qubits[12], qubits[13]))
    circuit.append(cirq.Moment(momentUni))

    momentProj =[]
    totKey = []
    momentProj.append(cirq.measure(qubits[0], qubits[7], key='M0'))
    totKey.append('M0')
    momentProj.append(cirq.measure(qubits[1], qubits[8], key='M7'))
    totKey.append('M7')
    # momentProj.append(cirq.measure(qubits[1], qubits[6], key='M1'))
    # totKey.append('M1')
    # momentProj.append(cirq.measure(qubits[2], qubits[7], key='M2'))
    # totKey.append('M2')
    # momentProj.append(cirq.measure(qubits[3], qubits[8], key='M3'))
    # totKey.append('M3')
    circuit.append(cirq.Moment(momentProj))


    momentProj =[]
    momentProj.append(cirq.measure(qubits[6], qubits[13], key='MF'))
    totKey.append('MF')
    circuit.append(cirq.Moment(momentProj))

    # print(circuit)
    # Creating Simulator
    reps = 10000
    simulator = cirq.Simulator()
    results = simulator.run(circuit, repetitions=reps)
    r = results.multi_measurement_histogram(keys=totKey)
    col = collections.Counter(r)

    # print(r)
    cor = []
    succ = col[0, 0, 0] + col[0, 0, 3] + col[0, 3, 0] + col[0, 3, 3] + col[3, 0, 0] + col[3, 0, 3] + col[3, 3, 0] + col[3, 3, 3]
    bad = col[0, 0, 1] + col[0, 0, 2] + col[0, 3, 1] + col[0, 3, 2] + col[3, 0, 1] + col[3, 0, 2] + col[3, 3, 1] + col[3, 3, 2]
    total = succ + bad
    rate = succ/total
    successRates.append(rate)
    print(rate)

print(successRates)
print(np.average(successRates))


now = datetime.now()
date_time = now.strftime("%m.%d.%Y.%H.%M.%S")
file_name ='N' + str(N) + 'test'+'Range' + str(trials)

with open(file_name + date_time + '.csv', 'a') as csv_file:
    csv_writer = csv.writer(csv_file, delimiter=',')
    csv_writer.writerow(successRates)
