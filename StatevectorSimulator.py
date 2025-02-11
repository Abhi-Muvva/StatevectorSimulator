import numpy as np

class StateVectorSimulator:
    def __init__(self, num_qubits):
        """Initialize an n-qubit state vector in the |0...0> state."""
        self.num_qubits = num_qubits
        self.state = np.zeros(2**num_qubits, dtype=complex)
        self.state[0] = 1  # Initialize to |00...0>

    def apply_gate(self, gate, qubit_indices):
        """Apply a k-qubit gate to the state vector, correctly handling arbitrary qubit orders."""
        num_target_qubits = len(qubit_indices)
        full_dim = 2 ** self.num_qubits
        target_dim = 2 ** num_target_qubits

        index_map = {q: i for i, q in enumerate(qubit_indices)}
        new_state = np.zeros_like(self.state, dtype=complex)

        for i in range(full_dim):
            target_bits = sum(((i >> q) & 1) << index_map[q] for q in qubit_indices)
            target_vector = np.zeros(target_dim, dtype=complex)
            for j in range(target_dim):
                modified_i = i
                for k, qubit in enumerate(qubit_indices):
                    if ((j >> k) & 1) != ((i >> qubit) & 1):
                        modified_i ^= (1 << qubit)
                target_vector[j] = self.state[modified_i]
            
            new_vector = gate @ target_vector
            for j in range(target_dim):
                modified_i = i
                for k, qubit in enumerate(qubit_indices):
                    if ((j >> k) & 1) != ((i >> qubit) & 1):
                        modified_i ^= (1 << qubit)
                new_state[modified_i] = new_vector[j]
        
        self.state = new_state

    def apply_multiple(self, gate, qubit_sets):
        gate_dim = gate.shape[0]
        num_target_qubits = int(np.log2(gate_dim))

        if num_target_qubits == 1:
            for qubit in qubit_sets:
                self.apply_gate(gate, [qubit])  
        else:
            for qubit_indices in qubit_sets:
                self.apply_gate(gate, qubit_indices)

    # ----- Standard Quantum Gates -----
    def h(self, qubit):
        H = (1 / np.sqrt(2)) * np.array([[1, 1], [1, -1]])
        self.apply_gate(H, [qubit])

    def x(self, qubit):
        X = np.array([[0, 1], [1, 0]])
        self.apply_gate(X, [qubit])

    def y(self, qubit):
        Y = np.array([[0, -1j], [1j, 0]])
        self.apply_gate(Y, [qubit])

    def z(self, qubit):
        Z = np.array([[1, 0], [0, -1]])
        self.apply_gate(Z, [qubit])

    def s(self, qubit):
        S = np.array([[1, 0], [0, 1j]])
        self.apply_gate(S, [qubit])

    def t(self, qubit):
        T = np.array([[1, 0], [0, np.exp(1j * np.pi / 4)]])
        self.apply_gate(T, [qubit])

    def i(self, qubit):
        I = np.array([[1, 0], [0, 1]])
        self.apply_gate(I, [qubit])

    def rx(self, qubit, theta):
        RX = np.array([[np.cos(theta / 2), -1j * np.sin(theta / 2)],
                       [-1j * np.sin(theta / 2), np.cos(theta / 2)]])
        self.apply_gate(RX, [qubit])

    def ry(self, qubit, theta):
        RY = np.array([[np.cos(theta / 2), -np.sin(theta / 2)],
                       [np.sin(theta / 2), np.cos(theta / 2)]])
        self.apply_gate(RY, [qubit])

    def rz(self, qubit, theta):
        RZ = np.array([[np.exp(-1j * theta / 2), 0],
                       [0, np.exp(1j * theta / 2)]])
        self.apply_gate(RZ, [qubit])

    def cs(self, control, target):
        CS = np.array([
            [1, 0, 0, 0],
            [0, 1, 0, 0],
            [0, 0, 1, 0],
            [0, 0, 0, 1j]
        ])
        self.apply_gate(CS, [control, target])

    def ct(self, control, target):
        CT = np.array([
            [1, 0, 0, 0],
            [0, 1, 0, 0],
            [0, 0, 1, 0],
            [0, 0, 0, np.exp(1j * np.pi / 4)]
        ])
        self.apply_gate(CT, [control, target])

    def cswap(self, control, q1, q2):
        CSWAP = np.array([
            [1, 0, 0, 0, 0, 0, 0, 0],
            [0, 1, 0, 0, 0, 0, 0, 0],
            [0, 0, 1, 0, 0, 0, 0, 0],
            [0, 0, 0, 1, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 1, 0, 0],
            [0, 0, 0, 0, 1, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 1],
            [0, 0, 0, 0, 0, 0, 1, 0]
        ])
        self.apply_gate(CSWAP, [control, q1, q2])
