# StateVectorSimulator

## Overview
The `StateVectorSimulator` is a Python-based quantum simulator that simulates an n-qubit quantum system using a state vector representation. It supports the application of single-qubit and multi-qubit gates in any arbitrary order, including controlled gates with decreasing-order control qubits.

## Features
- Supports **any number of qubits**.
- Allows **arbitrary ordering of control and target qubits**.
- Implements standard quantum gates (X, H, Y, Z, S, T, RX, RY, RZ).
- Supports multi-qubit gates, including **CX, CS, CT, Toffoli (CCX), CSWAP, and MCX**.
- Allows batch application of quantum gates to multiple qubits at once.

## Installation
Ensure you have Python installed along with NumPy:
```bash
pip install numpy
```

## Usage
### Initializing a Simulator
```python
from statevector_simulator import StateVectorSimulator

sim = StateVectorSimulator(num_qubits=2)
```
This initializes a 2-qubit system in the |00> state.

### Applying Gates
```python
sim.h(0)   # Apply Hadamard gate to qubit 0
sim.x(1)   # Apply X (NOT) gate to qubit 1
```

### Applying Controlled Gates
```python
sim.cs(0, 1)  # Controlled-S gate with qubit 0 as control and qubit 1 as target
sim.ccx(0, 1, 2)  # Toffoli (CCX) gate with qubits 0 and 1 as controls, 2 as target
```

### Batch Application
```python
# Apply a single-qubit gate to multiple qubits
sim.apply_multiple(x, [0, 1])  # Apply X gate to qubits 0 and 1

# Apply a two-qubit gate to multiple sets of qubits
sim.apply_multiple(cx, [[0, 1], [1, 2]])  # Apply CNOT to (0,1) and (1,2)
```

### Checking the State Vector
```python
print(sim.state)
```

## Example: Creating a Bell State
```python
sim = StateVectorSimulator(2)
sim.h(0)
sim.cx(0, 1)
print(sim.state)  # Should represent (|00> + |11>)/sqrt(2)
```
