import pennylane as qml
import numpy as np

# ========================================================================================
# This is the code for selecting parameterized quantum circuit. 
# The shapes of the circuits are motivated from https://arxiv.org/abs/1905.10876
# Chosen circuits : 1, 3, 5, 9
# ========================================================================================

class ParameterizedQuantumCircuits:
    def __init__(self, option = None, qubit = None, layers = None):
        self.options = [1, 3, 5, 9]
        assert option in self.options
        self.option = option
        self.qubit = qubit
        self.layers = layers
        
    def get_pqc(self, weights0, weights1):
        if self.option == 1:
            return self.pqc_1(weights0)
        if self.option == 3:
            return self.pqc_3(weights0, weights1)
        if self.option == 5:
            return self.pqc_5(weights0, weights1)
        if self.option == 9:
            return self.pqc_9(weights0)
    
    def weights_shape(self):
        if self.option == 1:
            return (self.layers, self.qubit, 2)
        if self.option == 3:
            return ((self.layers, self.qubit, 2), (self.layers, self.qubit - 1))
        if self.option == 5:
            return ((self.layers, self.qubit, 4), (self.layers, self.qubit, self.qubit - 1))
        if self.option == 9:
            return (self.layers, self.qubit)
        
    def pqc_1(self, weights):
        assert weights.shape == self.weights_shape()
        for l in range(self.layers):
            for q in range(self.qubit):
                qml.RX(weights[l, q, 0], wires = q)
                qml.RZ(weights[l, q, 1], wires = q)

    def pqc_3(self, weights0, weights1):
        assert weights0.shape == self.weights_shape()[0]
        assert weights1.shape == self.weights_shape()[1]
        for l in range(self.layers):
            for q in range(self.qubit):
                qml.RX(weights0[l, q, 0], wires = q)
                qml.RZ(weights0[l, q, 1], wires = q)

            for q in range(self.qubit - 1):
                qml.CRZ(weights1[l, q], wires = [q, (q + 1)])

    def pqc_5(self, weights0, weights1):
        assert weights0.shape == self.weights_shape()[0]
        assert weights1.shape == self.weights_shape()[1]
        for l in range(self.layers):
            for q in range(self.qubit):
                qml.RX(weights0[l, q, 0], wires = q)
                qml.RZ(weights0[l, q, 1], wires = q)

            for q in range(self.qubit):
                for i in range(self.qubit - 1):
                    qml.CRZ(weights1[l, q, i], wires = [q, (q + i + 1)%self.qubit])

            for q in range(self.qubit):
                qml.RX(weights0[l, q, 2], wires = q)
                qml.RZ(weights0[l, q, 3], wires = q)
    
    def pqc_9(self, weights):
        assert weights.shape == self.weights_shape()
        for l in range(self.layers):
            for q in range(self.qubit):
                qml.Hadamard(wires = q)
            for q in range(self.qubit - 1):
                qml.CZ(wires=[q, (q + 1)])
            for q in range(self.qubit):
                qml.RX(weights[l, q], wires = q)



    
        



    



