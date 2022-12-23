import pennylane as qml
import numpy as np

# ========================================================================================
# This is the code for building a parameterized quantum circuit for machine learning task
# The final circuit contains encoder, pqc, measurement scheme. 
# ========================================================================================
if __name__ != '__main__':
    from src.Encoder import EncodingCircuits
    from src.PQCs import ParameterizedQuantumCircuits
    from src.Measurement import MeasurementMethod
    from src.Pruning import PrunedCircuits

class PQCBuilder:
    def __init__(self, enc_shape = 1, pqc_shape = 1, meas = 1, layers = 1, qubit = 1, pratio = 0.0):
        self.enc_shape = enc_shape
        self.pqc_shape = pqc_shape
        self.meas = meas
        self.qubit = qubit
        self.layers = layers
        self.pratio = pratio
        
        if self.pratio > 0.0 :
            self.enc = EncodingCircuits(option = self.enc_shape, qubit = self.qubit)
            self.pqc = PrunedCircuits(option = self.pqc_shape, qubit = self.qubit, layers = self.layers, \
                                        method = 'random', pratio = self.pratio)
            self.measurement = MeasurementMethod(option = self.meas, qubit = self.qubit)

        else:
            self.enc = EncodingCircuits(option = self.enc_shape, qubit = self.qubit)
            self.pqc = ParameterizedQuantumCircuits(option = self.pqc_shape, qubit = self.qubit, layers = self.layers)
            self.measurement = MeasurementMethod(option = self.meas, qubit = self.qubit)
        
    def construct_circuit(self, inputs, weights0, weights1 = 0):
        assert len(inputs) <= self.enc.inputs_length()
        pqc_weights_shape = self.pqc.weights_shape()
        if isinstance(pqc_weights_shape[0], tuple):
            assert weights0.shape == pqc_weights_shape[0]
            assert weights1.shape == pqc_weights_shape[1]
        else:
            assert weights0.shape == pqc_weights_shape
        self.enc.get_encoder(inputs)
        self.pqc.get_pqc(weights0, weights1 = weights1)
        return self.measurement.get_meas()


if __name__ == '__main__':
    from Encoder import EncodingCircuits
    from PQCs import ParameterizedQuantumCircuits
    from Measurement import Measurement
    from Pruning import PrunedCircuits

    qnn = PQCBuilder(encoder = 2, qubit = 4, layers = 2, pqc_shape = 9, measure = 3, pratio = 0.9)
    input_length = qnn.enc.inputs_length()
    weight_shape = qnn.pqc.weights_shape()
    print(weight_shape)
    inputs = np.random.random(input_length)
    dev = qml.device("default.qubit", wires = 10) #target pennylane device
    qnode = qml.QNode(qnn.construct_circuit, dev) #circuit
    if isinstance(weight_shape[0], tuple):
        weights0 = np.random.random(weight_shape[0])
        weights1 = np.random.random(weight_shape[1])
        qnode(inputs, weights0, weights1)
    else:
        weights = np.random.random(weight_shape)
        qnode(inputs, weights)

    print(qnode.draw())