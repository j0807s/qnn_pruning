import pennylane as qml
import numpy as np

# ========================================================================================
# This is the encoder code. Encoder prepares the qubit states according to the given data
# Option 1: Rz; Option Option 2: Ry-Rz-Rx-Ry
# ========================================================================================

class EncodingCircuits:
    def __init__(self, option = None, qubit = None):
        self.options=[1, 2, 3]
        assert option in self.options
        self.option = option
        self.qubit = qubit

    def get_encoder(self, x):
        if self.option == 1:
            return self.rz_encoder(x)
        
        elif self.option == 2:
            return self.ryzxy_encoder(x)

    def inputs_length(self):
        if self.option == 1:
            return self.qubit

        if self.option == 2:
            return self.qubit * 4
    

    def rz_encoder(self, x):
        assert len(x) < self.inputs_length()
        gates_list = ['RZ']
        for q in range(self.qubit):
            exec('qml.{}({}, wires = {})'.format(gates_list[0], x[q], q))
    

    def ryzxy_encoder(self, x):
        #assert len(x) < self.inputs_length()
        gates_list = ['RY', 'RZ', 'RX', 'RY']
        encoder_width = len(gates_list)
        for q in range(self.qubit):
            for i in range(encoder_width):  
                exec('qml.{}({}, wires = {})'.format(gates_list[i], x[q * encoder_width + i], q))

