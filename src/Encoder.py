import pennylane as qml
import numpy as np

# ========================================================================================
# This is the encoder code. Encoder prepares the qubit states according to the given data
# Option 1: Rz; Option Option 2: Ry-Rz-Rx-Ry
# ========================================================================================

def dummy_measurements_for_test(func):
    def inner(*args, **kwargs):
        func(*args, **kwargs)
        if test == True:
            return qml.expval(qml.PauliY(0))
    return inner


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
    
    @dummy_measurements_for_test
    def rz_encoder(self, x):
        assert len(x) < self.inputs_length()
        gates_list = ['RZ']
        for q in range(self.qubit):
            exec('qml.{}({}, wires = {})'.format(gates_list[0], x[q], q))
    
    @dummy_measurements_for_test
    def ryzxy_encoder(self, x):
        #assert len(x) < self.inputs_length()
        gates_list = ['RY', 'RZ', 'RX', 'RY']
        encoder_width = len(gates_list)
        for q in range(self.qubit):
            for i in range(encoder_width):  
                exec('qml.{}({}, wires = {})'.format(gates_list[i], x[q * encoder_width + i], q))

if __name__ == '__main__':
    test = True
    enc = EncodingCircuits(option = 2, qubit = 4)
    inputs_length = enc.inputs_length()
    print(type(inputs_length))
    inputs = np.random.random(inputs_length)
    print("inputs size:",len(inputs))
    dev = qml.device("default.qubit", wires = 4) #target pennylane device
    qnode = qml.QNode(enc.get_encoder, dev) #circuit
    qnode(inputs)
    print(qnode.draw())
    
else:
    test = False
