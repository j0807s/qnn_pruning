import pennylane as qml

# ========================================================================================
# This is the code for mesurement schemes
# Option 1: PauliX; Option 2: PauliY; Option 3: PauliZ
# ========================================================================================

class MeasurementMethod:
    def __init__(self, option = None, qubit = None):
        self.options = [1, 2, 3]
        assert option in self.options
        self.measure = option
        self.qubit = qubit
    
    def get_meas(self):
        if self.measure == 1:
            return self.measurement_1()
        if self.measure == 2:
            return self.measurement_2()
        if self.measure == 3:
            return self.measurement_3()

    def output_dim(self):
        if self.measure == 1:
            return self.qubit
        if self.measure == 2:
            return self.qubit
        if self.measure == 3:
            return self.qubit

    def measurement_1(self):
        return [qml.expval(qml.PauliX(i)) for i in range(self.qubit)]

    def measurement_2(self):
        return [qml.expval(qml.PauliY(i)) for i in range(self.qubit)]

    def measurement_3(self):
        return [qml.expval(qml.PauliZ(i)) for i in range(self.qubit)]
    

