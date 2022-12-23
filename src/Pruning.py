import pennylane as qml
import numpy as np


# ========================================================================================
# This is the code for pruning parameterized quantum circuit before training. 
# Current implementation only supports random pruning, pruning ratio : 0.0 ~ 1.0
# Chosen circuits : 1, 3, 5, 9
# ========================================================================================

class PrunedCircuits:
    def __init__(self, option = None, qubit = None, layers = None, method = None, pratio = None):
        self.options = [1, 3, 5, 9]
        assert option in self.options
        self.option = option
        self.qubit = qubit
        self.layers = layers
        
        self.method = method
        self.pratio = pratio

    def get_pqc(self, weights0, weights1):
        if self.option == 1:
            return self.pruned_pqc_1(weights0)
        if self.option == 3:
            return self.pruned_pqc_3(weights0, weights1)
        if self.option == 5:
            return self.pruned_pqc_5(weights0, weights1)
        if self.option == 9:
            return self.pruned_pqc_9(weights0)

    def weights_shape(self):
        if self.option == 1:
            return (self.layers, self.qubit, 2)
        if self.option == 3:
            return ((self.layers, self.qubit, 2), (self.layers, self.qubit - 1))
        if self.option == 5:
            return ((self.layers, self.qubit, 4), (self.layers, self.qubit, self.qubit - 1))
        if self.option == 9:
            return (self.layers, self.qubit)

    def random_pruning(self):
        pruned_layer = []
        control_pruned_layer = []
        rx_params = []

        if self.option == 1:
            for q in range(self.qubit):
                tmp=[]
                binary_array_RX = np.random.binomial(1, 1-self.pratio, size=self.layers)
                binary_array_RZ = np.random.binomial(1, 1-self.pratio, size=self.layers)
                tmp.append(binary_array_RX)
                tmp.append(binary_array_RZ)
                pruned_layer.append(tmp)
            
        if self.option == 3:
            for q in range(self.qubit):
                tmp=[]
                binary_array_RX = np.random.binomial(1, 1-self.pratio, size=self.layers)
                binary_array_RZ = np.random.binomial(1, 1-self.pratio, size=self.layers)
                tmp.append(binary_array_RX)
                tmp.append(binary_array_RZ)
                pruned_layer.append(tmp)
            
            for q in range(self.qubit-1):
                tmp=[]
                binary_array_CRZ = np.random.binomial(1, 1-self.pratio, size=self.layers)
                tmp.append(binary_array_CRZ)
                control_pruned_layer.append(tmp)

        if self.option == 5:
            for q in range(self.qubit):
                tmp=[]
                binary_array_RX = np.random.binomial(1, 1-self.pratio, size=self.layers)
                binary_array_RZ = np.random.binomial(1, 1-self.pratio, size=self.layers)
                binary_array_RX2 = np.random.binomial(1, 1-self.pratio, size=self.layers)
                binary_array_RZ2 = np.random.binomial(1, 1-self.pratio, size=self.layers)
                tmp.append(binary_array_RX)
                tmp.append(binary_array_RZ)
                tmp.append(binary_array_RX2)
                tmp.append(binary_array_RZ2)
                pruned_layer.append(tmp)
            
            for q in range(self.qubit):
                tmp=[]
                binary_array_CRZ1 = np.random.binomial(1, 1-self.pratio, size=self.layers)
                binary_array_CRZ2 = np.random.binomial(1, 1-self.pratio, size=self.layers)
                binary_array_CRZ3 = np.random.binomial(1, 1-self.pratio, size=self.layers)
                tmp.append(binary_array_CRZ1)
                tmp.append(binary_array_CRZ2)
                tmp.append(binary_array_CRZ3)
                control_pruned_layer.append(tmp)

        if self.option == 9:
            for q in range(self.qubit):
                tmp=[]
                binary_array_H = np.random.binomial(1, 1-self.pratio, size=self.layers)
                binary_array_RX = np.random.binomial(1, 1-self.pratio, size=self.layers)
                tmp.append(binary_array_H)
                tmp.append(binary_array_RX)
                pruned_layer.append(tmp)
                rx_params.append(binary_array_RX)

            for q in range(self.qubit-1):
                tmp=[]
                binary_array_CNOT = np.random.binomial(1, 1-self.pratio, size=self.layers)
                tmp.append(binary_array_CNOT)
                control_pruned_layer.append(tmp)

        pruned_layer = np.array(pruned_layer)
        control_pruned_layer = np.array(control_pruned_layer)
        
        num_params = int(pruned_layer.sum() + control_pruned_layer.sum())
        if len(rx_params) > 0:
            num_params = int(np.array(rx_params).sum() + control_pruned_layer.sum())
            
        #params = np.random.random([num_params], requires_grad=True)
        params = np.random.random([num_params])
        
        return pruned_layer, control_pruned_layer, params, num_params

    def pruned_pqc_1(self, weights):
        assert weights.shape == self.weights_shape()
        pruned_layer, _, params, num_params = self.random_pruning()
        counter = 0
        
        for l in range(self.layers):
            for q in range(self.qubit):

                if pruned_layer[q][0][l] == 1:
                    qml.RX(weights[l, q, 0], wires=q)
                    counter = counter + 1
                    if counter == num_params:
                        break

                if pruned_layer[q][1][l] == 1:
                    qml.RZ(weights[l, q, 1], wires=q)
                    counter = counter + 1
                    if counter == num_params:
                        break
    
    def pruned_pqc_3(self, weights0, weights1):
        assert weights0.shape == self.weights_shape()[0]
        assert weights1.shape == self.weights_shape()[1]
        pruned_layer, control_pruned_layer, params, num_params = self.random_pruning()

        counter = 0
        for l in range(self.layers):
            for q in range(self.qubit):
                if pruned_layer[q][0][l] == 1:
                    qml.RX(weights[l, q, 0], wires=q)
                    counter = counter + 1
                    if counter == num_params:
                        break

                if pruned_layer[q][1][l] == 1:
                    qml.RZ(weights[l, q, 1], wires=q)
                    counter = counter + 1
                    if counter == num_params:
                        break
            
            for q in range(self.qubit-1):
                if control_pruned_layer[q][0][l] == 1:
                    qml.CRZ(params[counter], wires = [q, (q + 1)])
                    counter = counter + 1
                    if counter == num_params:
                        break

    def pruned_pqc_5(self, weights0, weights1):
        assert weights0.shape == self.weights_shape()[0]
        assert weights1.shape == self.weights_shape()[1]
        pruned_layer, control_pruned_layer, params, num_params = self.random_pruning()

        counter = 0
        for l in range(self.layers):
            for q in range(self.qubit):
                if pruned_layer[q][0][l] == 1:
                    qml.RX(weights0[l, q, 0], wires=q)
                    counter = counter + 1
                    if counter == num_params:
                        break
                if pruned_layer[q][1][l] == 1:
                    qml.RZ(weights0[l, q, 1], wires=q)
                    counter = counter + 1
                    if counter == num_params:
                        break
            
            for q in range(self.qubit):    
                if control_pruned_layer[q][0][l] == 1:
                    qml.CRZ(weights1[l,q,0], wires = [q, (q + 3)%self.qubit])
                    counter = counter + 1
                    if counter == num_params:
                        break
                if control_pruned_layer[q][1][l] == 1:
                    qml.CRZ(weights1[l,q,1], wires = [q, (q + 2)%self.qubit])
                    counter = counter + 1
                    if counter == num_params:
                        break
                    
                if control_pruned_layer[q][2][l] == 1:
                    qml.CRZ(weights1[l,q,2], wires = [q, (q + 1)%self.qubit])
                    counter = counter + 1
                    if counter == num_params:
                        break
                            
            for q in range(self.qubit):
                if pruned_layer[q][2][l] == 1:
                    qml.RX(weights0[l,q,2], wires=q)
                    counter = counter + 1
                    if counter == num_params:
                        break
                    
                if pruned_layer[q][3][l] == 1:
                    qml.RZ(weights0[l,q,3], wires=q)
                    counter = counter + 1
                    if counter == num_params:
                        break

    def pruned_pqc_9(self, weights):
        assert weights.shape == self.weights_shape()
        pruned_layer, control_pruned_layer, params, num_params = self.random_pruning()

        counter = 0
        for l in range(self.layers):
            for q in range(self.qubit):
                if pruned_layer[q][0][l] == 1:
                    qml.Hadamard(wires=q)
                    if counter == num_params:
                        break
                
            for q in range(self.qubit-1):
                if control_pruned_layer[q][0][l] ==1:
                    qml.CZ(wires = [q, (q + 1)])
                    if counter == num_params:
                        break
                    
            for q in range(self.qubit):
                if pruned_layer[q][1][l] == 1:
                    qml.RX(weights[l,q,0], wires=q)
                    counter = counter + 1
                    if counter == num_params:
                        break


        



    

    
