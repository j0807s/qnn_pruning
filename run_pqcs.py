import pennylane as qml
import tensorflow as tf
from sklearn.preprocessing import normalize
from sklearn.datasets import load_digits
from src.Builder import PQCBuilder
from src.loader import DownsampledMNIST
import argparse
import time
import os
import random
import numpy as np

np.random.random(42)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-nl", "--num_layers", help="Number of pqc layers", type = int)
    parser.add_argument("-pqc", "--pqc_shape", help="Circuit shape of pqc", type = int)
    parser.add_argument("-p", "--pruning_ratio", help="Pruning ratio", type = float, default = 0.0)
    args = parser.parse_args()

    num_layers = args.num_layers
    n_pqc = args.pqc_shape
    pratio = args.pruning_ratio

    num_qubit = 4
    encoder_shape = 2
    n_measure = 3

    classes = [0,1]
    trainset = DownsampledMNIST(classes = classes)

    circuit = PQCBuilder(enc_shape = encoder_shape, qubit = num_qubit, layers = num_layers, \
                        pqc_shape = n_pqc, meas = n_measure, pratio = pratio)

    
    weight_shape = circuit.pqc.weights_shape()
    if isinstance(weight_shape[0], tuple):
        ql_weights_shape = {'weights0': weight_shape[0], 'weights1': weight_shape[1]}
    else:
        ql_weights_shape = {'weights0': weight_shape, 'weights1': ()}

    output_dim = circuit.measurement.output_dim()
    dev = qml.device("default.qubit", wires = num_qubit) #target pennylane device
    qnode = qml.QNode(circuit.construct_circuit, dev, interface = 'tf', diff_method = 'backprop') #circuit
    qlayer = qml.qnn.KerasLayer(qnode, ql_weights_shape, output_dim = output_dim)
    clayer = tf.keras.layers.Dense(len(classes))
    model = tf.keras.models.Sequential([qlayer, clayer])

    #compile and train the model
    opt = tf.keras.optimizers.SGD(learning_rate = 0.5)

    model.compile(opt, tf.keras.losses.SparseCategoricalCrossentropy(from_logits = True), metrics = ['accuracy'])
    start_time = time.time()
    history = model.fit(trainset.X, trainset.Y, epochs = 10, batch_size = 20, validation_split = 0.2, use_multiprocessing = True)
    et = time.time()-start_time

    print('Circuit '+str(n_pqc)+' Training time for 10 epochs : {:.4f}'.format(et))
    print('Circuit '+str(n_pqc)+' Best Accuracy : {:.4f}'.format(max(history.history['val_accuracy'])))

