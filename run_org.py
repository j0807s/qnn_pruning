import pennylane as qml
import tensorflow as tf
from sklearn.preprocessing import normalize
from sklearn.datasets import load_digits
import src.pqc_builder as pqc_builder
import src.loader as loader

parser = argparse.ArgumentParser()
parser.add_argument("-nl", "--num_layers", help="Number of pqc layers", type = int)
parser.add_argument("-pqc", "--pqc", help="Circuit shape of pqc", type = int)
args = parser.parse_args()

num_layers = args.num_layers
n_pqc = args.pqc

num_qubit = 4
encoder_shape = 2
pqc_shape = n_pqc
measure = 3
circuit = pqc_builder(encoder = encoder_shape, qubit = qubit, layers = layers, \
                pqc_shape = pqc_shape, measure = measure)

classes = [3,6]
trainset = loader.DownsampledMNIST(classes = classes)

weight_shape = circuit.pqc.weigths_shape()
if isinstance(weight_shape[0], tuple):
    ql_weights_shape = {'weights0': weight_shape[0], 'weights1': weight_shape[1]}
else:
    ql_weights_shape = {'weights0': weight_shape, 'weights1': ()}

output_dim = pqc.measurement.output_dim()
dev = qml.device("default.qubit", wires = qubit) #target pennylane device
qnode = qml.QNode(pqc.construct_circuit, dev, interface = 'tf', diff_method = 'backprop') #circuit
qlayer = qml.qnn.KerasLayer(qnode, ql_weights_shape, output_dim = output_dim)
clayer = tf.keras.layers.Dense(len(classes))
model = tf.keras.models.Sequential([qlayer, clayer])

#compile and train the model
opt = tf.keras.optimizers.SGD(learning_rate = 0.5)

model.compile(opt, tf.keras.losses.SparseCategoricalCrossentropy(from_logits = True), metrics = ['accuracy'])
start_time = time.time()
history = model.fit(trainset.X, trainset.Y, epochs = 10, batch_size = 20, validation_split = 0.2, use_multiprocessing = True)
et = time.time()-start_time


