from keras.callbacks import TensorBoard
from keras.callbacks import ModelCheckpoint
from rastrigin import *
from model import *
from time import gmtime, strftime


""" Parameters """
lRate = 0.01
layers = [30, 1]
numberOfInputs = 1000
epochs = 200
batchSize = 25
decay = 0


""" Initializing training data """
trainingData = RastriginInput()
trainingData.initRastriginPointsRand(numberOfInputs)
trainingDataInput = trainingData.getInputArray()
trainingDataExpectedOutput = trainingData.getOutputArray()

""" Initializing validation data """
validationData = RastriginInput()
validationData.initRastriginPoints(0.5)
valDataOutput = validationData.getOutputArray()
valDataInput = validationData.getInputArray()

""" Initializing Tensor Board, which contains charts, histograms etc.  """
dateTime = strftime("%Y-%m-%d--%H:%M:%S", gmtime())
tensorBoard = TensorBoard(  log_dir='./logs/{}_lr={:.2f}_noIn={:d}_ep={:d}_bs={:d}'.format(dateTime, lRate,
                                                                        numberOfInputs, epochs, batchSize),
                            histogram_freq=5,
                            batch_size=batchSize,
                            write_graph=True,
                            write_grads=False,
                            write_images=True,
                            embeddings_freq=0,
                            embeddings_layer_names=None,
                            embeddings_metadata=None
)

""" Saves weights of the model """
checkpointer = ModelCheckpoint(filepath='./checkpoints/weights.hdf5', verbose=1, save_best_only=True)


""" Keras Model """
kModel = KerasModel(layers, [tensorBoard, checkpointer])
kModel.createModel(lRate, decay)
#kModel.loadWeights('/Users/maikel/Documents/code/Python/PSI/02-Multilayer/Keras/checkpoints/weights.hdf5')
kModel.train(
    trainingDataInput,
    trainingDataExpectedOutput,
    epochs,
    batchSize,
    (valDataInput, valDataOutput)
)

""" Summary table """
print('\n\n\tSummary:\n', kModel._model.summary())
kModel.printEvaluation(valData=[valDataInput, valDataOutput])
