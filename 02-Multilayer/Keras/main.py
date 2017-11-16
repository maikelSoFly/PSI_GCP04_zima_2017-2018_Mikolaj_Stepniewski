from keras.callbacks import TensorBoard
from keras.callbacks import ModelCheckpoint
from rastrigin import *
from model import *
from time import gmtime, strftime


""" Parameters """
lRate = 0.01
layers = [30, 1]
numberOfInputs = 20000
epochs = 25000
batchSize = 100
decay = 0


""" Initializing training data """
trainingData = RastriginInput()
trainingData.initRastriginPointsRand(numberOfInputs)
trainingDataInput = trainingData.getInputArray()
trainingDataExpectedOutput = trainingData.getOutputArray()

""" Initializing validation data """
validationData = RastriginInput()
validationData.initRastriginPoints(0.1)
valDataOutput = validationData.getOutputArray()
valDataInput = validationData.getInputArray()

""" Initializing Tensor Board, which contains charts, histograms etc.  """
dateTime = strftime("%Y-%m-%d--%H:%M:%S", gmtime())
layerStr = ''
for i, lr in enumerate(layers):
    layerStr += ('+' if i > 0 else '') + str(lr)
tensorBoard = TensorBoard(  log_dir='./logs/{}_lr={:.2f}_noIn={:d}_ep={:d}_bs={:d}--lay:{}'.format(dateTime, lRate,
                                                                        numberOfInputs, epochs, batchSize, layerStr),
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
checkpointer = ModelCheckpoint(filepath='./best_validated/weights.hdf5', verbose=1, save_best_only=True)


""" Keras Model """
kModel = KerasModel(layers, [checkpointer])
kModel.createModel(lRate, decay)
kModel.loadWeights('./best_validated/weights.hdf5')
# kModel.train(
#     trainingDataInput,
#     trainingDataExpectedOutput,
#     epochs,
#     batchSize,
#     (valDataInput, valDataOutput)
# )

""" Summary table """
print('\n\n\tSummary:\n', kModel._model.summary())
kModel.printEvaluation(valData=[valDataInput, valDataOutput])
