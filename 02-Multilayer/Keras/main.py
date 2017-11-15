from keras.models import Sequential
from keras.callbacks import TensorBoard
from keras.layers import Dense
from keras import optimizers
from prettytable import PrettyTable
from rastrigin import *


""" Parameters """
lRate = 0.05
layers = [30, 1]
numberOfInputs = 2000
epochs = 1000
batchSize = 20
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
tensorBoard = TensorBoard(  log_dir='./logs',
                            histogram_freq=5,
                            batch_size=20,
                            write_graph=True,
                            write_grads=False,
                            write_images=False,
                            embeddings_freq=0,
                            embeddings_layer_names=None,
                            embeddings_metadata=None
)

""" CREATING KERAS MODEL """
model=Sequential()
for i in range(len(layers)):
    if i == 0:
        model.add(Dense(layers[i], input_dim=2, activation='sigmoid'))
    elif i == len(layers) - 1:
        model.add(Dense(layers[i], activation='linear'))
    else:
        model.add(Dense(layers[i], activation='sigmoid'))

adam = optimizers.Adam(lr=lRate, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=decay)
model.compile(loss='mean_squared_error', optimizer=adam, metrics=['accuracy'])


""" Training the net """
model.fit(  trainingDataInput,
            trainingDataExpectedOutput,
            epochs=epochs,
            batch_size=batchSize,
            validation_data=(valDataInput, valDataOutput),
            callbacks=[tensorBoard]
)


""" Summary table """
print(model.summary())

evaluation = model.evaluate(valDataInput, valDataOutput)
print("\n%s: %.2f%%\n" % (model.metrics_names[1], evaluation[1]*100))


""" Testing table """
print("\tTesting table")
yPredict = model.predict(valDataInput, verbose=0)
table = PrettyTable()
table.field_names = ['x1', 'x2', 'PREDICTED', 'EXPECTED']
for i in range(0, len(yPredict)):
    table.add_row([valDataInput[i][0], valDataInput[i][1], yPredict[i][0], valDataOutput[i]])
print(table)

# opt = input('\nDo you want to save this model?: ')
# print(True if opt == 'y' else False)
