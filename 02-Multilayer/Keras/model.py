from keras.models import Sequential
from keras.callbacks import ModelCheckpoint
from keras.layers import Dense
from keras import optimizers
from prettytable import PrettyTable
import csv

class KerasModel:
    def __init__(self, layers, callbacks=None):
        self.__dict__['_layers'] = layers
        self.__dict__['_callbacks'] = callbacks
        self.__dict__['_model'] = None

    def createModel(self, lRate, decay):
        model = Sequential()
        """ Every layer is sigmoidal but last one, which is linear.
            Linear function is necessary to return value from bigger range.

            model.add(Dense(...)) - appending layers.
        """
        for i in range(len(self._layers)):
            if i == 0:
                model.add(Dense(self._layers[i], input_dim=2, activation='sigmoid'))
            elif i == len(self._layers) - 1:
                model.add(Dense(self._layers[i], activation='linear'))
            else:
                model.add(Dense(self._layers[i], activation='sigmoid'))
        """ Adding optimizer and MSE as a loss """
        adam = optimizers.Adam(lr=lRate, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=decay)
        """ Compiling model """
        model.compile(loss='mean_squared_error', optimizer=adam, metrics=['accuracy'])
        self._model = model

    def train(self, trainingDataInput, trainingDataExpectedOutput, epochs, batchSize, validationData):

        """ Training the net """
        self._model.fit(  trainingDataInput,
                    trainingDataExpectedOutput,
                    epochs=epochs,
                    batch_size=batchSize,
                    validation_data=validationData,
                    callbacks=self._callbacks
        )

    def printEvaluation(self, valData, CSVFormat=False):
        if CSVFormat == False:
            evaluation = self._model.evaluate(valData[0], valData[1])
            print("\n%s: %.2f%%\n" % (self._model.metrics_names[1], evaluation[1]*100))

            """ Testing table """
            print("\tTesting table")
            yPredict = self._model.predict(valData[0], verbose=0)
            table = PrettyTable()
            table.field_names = ['x1', 'x2', 'PREDICTED', 'EXPECTED']
            for i in range(0, len(yPredict)):
                table.add_row([valData[0][i][0], valData[0][i][1], yPredict[i][0], valData[1][i]])
            print(table)
        else:
            print("\tWiriting to csv file...")
            yPredict = self._model.predict(valData[0], verbose=0)
            with open('./docs/best_validation_data.csv', 'w') as csvfile:
                valWriter = csv.writer(csvfile, delimiter=',',
                            quotechar='|', quoting=csv.QUOTE_MINIMAL)
                valWriter.writerow(['X1', 'X2', 'PREDICTED', 'EXPECTED'])
                for i in range(0, len(yPredict)):
                    valWriter.writerow([valData[0][i][0], valData[0][i][1], yPredict[i][0], valData[1][i]])


    def loadWeights(self, weightsPath):
        self._model.load_weights(weightsPath)

    """ Access method """
    def __getitem__(self, index):
        if index == 'model':
            return self._model
