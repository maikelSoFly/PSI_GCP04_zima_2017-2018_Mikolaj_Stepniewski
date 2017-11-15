from keras.models import Sequential
from keras.callbacks import ModelCheckpoint
from keras.layers import Dense
from keras import optimizers
from prettytable import PrettyTable

class KerasModel:
    def __init__(self, layers, callbacks=None):
        self.__dict__['_layers'] = layers
        self.__dict__['_callbacks'] = callbacks
        self.__dict__['_model'] = None

    def createModel(self):
        model = Sequential()
        for i in range(len(self._layers)):
            if i == 0:
                model.add(Dense(self._layers[i], input_dim=2, activation='sigmoid'))
            elif i == len(self._layers) - 1:
                model.add(Dense(self._layers[i], activation='linear'))
            else:
                model.add(Dense(self._layers[i], activation='sigmoid'))
        self._model = model

    def train(self, lRate, decay, trainingDataInput, trainingDataExpectedOutput, epochs, batchSize, validationData):
        adam = optimizers.Adam(lr=lRate, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=decay)
        self._model.compile(loss='mean_squared_error', optimizer=adam, metrics=['accuracy'])

        """ Training the net """
        self._model.fit(  trainingDataInput,
                    trainingDataExpectedOutput,
                    epochs=epochs,
                    batch_size=batchSize,
                    validation_data=validationData,
                    callbacks=self._callbacks
        )

    def printEvaluation(self, valData):
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


    def __getitem__(self, index):
        if index == 'model':
            return self._model
