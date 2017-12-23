from neurons import *
import numpy as np
from emojis import *

#np.random.seed(5)
neuronGrid = [15, 15]
lRate=0.007
fRate=0.4

emoji = Emoji()

def countUniqueItems(arr):
    return len(Counter(arr).keys())

trainingSet = [
    emoji.getEmoji('sad'),
    emoji.getEmoji('sad_noised'),
    emoji.getEmoji('smile'),
    emoji.getEmoji('angry'),
    emoji.getEmoji('xD'),
    emoji.getEmoji('confused'),
    emoji.getEmoji('test')
]


hebbGroup = HebbNeuronGroup(
    numOfInputs=64,
    numOfNeurons=neuronGrid,
    processFunc=SignSigm()(0.5),
    lRateFunc=Linear()(),
    lRate=lRate,
    fRate=fRate
)

# for neuron in hebbNrs:
#     neuron.setTrainingData(trainingSet)

winners = []
for i in range(100):
    winners = hebbGroup.train(trainingSet)

for winner in winners:
    print(winner._iid)

print('Active neurons: {:d}'.format(countUniqueItems(winners)))
