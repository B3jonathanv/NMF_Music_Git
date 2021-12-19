
import os
import numpy as np
import scipy.io.wavfile as wav
import matplotlib.pyplot as plt
import IPython.display as ipd

from NMFtoolbox.python.NMFtoolbox.forwardSTFT import forwardSTFT
from NMFtoolbox.python.NMFtoolbox.inverseSTFT import inverseSTFT
from NMFtoolbox.python.NMFtoolbox.initTemplates import initTemplates
from NMFtoolbox.python.NMFtoolbox.initActivations import initActivations
from NMFtoolbox.python.NMFtoolbox.NMF import NMF
from NMFtoolbox.python.NMFtoolbox.alphaWienerFilter import alphaWienerFilter
from NMFtoolbox.python.NMFtoolbox.visualizeComponentsNMF import visualizeComponentsNMF
from NMFtoolbox.python.NMFtoolbox.utils import make_monaural, pcmInt16ToFloat32Numpy

inpPath = 'NMFtoolbox/data/'
outPath = 'output/'

# create the output directory if it doesn't exist
if not os.path.isdir(outPath):
    os.makedirs(outPath)

# convert wav from int16 to float32
filename = 'eqt-major-sc.wav'


# 1. Load the audio signal


fs, x = wav.read(os.path.join(inpPath, filename))

# make monaural if necessary
x = make_monaural(x)

x = pcmInt16ToFloat32Numpy(x)

# 2. compute STFT


# spectral parameters
paramSTFT = dict()
paramSTFT['blockSize'] = 2048
paramSTFT['hopSize'] = 512
paramSTFT['winFunc'] = np.hanning(paramSTFT['blockSize'])
paramSTFT['reconstMirror'] = True
paramSTFT['appendFrame'] = True
paramSTFT['numSamples'] = len(x)

# STFT computation
X, A, P = forwardSTFT(x, paramSTFT)

# get dimensions and time and freq resolutions
numBins, numFrames = X.shape
deltaT = paramSTFT['hopSize'] / fs
deltaF = fs / paramSTFT['blockSize']

# 3. Apply NMF variants to STFT magnitude


# set common parameters
numComp = 8
numIter = 50
numTemplateFrames = 8

# generate initial guess for templates
paramTemplates = dict()
paramTemplates['deltaF'] = deltaF
paramTemplates['numComp'] = numComp
paramTemplates['numBins'] = numBins
# paramTemplates['numTemplateFrames'] = numTemplateFrames
initW = initTemplates(paramTemplates, 'random')

# generate initial activations
paramActivations = dict()
paramActivations['numComp'] = numComp
paramActivations['numFrames'] = numFrames

initH = initActivations(paramActivations, 'uniform')

# NMF parameters
paramNMF = dict()
paramNMF['costFunc'] = 'EucDist'
paramNMF['reg'] = 'None'
paramNMF['p'] = 0.000005
paramNMF['numComp'] = numComp
paramNMF['numFrames'] = numFrames
paramNMF['numIter'] = numIter
paramNMF['numTemplateFrames'] = numTemplateFrames
paramNMF['initW'] = np.concatenate(initW, axis=1)
paramNMF['initH'] = initH

# NMFD core method
noise_level = 0
A = A + noise_level * np.linalg.norm(A) * abs(np.random.randn(*A.shape))
nmfW, nmfH, nmfV, f, w_change, h_change = NMF(A, paramNMF)
print(f)

# alpha-Wiener filtering
nmfA, _ = alphaWienerFilter(A, nmfV, 1.0)



#Graphing the change in f
x = range(0, numIter+1)
plt.figure(1)
plt.plot(x, f)
plt.yscale('log')
plt.xlabel('Iteration Number')
plt.ylabel('Value of Objective Function')
plt.title('Change in Objective Function Between Iterations')
plt.show()

#Graphing the change in W
plt.figure(2)
x = range(1, numIter+1)
plt.plot(x, w_change)
plt.yscale('log')
plt.xlabel('Iteration Number')
plt.ylabel('Change in W')
plt.title('Change in W Between Iterations')
plt.show()


#Graphing the change in H
plt.figure(3)
x = range(1, numIter+1)
plt.plot(x, h_change)
plt.yscale('log')
plt.xlabel('Iteration Number')
plt.ylabel('Change of H')
plt.title('Change in H Between Iterations')
plt.show()


#visualize the spectrogram, W and H
paramVis = dict()
paramVis['deltaT'] = deltaT
paramVis['deltaF'] = deltaF
#paramVis['endeSec'] = 3.8
#paramVis['fontSize'] = 14
fh1, _ = visualizeComponentsNMF(A, nmfW, nmfH, nmfA, paramVis)
plt.show()