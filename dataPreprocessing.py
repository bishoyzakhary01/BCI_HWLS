import numpy as np
import scipy.io

def normalizeSentenceDataCube(sentenceDat, singleLetterDat):
    """
    Normalizes the neural data cube by subtracting means and dividing by the standard deviation. 
    Important: we use means and standard deviations from the single letter data. This is needed since we 
    initialize the HMM parameters using the single letter data, so the sentence data needs to be normalized in the same way. 
    """
    neuralCube = sentenceDat['neuralActivityCube'].astype(np.float64)

    #subtract block-specific means from each trial to counteract the slow drift in feature means over time
    for b in range(sentenceDat['blockList'].shape[0]):
        trialsFromThisBlock = np.squeeze(sentenceDat['sentenceBlockNums']==sentenceDat['blockList'][b])
        trialsFromThisBlock = np.argwhere(trialsFromThisBlock)

        closestIdx = np.argmin(np.abs(singleLetterDat['blockList'].astype(np.int32) - sentenceDat['blockList'][b].astype(np.int32)))
        blockMeans = singleLetterDat['meansPerBlock'][closestIdx,:]

        neuralCube[trialsFromThisBlock,:,:] -= blockMeans[np.newaxis,np.newaxis,:]

    #divide by standard deviation to normalize the units
    neuralCube = neuralCube / singleLetterDat['stdAcrossAllData'][np.newaxis,:,:]
    
    return neuralCube

def prepareDataCubesForLSTM(sentenceFile, singleLetterFile, labelFile, cvPartitionFile, sessionName, lstmBinSize, nTimeSteps, isTraining):
    """
    Loads raw data & HMM labels and returns training and validation data cubes for LSTM training (or inference). 
    Normalizes the neural activity using the single letter means & standard deviations.
    Does some additional pre-processing, including zero-padding the data and cutting off the end of the last character if it is too long.
    (Long pauses occur at the end of some sentences since T5 often paused briefly after finishing instead of 
    continuing immediately to the next sentence).
    """
    # Load data
    sentenceDat = scipy.io.loadmat(sentenceFile)
    slDat = scipy.io.loadmat(singleLetterFile)
    labelsDat = scipy.io.loadmat(labelFile)
    cvPart = scipy.io.loadmat(cvPartitionFile)
                      
    errWeights = 1 - labelsDat['ignoreErrorHere']
    charProbTarget = labelsDat['charProbTarget']
    charStartTarget = labelsDat['charStartTarget'][:,:,np.newaxis]

    # Update error weights and cut off end of the trial if there is a long pause
    for t in range(labelsDat['timeBinsPerSentence'].shape[0]):
        errWeights[t, labelsDat['timeBinsPerSentence'][t, 0]:] = 0
        maxPause = 150
        lastCharStart = np.argwhere(charStartTarget[t, :] > 0.5)
        if lastCharStart.size > 0:
            errWeights[t, (lastCharStart[-1, 0] + maxPause):] = 0
            labelsDat['timeBinsPerSentence'][t, 0] = (lastCharStart[-1, 0] + maxPause)

    # Combine targets
    combinedTargets = np.concatenate([charProbTarget, charStartTarget], axis=2)

    # Binning
    nLSTMOutputs = combinedTargets.shape[2]
    binsPerTrial = np.round(labelsDat['timeBinsPerSentence'] / lstmBinSize).astype(np.int32)
    binsPerTrial = np.squeeze(binsPerTrial)

    # Normalize neural data
    neuralData = normalizeSentenceDataCube(sentenceDat, slDat)

    # Bin data across the time axis
    if lstmBinSize > 1:
        neuralData = binTensor(neuralData, lstmBinSize)
        combinedTargets = binTensor(combinedTargets, lstmBinSize)
        errWeights = np.squeeze(binTensor(errWeights[:,:,np.newaxis], lstmBinSize))

    # Zero padding
    if isTraining:
        edgeSpace = (nTimeSteps - 100)
        padTo = neuralData.shape[1] + edgeSpace * 2
        
        padNeuralData = np.zeros([neuralData.shape[0], padTo, neuralData.shape[2]])
        padCombinedTargets = np.zeros([combinedTargets.shape[0], padTo, combinedTargets.shape[2]])
        padErrWeights = np.zeros([errWeights.shape[0], padTo])

        padNeuralData[:, edgeSpace:(edgeSpace + neuralData.shape[1]), :] = neuralData
        padCombinedTargets[:, edgeSpace:(edgeSpace + combinedTargets.shape[1]), :] = combinedTargets
        padErrWeights[:, edgeSpace:(edgeSpace + errWeights.shape[1])] = errWeights
    else:
        padTo = nTimeSteps

        padNeuralData = np.zeros([neuralData.shape[0], padTo, neuralData.shape[2]])
        padCombinedTargets = np.zeros([combinedTargets.shape[0], padTo, combinedTargets.shape[2]])
        padErrWeights = np.zeros([errWeights.shape[0], padTo])

        padNeuralData[:, 0:neuralData.shape[1], :] = neuralData
        padCombinedTargets[:, 0:combinedTargets.shape[1], :] = combinedTargets
        padErrWeights[:, 0:errWeights.shape[1]] = errWeights

    # Gather the train/validation fold indices
    cvIdx = {}
    cvIdx['trainIdx'] = np.squeeze(cvPart[sessionName + '_train'])
    cvIdx['testIdx'] = np.squeeze(cvPart[sessionName + '_test'])

    return padNeuralData, padCombinedTargets, padErrWeights, binsPerTrial, cvIdx

def binTensor(data, binSize):
    """
    A simple utility function to bin a 3d numpy tensor along axis 1 (the time axis here). Data is binned by
    taking the mean across a window of time steps. 
    
    Args:
        data (tensor : B x T x N): A 3d tensor with batch size B, time steps T, and number of features N
        binSize (int): The bin size in # of time steps
        
    Returns:
        binnedTensor (tensor : B x S x N): A 3d tensor with batch size B, time bins S, and number of features N.
                                           S = floor(T/binSize)
    """
    
    nBins = np.floor(data.shape[1]/binSize).astype(int)
    
    sh = np.array(data.shape)
    sh[1] = nBins
    binnedTensor = np.zeros(sh)
    
    binIdx = np.arange(0,binSize).astype(int)
    for t in range(nBins):
        binnedTensor[:,t,:] = np.mean(data[:,binIdx,:],axis=1)
        binIdx += binSize;
    
    return binnedTensor