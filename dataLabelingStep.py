#--Utility functions for the data labeling step--
import numpy as np
import scipy.ndimage.filters
import multiprocessing
from forcedAlignmentHMM import forcedAlignmentLabeling, initializeCharacterTemplates
from dataPreprocessing import normalizeSentenceDataCube
    
def labelDataset(sentenceDat, singleLetterDat, twCubes, trainPartitionIdx, testPartitionIdx, charDef):
    """
    La funzione labelDataset serve per etichettare i dati neurali relativi a diverse frasi e per aggiornare le probabilità di emissione del Modello di Markov Nascosto (HMM) basato sui dati etichettati. Etichetta sia i dati di addestramento che quelli di test, ma aggiorna l'HMM solo utilizzando i dati di addestramento.
    Labels all sentences in the dataset 'sentenceDat', using the single character data 'singleLetterDat' and
    time-warped characters 'twCubes' to initialize the HMM emission probabilities. Uses only the sentences specified
    by 'trainPartitionIdx' to update the HMM probabilities. This function will label all sentences in 'trainPartitionIdx'
    AND 'testPartitionIdx', but will only update the HMM using sentences from 'trainPartitionIdx'. 
    
    Args:
        sentenceDat (dict): The sentences dataset returned by scipy.io.loadmat.
        singleLetterDat (dict): Single letter data dictionary
        twCubes (dict): A dictionary of matrices defining the spatiotemporal pattern of neural activity for each character,
                  indexed by that character's string representation.
        trainPartitionIdx (vector : C x 1): A vector containing the index of each sentence that belongs to the training set
        testPartitionIdx (vector : D x 1): A vector containing the index of each sentence that belongs to the test set
        charDef (dict): A definition of character names and lengths (see characterDefinitions.py)
        
    Returns:
        letterStarts (matrix : N x 200) A matrix of character start times, each row corresponds to a single sentence.
        letterDurations (matrix : N x 200): A matrix of character durations, each row corresponds to a single sentence.
        neuralCube (matrix : N x T x E): A normalized, smooth neural activity cube (N = # of sentences, T = # of time steps, 
                                         E = # of electrodes)
        blankWindows (list): A nested list of time windows where periods of 'blank' pauses occur in the data. This is used to extract
                             'blank' snippets for simulating pauses. 
    """
            
    #Prepare the neural data to be processed by the HMM by normalizing, smoothing and binning it.
    normalizedNeuralCube = normalizeSentenceDataCube(sentenceDat, singleLetterDat)

    #smooth neural activity (important!)
    neuralCubeSmoothed = scipy.ndimage.filters.gaussian_filter1d(normalizedNeuralCube, 4.0, axis=1)

    #initialize spatiotemporal activity templates based on time-warped single letter data (computed in step 1)
    templates = initializeCharacterTemplates(twCubes, charDef)

    nSentences = sentenceDat['neuralActivityCube'].shape[0]
    letterStarts = np.zeros([nSentences, 200])
    letterDurations = np.zeros([nSentences, 200])
    blankWindows = []
    for x in range(nSentences):
        blankWindows.append([])

    #update the emission probabilities only once
    nHMMIters = 2
    
    #we will label all trials in the train AND test set, but only use the train set to update the HMM
    if len(testPartitionIdx)>0:
        trainAndTestIdx = np.squeeze(np.concatenate([trainPartitionIdx, testPartitionIdx], axis=1))
    else:
        trainAndTestIdx = np.squeeze(trainPartitionIdx)
    
    #iteratively label the data, updating the HMM emission probabilities with each iteration
    for iterIdx in range(nHMMIters):
        print('HMM Iteration ' + str(iterIdx))
        
        #--data labeling in parallel--
        #construct  a list of arguments for the 'forcedAlignmnetLabeling' function (one element for each sentence)
        args = []
        for x in trainAndTestIdx:                
            args.append([neuralCubeSmoothed[x,0:sentenceDat['numTimeBinsPerSentence'][x,0],:],
                         sentenceDat['sentencePrompt'][x,0][0],
                         templates])
        
        #label the sentences in parallel
        pool = multiprocessing.Pool(int(np.ceil(multiprocessing.cpu_count()/2)))
        results = pool.starmap(forcedAlignmentLabeling, args)
        
        #unpack the list of results 
        for x in range(len(results)):
            sentence = sentenceDat['sentencePrompt'][trainAndTestIdx[x],0][0]
            letterStarts[trainAndTestIdx[x],0:len(sentence)] = results[x][0][:,0]
            letterDurations[trainAndTestIdx[x],0:len(sentence)] = results[x][1][:,0]
            blankWindows[trainAndTestIdx[x]] = results[x][2]
            
        #close the pool
        pool.close()
        pool.join()
        
        #--update the templates--
        #first, initialize new templates 
        newTemplates = {}
        charCount = {}

        charList = list(templates.keys())
        for x in charList:
            newTemplates[x] = np.zeros(templates[x].shape)
            charCount[x] = 0

        #loop through each labeled character, adding it to newTemplates
        for x in np.squeeze(trainPartitionIdx):
            sentence = sentenceDat['sentencePrompt'][x,0][0]
            nChars = len(sentence)
            for c in range(nChars):
                #get the snippet of neural data corresponding to this character
                timeSteps = np.arange(letterStarts[x,c], letterStarts[x,c]+letterDurations[x,c]).astype(np.int32)
                timeSteps = timeSteps[timeSteps<neuralCubeSmoothed.shape[1]]

                neuralDat = neuralCubeSmoothed[x, timeSteps, :]

                #resample to normalize the number of time steps
                resampleDat = np.zeros(templates[sentence[c]].shape)
                for e in range(resampleDat.shape[1]):
                    resampleDat[:,e] = np.interp(np.linspace(0,1,resampleDat.shape[0]),
                                                np.linspace(0,1,neuralDat.shape[0]),
                                                neuralDat[:,e])

                #add to the running sum (we divide at the end to compute the mean)
                newTemplates[sentence[c]] += resampleDat
                charCount[sentence[c]] += 1

        #divide by the total number of characters per template to compute the mean
        for x in charList:
            if charCount[x]>0: #avoid dividing by zero and causing a warning message for rare letters that may have no examples
                newTemplates[x] /= charCount[x]

        #for characters that occur too infrequently, stick with the original template
        for x in charList:
            if charCount[x]<10:
                newTemplates[x] = templates[x].copy()

        #onwards to the next iteration
        templates = newTemplates
        
    return letterStarts, letterDurations, blankWindows

import torch
import torch.nn as nn
import torch.optim as optim

def preparedataforLSTM(letterStarts, letterDurations, maxTimeSteps, sentences, charDef):
    '''
    La funzione prepare_data_for_pytorch elabora e converte i dati in
    tensori PyTorch, creando i target necessari per l'allenamento 
    di un modello di rete neurale. La funzione genera:
    charProbTarget: Le probabilità previste dei caratteri in ogni passo temporale.
    charStartTarget: Un segnale binario che indica l'inizio di un carattere.
    ignoreErrorHere: Un maschera per ignorare gli errori nei passi temporali iniziali.
    '''
    
    # Convert the numpy arrays to PyTorch tensors
    nSentences = len(sentences)
    charProbTarget = torch.zeros(nSentences, maxTimeSteps, len(charDef['charList']))
    charStartTarget = torch.zeros(nSentences, maxTimeSteps)
    ignoreErrorHere = torch.zeros(nSentences, maxTimeSteps)
    #Creazione dei Target 
    #Il ciclo for successivo elabora ogni frase e crea i target:
    for sentenceIdx in range(nSentences):
        sentence = sentences[sentenceIdx][0]
        
        for x in range(len(sentence)):
            if x < (len(sentence) - 1):
                stepIdx = torch.arange(letterStarts[sentenceIdx, x], letterStarts[sentenceIdx, x + 1])
            else:
                stepIdx = torch.arange(letterStarts[sentenceIdx, x], maxTimeSteps)
                
            charIdx = torch.tensor(charDef['charListAbbr']).eq(sentence[x]).nonzero(as_tuple=True)[0]
            charProbTarget[sentenceIdx, stepIdx, charIdx] = 1
            
            stepIdx = torch.arange(letterStarts[sentenceIdx, x], letterStarts[sentenceIdx, x] + 21)
            charStartTarget[sentenceIdx, stepIdx] = 1
        
        charIdx = torch.tensor(charDef['charListAbbr']).eq(sentence[0]).nonzero(as_tuple=True)[0]
        charProbTarget[sentenceIdx, 0:letterStarts[sentenceIdx, 0].astype(int), charIdx] = 1
        ignoreErrorHere[sentenceIdx, 0:letterStarts[sentenceIdx, 0].astype(int)] = 1

    return charStartTarget, charProbTarget, ignoreErrorHere
