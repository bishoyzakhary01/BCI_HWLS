import numpy as np
import torch
import torch.nn.functional as F
import scipy.special

def evaluateLSTMOutput(lstmOutput, numBinsPerSentence, trueText, charDef, charStartThresh=0.3, charStartDelay=15):
    """
    Converts the LSTM output (character probabilities & a character start signal) into a discrete sentence and computes
    char/word error rates. Returns error counts and the decoded sentences.
    """  
    # Separate character logits and character start signal
    lgit = lstmOutput[:, :, :-1]  # Character probabilities
    charStart = lstmOutput[:, :, -1]  # Character start signal

    # Convert logits to probabilities using softmax
    charProb = F.softmax(lgit, dim=-1).cpu().detach().numpy()

    # Convert output to character strings
    decStr = decodeCharStr(charProb, charStart.cpu().detach().numpy(), charStartThresh, charStartDelay, 
                           numBinsPerSentence, charDef['charListAbbr'])

    allErrCounts = {
        'charCounts': np.zeros([len(trueText)], dtype=int),
        'charErrors': np.zeros([len(trueText)], dtype=int),
        'wordCounts': np.zeros([len(trueText)], dtype=int),
        'wordErrors': np.zeros([len(trueText)], dtype=int),
    }
    
    allDecSentences = []

    # Compute error rates
    for t in range(len(trueText)):
        thisTrueText = trueText[t, 0][0]
        thisTrueText = thisTrueText.replace(' ', '')
        thisTrueText = thisTrueText.replace('>', ' ')
        thisTrueText = thisTrueText.replace('~', '.')
        thisTrueText = thisTrueText.replace('#', '')

        thisDec = decStr[t]
        thisDec = thisDec.replace('>', ' ')
        thisDec = thisDec.replace('~', '.')

        nCharErrors = wer(list(thisTrueText), list(thisDec))
        nWordErrors = wer(thisTrueText.strip().split(), thisDec.strip().split())
        
        allErrCounts['charCounts'][t] = len(thisTrueText)
        allErrCounts['charErrors'][t] = nCharErrors
        allErrCounts['wordCounts'][t] = len(thisTrueText.strip().split())
        allErrCounts['wordErrors'][t] = nWordErrors

        allDecSentences.append(thisDec)

    return allErrCounts, allDecSentences

def decodeCharStr(logitMatrix, transSignal, transThresh, transDelay, numBinsPerTrial, charList):
    """
    Converts the LSTM output (character probabilities & a character start signal) into a discrete sentence.
    """
    decWords = []
    for v in range(logitMatrix.shape[0]):
        logits = np.squeeze(logitMatrix[v, :, :])
        bestClass = np.argmax(logits, axis=1)
        letTrans = scipy.special.expit(transSignal[v, :])

        endIdx = np.ceil(numBinsPerTrial[v]).astype(int)
        letTrans = letTrans[:endIdx]

        transIdx = np.argwhere(np.logical_and(letTrans[:-1] < transThresh, letTrans[1:] > transThresh)).flatten()
        
        wordStr = ''
        for x in range(len(transIdx)):
            index = transIdx[x] + transDelay
            if index < len(bestClass):
                wordStr += charList[bestClass[index]]

        decWords.append(wordStr)
        
    return decWords

def wer(r, h):
    """
    Calculation of WER with Levenshtein distance.

    Works only for iterables up to 254 elements (uint8).
    O(nm) time ans space complexity.

    Parameters
    ----------
    r : list
    h : list

    Returns
    -------
    int

    Examples
    --------
    >>> wer("who is there".split(), "is there".split())
    1
    >>> wer("who is there".split(), "".split())
    3
    >>> wer("".split(), "who is there".split())
    3
    """
    # initialisation
    import numpy
    d = numpy.zeros((len(r)+1)*(len(h)+1), dtype=numpy.uint8)
    d = d.reshape((len(r)+1, len(h)+1))
    for i in range(len(r)+1):
        for j in range(len(h)+1):
            if i == 0:
                d[0][j] = j
            elif j == 0:
                d[i][0] = i

    # computation
    for i in range(1, len(r)+1):
        for j in range(1, len(h)+1):
            if r[i-1] == h[j-1]:
                d[i][j] = d[i-1][j-1]
            else:
                substitution = d[i-1][j-1] + 1
                insertion    = d[i][j-1] + 1
                deletion     = d[i-1][j] + 1
                d[i][j] = min(substitution, insertion, deletion)

    return d[len(r)][len(h)]
import numpy as np
import torch
import scipy.special

def lstmOutputToKaldiMatrices(lstmOutput, numBinsPerSentence, charDef, kaldiDir):
    """
    Converts the LSTM output into probability matrices that Kaldi can read, one for each sentence.
    As part of the conversion, this function creates a CTC blank signal from the character start signal so
    that the language model is happy (it was designed for a CTC loss).
    """
    # Separate logits and character start signal
    lgit = lstmOutput[:, :, :-1]  # Character probabilities (logits)
    charStart = lstmOutput[:, :, -1]  # Character start signal

    # Convert logits to probabilities using softmax
    charProb = torch.softmax(torch.tensor(lgit), dim=-1).cpu().detach().numpy()

    # Create a fake CTC blank signal
    fakeCTC = np.ones(charStart.shape)
    fakeCTC[:, 20:] = 1 - scipy.special.expit(4 + 4 * charStart[:, :-20])
    
    nChar = lstmOutput.shape[2] - 1
    probCombined = np.concatenate([charProb, fakeCTC[:, :, np.newaxis]], axis=2)
    probCombined[:, :, :nChar] *= 1 - fakeCTC[:, :, np.newaxis]
    
    allMatrices = []
    for t in range(lstmOutput.shape[0]):
        startIdx = 0
        endIdx = int(numBinsPerSentence[t, 0])
        charProb = np.transpose(probCombined[t, startIdx:endIdx:5, charDef['idxToKaldi']])

        charProb[charProb == 0] = 1e-13
        charProb = np.log(charProb)

        # Write the Kaldi matrix
        writeKaldiProbabilityMatrix(charProb, t, kaldiDir + 'kaldiMat_' + str(t) + '.txt')
        allMatrices.append(charProb)
        
    return allMatrices

def writeKaldiProbabilityMatrix(matrix, index, filePath):
    """
    Write the Kaldi probability matrix to a file in Kaldi format.
    """
    with open(filePath, 'w') as f:
        for row in matrix:
            f.write(' '.join(map(str, row)) + '\n')