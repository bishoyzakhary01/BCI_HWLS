
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from scipy.io import loadmat

def generateCharacterSequences(args):
    np.random.seed(args['seed'])

    # Load character snippets
    char_snippets = loadmat(args['snippetFile'])

    # Generate synthetic sentences
    synth_neural, synth_char_prob, synth_char_start = makeSyntheticDataFromRawSnippets(
        char_def=args['charDef'], 
        char_snippets=char_snippets, 
        n_sentences=args['nSentences'], 
        n_steps=args['nSteps'] + 2000, 
        word_list=args['wordListFile'], 
        blank_prob=0.20,
        account_for_pen_state=args['accountForPenState']
    )

    # Truncate and prepare training sequences
    synth_neural_cut = np.zeros([args['nSentences'], args['nSteps'], synth_neural.shape[2]])
    synth_targets_cut = np.zeros([args['nSentences'], args['nSteps'], synth_char_prob.shape[2] + 1])

    for t in range(args['nSentences']):
        rand_start = np.random.randint(args['nSteps'])
        synth_neural_cut[t, :, :] = synth_neural[t, rand_start:(rand_start + args['nSteps']), :]
        synth_targets_cut[t, :, :-1] = synth_char_prob[t, rand_start:(rand_start + args['nSteps']), :]
        synth_targets_cut[t, :, -1] = synth_char_start[t, rand_start:(rand_start + args['nSteps'])]

    # Convert to PyTorch tensors
    synth_neural = torch.tensor(synth_neural_cut, dtype=torch.float32)
    synth_targets = torch.tensor(synth_targets_cut, dtype=torch.float32)

    return synth_neural, synth_targets


import numpy as np

def makeSyntheticDataFromRawSnippets(charDef, charSnippets, nSentences, nSteps, wordList, 
                                     blankProb=0.05, accountForPenState=True, 
                                     rareLetterIncrease=False, rareWordList=[]):
    """
    Generates synthetic data by arranging character snippets from 'charSnippets' into random sentences.
    
    Args:
        charDef (dict): Definition of character names and lengths.
        charSnippets (dict): Library of neural data snippets corresponding to single characters.
        nSentences (int): Number of sentences to generate.
        nSteps (int): Number of time steps per sentence.
        wordList (list): List of valid words for generating sentences.
        blankProb (float): Probability of generating a 'blank' pause.
        accountForPenState (bool): If true, ensures continuity of pen movements between characters.
        rareLetterIncrease (bool): Increases the frequency of rare letter words.
        rareWordList (list): List of word indices containing rare letters.
    
    Returns:
        synthNeural (matrix : N x T x E): Synthetic neural data tensor.
        synthCharProb (matrix : N x T x C): Character probability target tensor.
        synthCharStart (matrix : N x T): Character start signal target tensor.
    """

    nNeurons = charSnippets['a'][0,0].shape[1]
    nClasses = len(charDef['charList'])

    # Initialize tensors
    synthNeural = np.zeros([nSentences, nSteps, nNeurons])
    synthCharProb = np.zeros([nSentences, nSteps, nClasses])
    synthCharStart = np.zeros([nSentences, nSteps])

    for t in range(nSentences):
        currIdx = 0
        currentWord = []
        currentLetterIdx = 0
        
        # Generate sentence one character at a time
        while currIdx < nSteps:
            # Pick a new word if needed
            if currentLetterIdx >= len(currentWord):
                currentLetterIdx = 0
                currentWord = pickWordForSentence(wordList, rareLetterIncrease=rareLetterIncrease, 
                                                  rareWordList=rareWordList)

            # Pick the character snippet for the current character
            classIdx = charDef['strToCharIdx'][currentWord[currentLetterIdx]]
            
            if (currentLetterIdx < len(currentWord) - 1) and accountForPenState:
                nextClassIdx = charDef['strToCharIdx'][currentWord[currentLetterIdx + 1]]
                nextPenStartLoc = charDef['penStart'][nextClassIdx]

                penEndStates = charSnippets[charDef['charList'][classIdx] + '_penEndState']
                validIdx = np.argwhere(np.logical_or(penEndStates[0, :] == nextPenStartLoc, penEndStates[0, :] < -1.5))
                
                if validIdx.shape[0] == 0:
                    choiceIdx = np.random.randint(len(charSnippets[charDef['charList'][classIdx]][0]))
                else:
                    choiceIdx = validIdx[np.random.randint(len(validIdx))][0]
            else:
                choiceIdx = np.random.randint(len(charSnippets[charDef['charList'][classIdx]][0]))

            # Fetch and process the selected character snippet
            currentSnippet = charSnippets[charDef['charList'][classIdx]][0, choiceIdx].copy()
            useIdx = np.logical_not(np.isnan(currentSnippet[:, 0]))
            currentSnippet = currentSnippet[useIdx, :]
            
            # Time-warping and scaling
            charLen = currentSnippet.shape[0]
            nStepsForChar = np.round(charLen * 0.7 + np.random.randint(charLen * 0.6))

            tau = np.linspace(0, currentSnippet.shape[0] - 1, int(nStepsForChar)).astype(int)
            currentSnippet = currentSnippet[tau, :]

            randScale = 0.7 + 0.6 * np.random.rand()
            currentSnippet *= randScale

            # Add blank pauses randomly
            if np.random.rand(1) < blankProb:
                blankData = charSnippets['blank'][0, np.random.randint(charSnippets['blank'].shape[1])]
                currentSnippet = np.concatenate([currentSnippet, blankData], axis=0)

            # Generate probability targets
            labels = np.zeros([currentSnippet.shape[0], nClasses])
            labels[:, classIdx] = 1

            # Fill data tensors
            nNewSteps = currentSnippet.shape[0]
            if nNewSteps + currIdx >= nSteps:
                stepLimit = nSteps - currIdx
                currentSnippet = currentSnippet[:stepLimit, :]
                labels = labels[:stepLimit, :]

            synthNeural[t, currIdx:currIdx + currentSnippet.shape[0], :] = currentSnippet
            synthCharProb[t, currIdx:currIdx + currentSnippet.shape[0], :] = labels
            synthCharStart[t, currIdx:(currIdx + 20)] = 1  # Assume 20 steps for start signal
            
            # Advance to the next character
            currIdx += nNewSteps
            currentLetterIdx += 1
            
    return synthNeural, synthCharProb, synthCharStart
import numpy as np

def pickWordForSentence(wordList, rareLetterIncrease=False, rareWordList=[]):
    """
    Implements a simple heuristic for randomly choosing a word for the next position in a sentence.
    Each word is chosen independently of the previous ones to avoid teaching the LSTM a language model.
    
    Args:
        wordList (list): List of possible words.
        rareLetterIncrease (bool): If true, increases the frequency of words with rare letters.
        rareWordList (list): List of indices pointing to words in 'wordList' with rare letters ('x', 'z', 'q', 'j').
    
    Returns:
        nextWord (list): A list of characters representing the randomly chosen word.
    """
    
    # Choose a new word
    if np.random.rand() < 0.2:
        # Choose a high-frequency word from the top 20
        wordIdx = np.random.randint(20)
    elif rareLetterIncrease and np.random.rand() < 0.2:
        # Choose a word with a rare letter
        rareIdx = np.random.randint(len(rareWordList))
        wordIdx = rareWordList[rareIdx]
    else:
        # Choose any word from the full list
        wordIdx = np.random.randint(len(wordList))

    nextWord = list(wordList[wordIdx])

    # With low probability, place an apostrophe before the last character in the word
    if np.random.rand() < 0.03 and len(nextWord) > 3:
        nextWord.insert(len(nextWord) - 1, "'")

    # Randomly add a comma, period, or question mark at the end of the word
    punctuation = None
    if np.random.rand() < 0.07:
        punctuation = ','
    elif np.random.rand() < 0.05:
        punctuation = '~'
    elif np.random.rand() < 0.05:
        punctuation = '?'

    if punctuation:
        nextWord.append(punctuation)
    else:
        # Add a space if no punctuation is added
        nextWord.append('>')

    return nextWord


def extractCharacterSnippets(letterStarts, blankWindows, neuralCube, sentences, sentenceLens, trainPartitionIdx, charDef):
    """
    Constructs the time series 'targets' used to train the lstm. The lstm is trained using supervised learning to 
    produce two outputs: a character probability vector with a one-hot encoding of the current character, 
    and a binary 'new character' signal which briefly goes high at the start of any new character. 

    This function also generates an 'ignoreError' mask to prevent the lstm from being penalized for errors that occur at the start of the trial, 
    before any characters are written (labeled as a 'blank' state by the HMM).
    
    Args:
        letterStarts (matrix : N x 200): Matrix of character start times; each row corresponds to a sentence.
        blankWindows (list): Nested list of time windows where 'blank' pauses occur, used to extract 'blank' snippets.
        neuralCube (matrix : N x T x E): Normalized, smooth neural activity (N = # of sentences, T = # of time steps, E = # of electrodes).
        sentences (vector : N x 1): Array of sentences.
        sentenceLens (vector : N x 1): Array of sentence lengths (number of time steps per sentence).
        trainPartitionIdx (vector : C x 1): Index vector for sentences belonging to the training set.
        charDef (dict): Definition of character names and lengths (from characterDefinitions.py).
        
    Returns:
        snippetDict (dict): Dictionary containing character snippets and an estimate of where the pen tip ended for each snippet.
    """
    
    # Initialize the snippet dictionary
    snippetDict = {char: [] for char in charDef['charList']}
    for char in charDef['charList']:
        snippetDict[char + '_penEndState'] = []
    snippetDict['blank'] = []

    # Iterate through each sentence to extract character snippets
    for sentIdx in range(len(sentences)):
        # Skip sentences not in the training data
        if not np.any(trainPartitionIdx == sentIdx):
            continue

        # Get the number of characters in the sentence
        nChars = len(sentences[sentIdx][0])

        # Extract each character one by one
        for charIdx in range(nChars):
            # Get the time indices for the character snippet
            if charIdx < nChars - 1:
                loopIdx = np.arange(letterStarts[sentIdx, charIdx], letterStarts[sentIdx, charIdx + 1]).astype(np.int32)
            else:
                loopIdx = np.arange(letterStarts[sentIdx, charIdx], sentenceLens[sentIdx]).astype(np.int32)

            # Extract the neural data for the current character
            newCharDat = neuralCube[sentIdx, loopIdx, :]

            # Estimate the pen state at the end of the snippet
            if charIdx < nChars - 1:
                nextChar = sentences[sentIdx][0][charIdx + 1]
                nextCharIdx = np.squeeze(np.argwhere(np.array(charDef['charListAbbr']) == nextChar))
                nextCharStartPenState = charDef['penStart'][nextCharIdx]
            else:
                nextCharStartPenState = -1  # Indicating no next character

            # Get the current character's name
            thisChar = sentences[sentIdx][0][charIdx]
            thisCharIdx = np.squeeze(np.argwhere(np.array(charDef['charListAbbr']) == thisChar))
            thisCharName = charDef['charList'][thisCharIdx]
            
            # Add the character data and pen end state to the dictionary
            snippetDict[thisCharName].append(newCharDat)
            snippetDict[thisCharName + '_penEndState'].append(nextCharStartPenState)

        # Extract 'blank' snippets
        bw = blankWindows[0, sentIdx]
        for blankIdx in bw:
            if blankIdx.size > 0:  # Ensure there are elements to process
                validIndices = blankIdx[blankIdx < neuralCube.shape[1]].flatten()
                snippetDict['blank'].append(neuralCube[sentIdx, validIndices, :])
       
    return snippetDict
import numpy as np

def addSingleLetterSnippets(snippetDict, slDat, twCubes, charDef):
    """
    Constructs the time series 'targets' used to train the lstm. The lstm is trained to output two targets:
    - A character probability vector with a one-hot encoding of the current character.
    - A binary 'new character' signal that briefly goes high at the start of each new character.
    
    This function also handles an 'ignoreError' mask to prevent the lstm from being penalized for errors at the start
    of the trial, before any character is written (in case it is labeled as a 'blank' state by the HMM).
    
    Args:
        snippetDict (dict): Dictionary containing character snippets and an estimate of where the pen tip ended.
        slDat (dict): Single letter data dictionary.
        twCubes (dict): Dictionary of matrices defining the spatiotemporal pattern of neural activity for each character,
                        indexed by the character's string representation.
        charDef (dict): Character definitions including names and lengths (from characterDefinitions.py).
        
    Returns:
        snippetDict (dict): Updated dictionary containing character snippets and pen tip end state estimates.
    """
    
    # Loop through each character and add it to the snippet library
    for charIdx, char in enumerate(charDef['charList']):
        neuralCube = slDat[f'neuralActivityCube_{char}'].astype(np.float64)

        # Get the trials that correspond to the current character
        trlIdx = [t for t in range(slDat['characterCues'].shape[0]) if slDat['characterCues'][t, 0] == char]

        # Identify the block for each trial
        blockIdx = slDat['blockNumsTimeSeries'][slDat['goPeriodOnsetTimeBin'][trlIdx]].squeeze()

        # Subtract block-specific means from each trial to remove variability due to block differences
        for block in range(slDat['blockList'].shape[0]):
            blockTrials = np.squeeze(blockIdx == slDat['blockList'][block])
            neuralCube[blockTrials, :, :] -= slDat['meansPerBlock'][np.newaxis, block, :]

        # Normalize the data by dividing by the standard deviation across all trials
        neuralCube /= slDat['stdAcrossAllData'][np.newaxis, :, :]

        # Add each example of the character to the snippet dictionary
        for trialIdx in range(neuralCube.shape[0]):
            # Determine the end of the trial based on temporal windows and character length
            endStep = np.argwhere(twCubes[char + '_T'][:, trialIdx] >= 60 + charDef['charLen'][charIdx]).astype(np.int32)
            
            if len(endStep) == 0:
                continue

            # Extract the relevant snippet for the current character and append it
            newExample = neuralCube[trialIdx, 60:endStep[0, 0], :]
            snippetDict[char].append(newExample)
            snippetDict[char + '_penEndState'].append(-2)  # Pen end state placeholder value

    return snippetDict
