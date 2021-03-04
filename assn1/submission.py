import collections


############################################################
# Problem 1

def problem_1a():
    """
    return a number between 0 and 1
    """
    # BEGIN_YOUR_ANSWER (our solution is 1 lines of code, but don't worry if you deviate from this)
    return 0.4
    # END_YOUR_ANSWER

def problem_1b():
    """
    return one of [1, 2, 3, 4]
    """
    # BEGIN_YOUR_ANSWER (our solution is 1 lines of code, but don't worry if you deviate from this)
    return 4
    # END_YOUR_ANSWER

############################################################
# Problem 2a

def getWordKey(word):
    """
    Design a key function that computeMaxWordLength exploits to choose a word.
    Before implementing it, you may need to examine how comparision operators (<, <=, >, >=, ==)
    between tuples and between strings compute outputs.
    """
    # BEGIN_YOUR_ANSWER (our solution is 1 lines of code, but don't worry if you deviate from this)
    return (len(word), word)
    # END_YOUR_ANSWER

def computeMaxWordLength(text):
    """
    Given a string |text|, return the longest word in |text|.  If there are
    ties, choose the word that comes latest in the alphabet.

    Note:
    - max function returns the maximum item with respect to the key argument.
    - You should not modify this function.
    """
    return max(text.split(), key=getWordKey)  # usually key argument is a function defined by 'def' or 'lambda'

############################################################
# Problem 2b

def manhattanDistance(loc1, loc2):
    """
    Return the generalized manhattan distance between two locations,
    where the locations are tuples of numbers.
    The distance is the sum of differences of all corresponding elements between two tuples.

    For exapmle:
    >>> loc1 = (2, 4, 5)
    >>> loc2 = (-1, 3, 6)
    >>> manhattanDistance(loc1, loc2)  # 5

    You can exploit sum, abs, zip functions and a generator to implement it as one line code!
    """
    # BEGIN_YOUR_ANSWER (our solution is 1 lines of code, but don't worry if you deviate from this)
    return sum(abs(comp1 - comp2) for comp1, comp2 in zip(loc1, loc2))
    # END_YOUR_ANSWER

############################################################
# Problem 2c

def countMutatedSentences(sentence):
    """
    Given a sentence (sequence of words), return the number of all possible
    mutated sentences of the same length, where each pair of adjacent words
    in the mutated sentences also occurs in the original sentence.

    For example:
    >>> countMutatedSentences('the cat and the mouse')  # 4

    where 4 possible mutated sentences exist:
    - 'and the cat and the'
    - 'the cat and the mouse'
    - 'the cat and the cat'
    - 'cat and the cat and'

    which consist of the following adjacent word pairs:
    - the cat
    - cat and
    - and the
    - the mouse

    Notes:
    - You don't need to generate actual mutated sentences.
    - You should apply dynamic programming for efficiency.
    """
    # BEGIN_YOUR_ANSWER (our solution is 17 lines of code, but don't worry if you deviate from this)
    words = sentence.split()

    suffixes, prefixes = [], collections.defaultdict(set)
    for word in words:
        if not word in suffixes:
            suffixes.append(word)
            for i in range(len(words) - 1):
                if words[i + 1] == word:
                    prefixes[word].add(words[i])

    counts = [collections.defaultdict(int) for i in range(len(words) - 1)]
    for suffix in suffixes:
        counts[0][suffix] = len(prefixes[suffix])
    for i in range(1, len(words) - 1):
        for suffix in suffixes:
            for prefix in prefixes[suffix]:
                counts[i][suffix] += counts[i - 1][prefix]
    
    return sum(counts[-1].values())
    # END_YOUR_ANSWER

############################################################
# Problem 2d

def sparseVectorDotProduct(v1, v2):
    """
    Given two sparse vectors |v1| and |v2|, each represented as collection.defaultdict(float), return
    return their dot product.
    You might find it useful to use sum() and a list comprehension.
    This function will be useful later for linear classifiers.
    """
    # BEGIN_YOUR_ANSWER (our solution is 3 lines of code, but don't worry if you deviate from this)
    return sum(v1[k] * v2[k] for k in v1)
    # END_YOUR_ANSWER

############################################################
# Problem 2e

def incrementSparseVector(v1, scale, v2):
    """
    Given two sparse vectors |v1| and |v2|, perform v1 += scale * v2.
    This function will be useful later for linear classifiers.
    """
    # BEGIN_YOUR_ANSWER (our solution is 2 lines of code, but don't worry if you deviate from this)
    for k in v2:
        v1[k] += scale * v2[k]
    # END_YOUR_ANSWER

############################################################
# Problem 2f

def computeMostFrequentWord(text):
    """
    Splits the string |text| by whitespace and returns two things as a pair: 
    the set of words that occur the maximum number of times, and their count
    i.e. (set of words that occur the most number of times, that maximum number/count)
    You might find it useful to use collections.defaultdict(int).
    """
    # BEGIN_YOUR_ANSWER (our solution is 6 lines of code, but don't worry if you deviate from this)
    raise NotImplementedError  # remove this line before writing code
    # END_YOUR_ANSWER
