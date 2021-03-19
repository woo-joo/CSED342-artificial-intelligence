#!/usr/bin/python

import random
import collections
import math
import sys
from collections import Counter
from util import *


############################################################
# Problem 1: hinge loss
############################################################

def problem_1a():
    """
    return a dictionary that contains the following words as keys:
        pretty, good, bad, plot, not, scenery
    """
    # BEGIN_YOUR_ANSWER (our solution is 1 lines of code, but don't worry if you deviate from this)
    return {'pretty': 1, 'good': 0, 'bad': -1, 'plot': -1, 'not': -1, 'scenery': 0}
    # END_YOUR_ANSWER

############################################################
# Problem 2: binary classification
############################################################

############################################################
# Problem 2a: feature extraction

def extractWordFeatures(x):
    """
    Extract word features for a string x. Words are delimited by
    whitespace characters only.
    @param string x: 
    @return dict: feature vector representation of x.
    Example: "I am what I am" --> {'I': 2, 'am': 2, 'what': 1}
    """
    # BEGIN_YOUR_ANSWER (our solution is 6 lines of code, but don't worry if you deviate from this)
    return dict(Counter(x.split()))
    # END_YOUR_ANSWER

############################################################
# Problem 2b: stochastic gradient descent

def learnPredictor(trainExamples, testExamples, featureExtractor, numIters, eta):
    '''
    Given |trainExamples| and |testExamples| (each one is a list of (x,y)
    pairs), a |featureExtractor| to apply to x, and the number of iterations to
    train |numIters|, the step size |eta|, return the weight vector (sparse
    feature vector) learned.

    You should implement stochastic gradient descent.

    Note:
    1. only use the trainExamples for training!
    You can call evaluatePredictor() on both trainExamples and testExamples
    to see how you're doing as you learn after each iteration.
    2. don't shuffle trainExamples and use them in the original order to update weights.
    3. don't use any mini-batch whose size is more than 1
    '''
    weights = {}  # feature => weight

    def sigmoid(n):
        return 1 / (1 + math.exp(-n))

    # BEGIN_YOUR_ANSWER (our solution is 14 lines of code, but don't worry if you deviate from this)
    for _ in range(numIters):
        for x, y in trainExamples:
            phi = featureExtractor(x)
            sigma = sigmoid(dotProduct(weights, phi))
            prob = sigma if y == 1 else (1 - sigma)
            scale = (eta * y * sigma * (1 - sigma)) / prob
            increment(weights, scale, phi)
    # END_YOUR_ANSWER
    return weights

############################################################
# Problem 2c: bigram features

def extractBigramFeatures(x):
    """
    Extract unigram and bigram features for a string x, where bigram feature is a tuple of two consecutive words. In addition, you should consider special words '<s>' and '</s>' which represent the start and the end of sentence respectively. You can exploit extractWordFeatures to extract unigram features.

    For example:
    >>> extractBigramFeatures("I am what I am")
    {('am', 'what'): 1, 'what': 1, ('I', 'am'): 2, 'I': 2, ('what', 'I'): 1, 'am': 2, ('<s>', 'I'): 1, ('am', '</s>'): 1}
    """
    # BEGIN_YOUR_ANSWER (our solution is 5 lines of code, but don't worry if you deviate from this)
    words = x.split()
    pairs = [('<s>', words[0])] + list(zip(words[:-1], words[1:])) + [(words[-1], '</s>')]
    phi = dict(Counter(pairs))
    phi.update(extractWordFeatures(x))
    # END_YOUR_ANSWER
    return phi

############################################################
# Problem 3a: k-means exercise
############################################################

def problem_3a_1():
    """
    Return two centers which are 2-dimensional vectors whose keys are 'mu0' and 'mu1'.
    Assume the initial centers are
    ({'mu0': 1, 'mu1': -1}, {'mu0': 1, 'mu1': 2})
    """
    # BEGIN_YOUR_ANSWER (our solution is 1 lines of code, but don't worry if you deviate from this)
    return ({'mu0': 1, 'mu1': 0}, {'mu0': 1, 'mu1': 1.5})
    # END_YOUR_ANSWER

def problem_3a_2():
    """
    Return two centers which are 2-dimensional vectors whose keys are 'mu0' and 'mu1'.
    Assume the initial centers are
    ({'mu0': -2, 'mu1': 0}, {'mu0': 3, 'mu1': -1})
    """
    # BEGIN_YOUR_ANSWER (our solution is 1 lines of code, but don't worry if you deviate from this)
    return ({'mu0': 0, 'mu1': 0.5}, {'mu0': 2, 'mu1': 1})
    # END_YOUR_ANSWER

############################################################
# Problem 3b: k-means implementation
############################################################

def kmeans(examples, K, maxIters):
    '''
    examples: list of examples, each example is a string-to-double dict representing a sparse vector.
    K: number of desired clusters. Assume that 0 < K <= |examples|.
    maxIters: maximum number of iterations to run for (you should terminate early if the algorithm converges).
    Return: (length K list of cluster centroids,
            list of assignments, (i.e. if examples[i] belongs to centers[j], then assignments[i] = j)
            final reconstruction loss)
    '''
    # BEGIN_YOUR_ANSWER (our solution is 40 lines of code, but don't worry if you deviate from this)
    examples_squared = [dotProduct(example, example) for example in examples]
    def distance(idx_c, idx_e):
        return centroids_squared[idx_c] - 2 * dotProduct(centroids[idx_c], examples[idx_e]) + examples_squared[idx_e]

    # initialize centroids
    centroids = random.sample(examples, K)
    centroids_squared = [dotProduct(centroid, centroid) for centroid in centroids]
    
    assignments = []
    for _ in range(maxIters):
        # assign each example to best cluster
        _assignments = []
        for i in range(len(examples)):
            distances = {j: distance(j, i) for j in range(K)}
            assignment = min(distances, key=distances.get)
            _assignments.append(assignment)
        if _assignments == assignments:
            break
        assignments = _assignments

        # update centroids
        means = [[{}, 0] for _ in range(K)]
        for i in range(len(examples)):
            increment(means[assignments[i]][0], 1, examples[i])
            means[assignments[i]][1] += 1
        for i, (mean, size) in enumerate(means):
            if size > 0:
                for k, v in list(mean.items()):
                    mean[k] = v / size
            centroids[i] = mean
            centroids_squared[i] = dotProduct(mean, mean)
    
    loss = sum(distance(assignments[i], i) for i in range(len(examples)))
    return (centroids, assignments, loss)
    # END_YOUR_ANSWER

