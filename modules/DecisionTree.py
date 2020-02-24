import numpy as np
from math import log2
import scipy.stats
import pprint

class DecisionTree:
  def __init__(self, criterion='entropy', maxDepth=None):
    criterionList = ['gini', 'entropy']

    if criterion not in criterionList:
      raise ValueError('the criterion parameter mort be one of: %s' % criterionList)
    if isinstance(maxDepth, int) == False and maxDepth is not None:
      raise ValueError('maxDepth should be an int.')
    self.criterion = criterion
    self.maxDepth = maxDepth

  def __transformDataIntoList(self, data):
    return [list(instance) for instance in data]

  # Add the target label at the end of the dataset. Needed to shuffle the data easily.
  def __concateTargetWithDataset(self, dataset, targetDataset):
    data = list()
    for index, instance in enumerate(dataset):
      tmp = list(instance)
      tmp.append(targetDataset[index])
      data.append(tmp)
    return data

  # Divide data by class.
  def __classSpliter(self, data, target):
    splitedClasses = dict()

    for index, target in enumerate(data):
      # Create new key in dict if class not already created.
      if target[index] not in splitedClasses:
        splitedClasses[target[index]] = list()
      # Add the instance to the corresponding class.
      splitedClasses[target[index]].append(target)
    return splitedClasses

  def __countUniqueProduct(self, data):
    return list(set(data))

  def __getEntropy(self, dataset, classes, clas):
    nbInstances = len(dataset)
    nbOccurence = dataset.count(clas)
    ratio = nbOccurence / nbInstances
    return (-ratio * log2(ratio))

  def __getGain(self, dataset, uniqueValues, priorEntropy):
    for value in uniqueValues:
k      pass

  def __getEntropySubsets(self, dataset, uniqueValues, currentValue):
    # subsetEntropy = self.__getEntropySubsets(attribute, uniqueValInAttr, value)
    return list()

  def __getPriorEntropy(self, attribute, uniqueValInAttr):
    entropys = list()

    for value in uniqueValInAttr:
      entropy = self.__getEntropy(attribute, uniqueValInAttr, value)
      entropys.append(entropy)
    return sum(entropys)

  def __createTree(self, data, attributeUsed=[]):
    entropyAttributes = list()

    for attribute in zip(*data):
      uniqueValue = self.__countUniqueProduct(list(attribute))
      priorEntropy = self.__getPriorEntropy(attribute, uniqueValue)
      gain = self.__getGain(attribute, uniqueValue, priorEntropy)
      entropyAttributes.append(gain)
    return entropyAttributes

  def fit(self, data, target):
    data = self.__transformDataIntoList(data)
    tree = self.__createTree(data)

  def predict(self, data, target):
    pass