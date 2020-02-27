import numpy as np
from math import log2
import pprint

class DecisionTree:
  def __init__(self):
    self.priorEntropy = None

  def __transformDataIntoList(self, data):
    return [list(instance) for instance in data]

  def __extractTargetFromDataset(self, data):
    newDataset = list()
    target = list()
    for instance in data:
      target.append(instance[len(instance) - 1])
      newDataset.append(instance[:-1])
    return newDataset, target

  # Add the target label at the end of the dataset. Needed to shuffle the data easily.
  def __concateTargetWithDataset(self, dataset, targetDataset):
    data = list()
    for index, instance in enumerate(dataset):
      tmp = list()
      if type(instance) is not list:
        tmp.append(instance)
      else:
        tmp = list(instance)
      tmp.append(targetDataset[index])
      data.append(tmp)
    return data

  def __countUniqueValue(self, data):
    return list(set(data))

  # Divide data by class.
  def __classSpliter(self, data, target):
    splitedClasses = dict()

    for index, value in enumerate(data):
      # Create new key in dict if class not already created.
      if target[index] not in splitedClasses:
        splitedClasses[target[index]] = list()
      # Add the instance to the corresponding class.
      splitedClasses[target[index]].append(value)
    return splitedClasses  

  def __getEntropy(self, dataset, clas):
    nbInstances = len(dataset)
    nbOccurence = dataset.count(clas)
    ratio = nbOccurence / nbInstances
    return (-ratio * log2(ratio))

  def __createTree(self, data, target, classesInfo):
    nbInstances = len(data)
    gains = list()

    # Get all value in one attribute (get each column of the dataset).
    for attribute in zip(*data):
      classeSplit = self.__classSpliter(self.__concateTargetWithDataset(attribute, target), attribute)
      entropys = list()
      for _key, split in classeSplit.items():
        size = len(split)
        subSplit, label = self.__extractTargetFromDataset(split)
        classDetails = self.__classSpliter(subSplit, label)
        entropy = self.__getPriorEntropy(subSplit, label, classDetails)
        entropys.append(size / nbInstances * entropy)
      gains.append(self.priorEntropy - sum(entropys))
    print(gains)
    print(gains.index(max(gains)))

  def __getPriorEntropy(self, data, target, classesInfo):
    nbInstances = len(data)
    entropys = list()

    for _key, value in classesInfo.items():
      size = len(value)
      entropy = -(size / nbInstances) * log2(size / nbInstances)
      entropys.append(entropy)
    return sum(entropys)

  def fit(self, data, target):
    data = self.__transformDataIntoList(data)
    classesInfo = self.__classSpliter(data, target)
    self.priorEntropy = self.__getPriorEntropy(data, target, classesInfo)
    self.__createTree(data, target, classesInfo)

  def predict(self, data, target):
    pass