from math import pow
from modules.Tree import Tree

class DecisionTreeGini:
  def __init__(self, maxDepth=None):
    self.__tree = None
    self.__maxDepth = maxDepth
  
  def __transformDataIntoList(self, data):
    return [list(instance) for instance in data]

  # Return the taget label from the dataset, target as to be at the end.
  def __extractTargetFromDataset(self, data):
    newDataset = list()
    target = list()
    for instance in data:
      target.append(instance[len(instance) - 1])
      newDataset.append(instance[:-1])
    return newDataset, target

  # Add target label to the dataset (add 1 column).
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

  # Return a dict with all the classes as key and the nb of each class as value.
  def __countAllElemInList(self, listElem):
    nbAllElem = dict()

    for elem in listElem:
      if elem not in nbAllElem:
        nbAllElem[elem] = 1
      else:
        nbAllElem[elem] += 1
    return nbAllElem

  def __calculateGiniScore(self, leafs):
    # Get size of all instance in each leaf.
    nbInstances = sum([len(leaf) for leaf in leafs])
    giniScore = list()

    for leaf in leafs:
      data, target = self.__extractTargetFromDataset(leaf)
      # Get size of the leaf and if 0 then return 0 since there is no data.
      sizeLeaf = len(data)
      if sizeLeaf == 0: continue
      # Count all instance depending on each classes.
      nbClassElem = self.__countAllElemInList(target)
      # Add the score for each class together.
      classScore = sum([pow((val / sizeLeaf), 2) for val in nbClassElem.values()])
      giniScore.append((1.0 - classScore) * (sizeLeaf / nbInstances))
    return sum(giniScore)

  # Return the 2 leaf containing the splitted data on the breakpoint.
  def __split(self, data, indexAttr, splitValue):
    leftLeaf = list()
    rightLeaf = list()

    # According to the subject, lower value to the left and rest at the right.
    for instance in data:
      if instance[indexAttr] < splitValue:
        leftLeaf.append(instance)
      else:
        rightLeaf.append(instance)
    return leftLeaf, rightLeaf

  # Return a dict for the best split node found.
  def __foundBestSplit(self, dataset):
    # Detached the target label from the dataset.
    data, _target = self.__extractTargetFromDataset(dataset)
    tree = dict()
    indexAttr = 0

    # Loop through each attribute, zip return all the column at once.
    for attribute in zip(*data):
      for value in attribute:
        # Get the two leaf for split (left and right leafs).
        leftLeaf, rightLeaf = self.__split(dataset, indexAttr, value)
        # Calculate gini scrore for value as breakpoint.
        giniScore = self.__calculateGiniScore((leftLeaf, rightLeaf, ))
        if not tree or tree['gini'] > giniScore:
          tree = {'breakpoint': value, 'indexAttr': indexAttr, 'leftLeaf': leftLeaf, 'rightLeaf': rightLeaf, 'gini': giniScore}
      indexAttr += 1
    return tree

  def __getResult(self, leafs):
    # Extract the target from the leafs to get the result.
    # If multiple target then take the highest one.
    _data, target = self.__extractTargetFromDataset(leafs)
    return max(self.__countUniqueValue(target))

  def __recursiveCreation(self, tree, depth, maxDepth):
    # Check empty data in split.
    if not tree['leftLeaf'] or not tree['rightLeaf']:
      joinLeaf = tree['leftLeaf'] + tree['rightLeaf']
      tree['leftLeaf'] = self.__getResult(joinLeaf)
      tree['rightLeaf'] = self.__getResult(joinLeaf)
      return
    elif maxDepth is not None and maxDepth >= depth:
      # If maxDepth set then check if value is reach.
      tree['leftLeaf'] = self.__getResult(tree['leftLeaf'])
      tree['rightLeaf'] = self.__getResult(tree['rightLeaf'])
      return
    
    # Split left
    tree['leftLeaf'] = self.__foundBestSplit(tree['leftLeaf'])
    self.__recursiveCreation(tree['leftLeaf'], depth + 1, maxDepth)

    # Split right
    tree['rightLeaf'] = self.__foundBestSplit(tree['rightLeaf'])
    self.__recursiveCreation(tree['rightLeaf'], depth + 1, maxDepth)


  def __createTree(self, dataset):
    # Create root node of the tree.
    self.__tree = self.__foundBestSplit(dataset)
    # Create rest of the tree.
    self.__recursiveCreation(self.__tree, 0, self.__maxDepth)

  # Simple recursion function to display the tree trained.
  def __displayTree(self, tree, depth, label='root'):
    sentence = ''
    if isinstance(tree, dict):
      for _space in range(0, depth):
        sentence += ' '
      sentence += '%s -> X%d, value < %.3f, gini: %.3f' % (label, tree['indexAttr'] + 1, tree['breakpoint'], tree['gini'])
      print(sentence)
      self.__displayTree(tree['leftLeaf'], depth + 1, 'left')
      self.__displayTree(tree['rightLeaf'], depth + 1, 'right')
    else:
      for _space in range(0, depth):
        sentence += ' '
      print(sentence, tree)

  # Function to call to display the training tree result.
  def show(self):
    if self.__tree is not None:
      self.__displayTree(self.__tree, 0)
    else:
      print('Error: No present tree. You need to fit the DecistionTree first.')

  # Function to train and create a decision tree.
  def fit(self, dataset, target):
    # Get the root of the tree at first.
    dataset = self.__transformDataIntoList(dataset)
    # Start creating the leaf of the tree with the split.
    dataset = self.__concateTargetWithDataset(dataset, target)
    self.__createTree(dataset)

  # Make recursive prediction through the all tree.
  def __makePrediction(self, instance, tree):
    # If attribute of the instance is lower than the breakpoint found in the training
    # then go to the right of the tree.
    if instance[tree['indexAttr']] > tree['breakpoint']:
      # Check if the right is a leaf or a endpoint.
      if isinstance(tree['rightLeaf'], dict):
        return self.__makePrediction(instance, tree['rightLeaf'])
      # If not a real leaf then return the result branch
      return tree['rightLeaf']
    else:
      # Doing exactly the same but for the left branch of the tree.
      if isinstance(tree['leftLeaf'], dict):
        return self.__makePrediction(instance, tree['leftLeaf'])
      # If not a real leaf then return the result branch
      return tree['leftLeaf']

  # Make prediction on an instance or a list.
  # Return a list of Tuple as (TARGET, PREDICTION).
  # If no target provide then return list of predictions.
  def predict(self, dataset, target=None):
    if self.__tree is None:
      print('Error: You need to fit the decision tree first with fit(dataset, target).')
      return 84
    
    predictions = list()
    # Check for one row only prediction.
    if isinstance(dataset, list):
      self.__makePrediction(dataset, self.__tree)
    else:
      # Otherwise iterate through the all dataset and make a prediction for each instance.
      for index, instance in enumerate(dataset):
        result = self.__makePrediction(instance, self.__tree)
        if target is None:
          predictions.append(result)
        else:
          predictions.append((target[index], result, ))
      return predictions