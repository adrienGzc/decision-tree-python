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
      if sizeLeaf == 0: return 0
      # Count all instance depending on each classes.
      nbClassElem = self.__countAllElemInList(target)
      # Add the score for each class together.
      classScore = sum([pow(val / sizeLeaf, 2) for val in nbClassElem.values()])
      giniScore.append((1.0 - classScore) * (sizeLeaf / nbInstances))
    return sum(giniScore)

  # Return the 2 leaf containing the splitted data on the breakpoint.
  def __split(self, data, indexAttr, splitValue):
    leftLeaf = list()
    rightLeaf = list()

    # According to the subject, lower value to the left and rest at the right.
    for instance in data:
      if instance[indexAttr] <= splitValue:
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
        leafs = self.__split(dataset, indexAttr, value)
        # Calculate gini scrore for value as breakpoint.
        giniScore = self.__calculateGiniScore(leafs)
        if not tree or tree['gini'] > giniScore:
          tree = {'breakpoint': value, 'indexAttr': indexAttr, 'leafs': leafs, 'gini': giniScore}
      indexAttr += 1
    leftLeaf, rightLeaf = tree['leafs']
    tree.pop('leafs')
    tree['leftLeaf'] = leftLeaf
    tree['rightLeaf'] = rightLeaf
    return tree

  def __test(self, leafs):
    result = self.__countUniqueValue(leafs)
    return max(result, key=result.count)

  def __recursiveCreation(self, tree, depth, maxDepth):
    # Check empty data in split.
    if not tree['leftLeaf'] or not tree['rightLeaf']:
      joinLeaf = tree['leftLeaf'] + tree['rightLeaf']
      tree['leftLeaf'] = self.__test(joinLeaf)
      tree['rightLeaf'] = self.__test(joinLeaf)
      return None
    elif maxDepth is not None and maxDepth >= depth:
      # If maxDepth set then check if value is reach.
      tree['leftLeaf'] = self.__test(tree['leftLeaf'])
      tree['rightLeaf'] = self.__test(tree['rightLeaf'])
      return None
    
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

  def __displayTree(self, tree, depth):
    if isinstance(tree, dict):
      for _space in range(0, depth):
        print(' ')
      print('Attribute: %d, split < %.3f' % (tree['indexAttr'] + 1, tree['breakpoint']))
      self.__displayTree(tree['leftLeaf'], depth + 1)
      self.__displayTree(tree['rightLeaf'], depth + 1)
    else:
      for _space in range(0, depth):
        print(' ')
      print(tree)

  def show(self):
    if self.__tree is not None:
      self.__displayTree(self.__tree, 0)
    else:
      print('Error: No present tree. You need to fit the DecistionTree first.')

  def fit(self, dataset, target):
    dataset = self.__transformDataIntoList(dataset)
    dataset = self.__concateTargetWithDataset(dataset, target)
    self.__createTree(dataset)

  def predict(self):
    pass
