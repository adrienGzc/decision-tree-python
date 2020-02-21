class DecisionTree:
  def __init__(self, criterion='gini', maxDepth=None):
    criterionList = ['gini', 'entropy']

    if criterion not in criterionList:
      raise ValueError('the criterion parameter mort be one of: %s' % criterionList)
    if isinstance(maxDepth, int) == False and maxDepth is not None:
      raise ValueError('maxDepth should be an int.')
    self.criterion = criterion
    self.maxDepth = maxDepth

  # Some magic here. Add the target label to the prediction information for the ROC.
  def __appendTargetToPrediction(self, targets, predictions):
    for index in range(len(predictions)):
      tmp = list(predictions[index])
      tmp.append(targets[index])
      predictions[index] = tmp
    return predictions

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

  def fit(self, data, target):
    pass

  def predict(self, data, target):
    pass


# Gini = 