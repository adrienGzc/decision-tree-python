import random
import copy

class CrossValidator:
  def __init__(self, algo=None, dataset=None, nbFolds=10):
    random.seed(1)
    self.folds = list()
    self.algorithm = algo
    self.dataset = self.__transformDataIntoList(dataset)
    self.nbFolds = nbFolds
    self.rocData = list()

  def __transformDataIntoList(self, data):
    if data is not None:
      return [list(instance) for instance in data]
    return None

  # Method to check if the CrossValidator class as everything needed to start.
  def __checkNotEmptyAttributes(self):
    if (self.algorithm is None or self.dataset is None or self.nbFolds <= 1):
      print("Error: Algorithm and dataset shouldn't be empty and nbFolds neither less nor equal to 0")
      return False
    return True

  # nbInstances as to be lower than nbFolds, I round the return to get a integer and not a float.
  def __getFoldSize(self, nbInstances, nbFolds):
    return round(nbInstances / nbFolds)

  # Fill the folds Class variable with all folds of instances shuffled.
  def __splitDatasetIntoKFolds(self):
    copyDataset = self.dataset.copy()
    # Desorganize the dataset
    random.shuffle(copyDataset)
    # Get the number of instances in each fold.
    foldSize = self.__getFoldSize(len(self.dataset), self.nbFolds)

    # I move the pointer start and end to cur the dataset into the number of instances calculated.
    for nb in range(self.nbFolds):
      start = foldSize * nb
      end = foldSize + start
      self.folds.append(copyDataset[start:end])

  # Return the taget label from the dataset, target as to be at the end.
  def __extractTargetFromDataset(self, data):
    newDataset = list()
    target = list()
    for instance in data:
      target.append(instance[len(instance) - 1])
      newDataset.append(list(instance[:-1][0]))
    return newDataset, target

  # Count the correct answer and return the accuracy of them, scaled on 0 to 100%.
  def __getAccuracy(self, original, predictions):
    nbCorrectPredictions = 0

    # Loop through all the predictions.
    for index in range(len(predictions)):
      # Get the class predicted.
      predictClass = predictions[index][0]

      # If she correspond to the target label then add a correct answer.
      if (original[index] == predictClass):
        nbCorrectPredictions += 1

    return nbCorrectPredictions / len(original) * 100

  # Return a simple list of instances as a deep copy and delete the testing fold.
  def __getTrainData(self, dataToSquash, indexToRemove):
    data = copy.deepcopy(self.folds)
    data.pop(indexToRemove)
    return sum(data, [])

  # Some magic here. Add the target label to the prediction information for the ROC.
  def __appendTargetToPrediction(self, targets, predictions):
    for index in range(len(predictions)):
      tmp = list(predictions[index])
      tmp.append(targets[index])
      predictions[index] = tmp
    return predictions

  # Calculate the score of the accuracy:
  #   - all the accuracy as a list, len(list accuracy) = nbFolds.
  #   - the mean accuracy based on all the accuracy.
  #   - a list with -> Prediction, Classes probabilities, Real target expected.
  def score(self):
    if (self.__checkNotEmptyAttributes() is False):
      return False

    # Split the data into K folds.
    self.__splitDatasetIntoKFolds()
    accuracyScores = list()
    for index, fold in enumerate(self.folds):
      trainData = self.__getTrainData(self.folds, index)
      # Extract the label from the dataset.
      X_train, targetTrain = self.__extractTargetFromDataset(trainData)
      testData = copy.deepcopy(fold)
      X_test, targetTest = self.__extractTargetFromDataset(testData)

      # Train the Naive Bayes algorithm.
      self.algorithm.fit(X_train, targetTrain)
      # Predict with the fold.
      acc, predictionFold = self.algorithm.predict(X_test, targetTest)
      # Add to the data for the ROC the prediction information with the target label.
      self.rocData.extend(self.__appendTargetToPrediction(targetTest, predictionFold))
      # Add the accuracy calculate.
      accuracyScores.append(acc)
    return accuracyScores, sum(accuracyScores) / len(accuracyScores), self.rocData