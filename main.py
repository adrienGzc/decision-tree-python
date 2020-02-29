from sklearn import tree
from sklearn.model_selection import train_test_split

from modules.Loader import Loader
from modules.CrossValidator import CrossValidator
from modules.DecisionTree import DecisionTree

loader = Loader()

# Add target label to the dataset (add 1 column).
def concateTargetWithDataset(dataset, targetDataset):
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

def mainBreastCancer():
  cancerData = loader.getDataset('cancer')
  cancerData = concateTargetWithDataset(cancerData['data'], cancerData['target'])
  decisionTree = DecisionTree()
  crossValidator = CrossValidator(algo=decisionTree, dataset=cancerData, nbFolds=10)
  _scoresByFold, meanAccuracy, _rocData = crossValidator.score()
  print('Dataset: Breast cancer\nAccuracy: %.2f%%\n' % meanAccuracy)

def mainSklearn():
  irisData = loader.getDataset('iris')
  X_train, X_test, y_train, y_test = train_test_split(irisData['data'], irisData['target'], test_size=0.80, random_state=42)
  clf = tree.DecisionTreeClassifier()
  clf = clf.fit(X_train, y_train)
  print('SKLEARN\nDataset: Iris\nAccuracy: %.2f%%\n' % clf.score(X_test, y_test))

def crossValTest():
  irisData = loader.getDataset('iris')
  irisData = concateTargetWithDataset(irisData['data'], irisData['target'])
  decisionTree = DecisionTree()
  crossValidator = CrossValidator(algo=decisionTree, dataset=irisData, nbFolds=10)
  _scoresByFold, meanAccuracy, _rocData = crossValidator.score()
  print('Dataset: Iris\nAccuracy: %.2f%%\n' % meanAccuracy)

def main():
  wineData = loader.getDataset('wine')
  X_train, X_test, y_train, y_test = train_test_split(wineData['data'], wineData['target'], test_size=0.80, random_state=42)
  decisionTree = DecisionTree()
  decisionTree.fit(X_train, y_train)
  acc, _predictions = decisionTree.predict(X_test, y_test)
  print('Dataset: Wine\nAccurary: %.2f%%\n' % acc)

if __name__ == "__main__":
    main()
    crossValTest()
    mainBreastCancer()
    mainSklearn()