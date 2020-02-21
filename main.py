from sklearn import tree
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

from modules.Loader import Loader
from modules.DecisionTree import DecisionTree


def mainTest():
  loader = Loader()
  irisData = loader.getDataset('iris')
  
  X_train, X_test, y_train, y_test = train_test_split(irisData['data'], irisData['target'], test_size=0.20, random_state=42)


  clf = tree.DecisionTreeClassifier()
  clf = clf.fit(irisData['data'], irisData['target'])
  tree.plot_tree(clf)
  plt.show()


def main():
  loader = Loader()
  irisData = loader.getDataset('iris')
  decisionTree = DecisionTree()
  # X_train, X_test, y_train, y_test = train_test_split(irisData['data'], irisData['target'], test_size=0.50, random_state=42)

if __name__ == "__main__":
    main()
    # mainTest()