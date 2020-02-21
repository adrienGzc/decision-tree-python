from sklearn import datasets

class Loader:
  def __init__(self):
    self.dataLoader = {
      'iris': datasets.load_iris,
      'wine': datasets.load_wine,
      'cancer': datasets.load_breast_cancer,
    }

  def getDataset(self, name):
    if name not in self.dataLoader:
      print('Error: the only dataset available are: ', [key for key in self.dataLoader])
      return(84)
    return self.dataLoader[name]()