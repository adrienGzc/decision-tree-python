class Tree:
  def __init__(self, data=list()):
    self.__left = None
    self.__right = None
    self.__data = data
    self.__info = None

  def setLeaf(self, data, branch):
    if branch == 'left':
      self.__left = Tree(data)
    elif branch == 'right':
      self.__right = Tree()

  def getLeftLeaf(self):
    return self.__left
  
  def getRightLeaf(self):
    return self.__right

  def setData(self, data):
    self.__data = data.copy

  def getData(self):
    return self.__data
  
  def setInfo(self, newInfo):
    self.__info = dict(newInfo)

  def getInfo(self):
    return self.__info