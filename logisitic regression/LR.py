from __future__ import print_function
from sklearn.linear_model import LogisticRegression

import numpy as np
import matplotlib.pyplot as plt

class LogisticRegression1(object):
  def __init__(self, input_size, reg, std=1e-4):
    """
    Initializing the weight vector
    
    Input:
    - input_size: the number of features in dataset, for MNIST this is 784
    - reg: the l2 regularization weight
    - std: the variance of initialized weights
    """
    self.W = std * np.random.randn(input_size)
    self.reg = reg
    
  def sigmoid(self,x):

    z = np.zeros_like(x,dtype=float)
    z[(x >= 0)] = np.exp(-x[(x >= 0)])
    z[(x < 0)] = np.exp(x[(x < 0)])
    y = np.ones_like(x,dtype=float)
    y[(x < 0)] = z[(x < 0)]
    return y / (1 + z)

  def loss(self, X, y):

    N, D = X.shape
    reg = self.reg
  
    scores = self.sigmoid(np.dot(X, self.W))

    reg_term = reg*np.sum(np.square(self.W))
    loss_term = -(np.sum(y*np.log(scores)+(1-y)*np.log(1-scores)))/N
    loss = loss_term + reg_term/(N)

    z = scores-y
    dLdW = ((np.dot(z.T, X)) + 2*reg*self.W)/N
    
    return loss, dLdW

  def gradDescent(self,X, y, learning_rate, num_epochs, val_X, val_Y, test_X, test_Y):

    N, D = X.shape
    acc_hist = np.zeros(num_epochs)
    train_acc_hist = np.zeros(num_epochs)
    test_acc_hist = np.zeros(num_epochs)
    for i in range(num_epochs):
      loss, dLdW = self.loss(X, y)
      self.W = self.W - learning_rate * dLdW
      #save the accuracy for each epoch as we are training. Save large amounts of time
      train_labels = self.predict(X)
      trainaccuracy = ((train_labels == y)).mean()
      val_labels = self.predict(val_X)
      valaccuracy = ((val_labels == val_Y)).mean()
      test_labels = self.predict(test_X)
      testaccuracy = ((test_labels == test_Y)).mean()
      train_acc_hist[i] = trainaccuracy
      acc_hist[i] = valaccuracy
      test_acc_hist[i] = testaccuracy
      
    return train_acc_hist, acc_hist, test_acc_hist

  def predict(self, X):

    N = X.shape[0]
    y_pred = np.zeros((N,))
    A = self.sigmoid(np.dot(X, self.W))
    for i in range(N-1):
      if A[i, ] > 0.5:
        y_pred[i,] = 1
      else:
        y_pred[i,] = 0
    
    return y_pred

def plot_accuracy(train_X, train_Y, val_X, val_Y, test_X, test_Y):
  #fixed other, plot learning rate
  reg_list = np.linspace(0.0, 0.2, num=5)
  lr_list = [1e-4, 1e-3, 1e-2, 1e-1, 1, 10]  
  train_acc = []; val_acc = []; test_acc = []
  
  for lr in lr_list:
    LR = LogisticRegression1(784, 0.1)
    LR.gradDescent(train_X, train_Y, lr, 500, val_X, val_Y, test_X, test_Y)
    train_labels = LR.predict(train_X)
    trainaccuracy = ((train_labels == train_Y)).mean()
    train_acc.append(trainaccuracy)
    val_labels = LR.predict(val_X)
    valaccuracy = ((val_labels == val_Y)).mean()
    val_acc.append(valaccuracy)
    test_labels = LR.predict(test_X)
    testaccuracy = ((test_labels == test_Y)).mean()
    test_acc.append(testaccuracy)

  plt.plot(np.log(lr_list), train_acc, color='green', label='train accuracy')
  plt.plot(np.log(lr_list), val_acc, color='blue', label='validation accuracy')
  plt.plot(np.log(lr_list), test_acc, color='red', label='train accuracy')
  plt.legend()
  plt.savefig('learning rate vs accuracy.png')
  plt.clf()
  
  
  #fixed other, plot regularization count
  train_acc = []; val_acc = []; test_acc = []
  for reg in reg_list:
    LR = LogisticRegression1(784, reg)
    LR.gradDescent(train_X, train_Y, 1, 500, val_X, val_Y, test_X, test_Y)
    train_labels = LR.predict(train_X)
    trainaccuracy = ((train_labels == train_Y)).mean()
    train_acc.append(trainaccuracy)
    val_labels = LR.predict(val_X)
    valaccuracy = ((val_labels == val_Y)).mean()
    val_acc.append(valaccuracy)
    test_labels = LR.predict(test_X)
    testaccuracy = ((test_labels == test_Y)).mean()
    test_acc.append(testaccuracy)

  plt.plot(reg_list, train_acc, color='green',label='train accuracy')
  plt.plot(reg_list, val_acc, color='blue',label='validation accuracy')
  plt.plot(reg_list, test_acc, color='red', label='test accuracy')
  plt.legend()
  plt.savefig('regularization vs accuracy.png')
  plt.clf()
  
def plot_accuracy_iter(train_acc, val_acc, test_acc):
  #fixed other, plot iteration numbers
  num_iter = np.arange(10,501)

  train_acc = train_acc[9:]
  val_acc = val_acc[9:]
  test_acc = test_acc[9:]

  plt.plot(num_iter, train_acc, color='green',label='train accuracy')
  plt.plot(num_iter, val_acc, color='blue',label='validation accuracy')
  plt.plot(num_iter, test_acc, color='red', label='test accuracy')
  plt.legend()
  plt.savefig('num of iterations vs accuracy.png')
  plt.clf()
  

def main():
    # Load training data
    train_X = np.load('../Data/X_train.npy')
    #print(train_X.shape)
    train_Y = np.load('../Data/y_train.npy')
    
    #Binarizing the labels, odd numbers will have label 1 and even ones will have 0
    train_Y = (train_Y%2)

    # Load test data
    test_X = np.load('../Data/X_test.npy')
    test_Y = np.load('../Data/y_test.npy')
    
    test_Y = (test_Y%2)
    
    # Load val data
    val_X = np.load('../Data/X_val.npy')
    val_Y = np.load('../Data/y_val.npy')
    
    val_Y = (val_Y%2)
    
    LR = LogisticRegression1(784, 0.2)
    train_acc, val_acc, test_acc = LR.gradDescent(train_X, train_Y, 1, 500, val_X, val_Y, test_X, test_Y)
    plot_accuracy(train_X, train_Y, val_X, val_Y, test_X, test_Y)
    #plot accuracy for num of iterations
    plot_accuracy_iter(train_acc, val_acc, test_acc)
    #model selection
    
    reg_list = np.linspace(0.0, 0.2, num=5)
    lr_list = [1e-4, 1e-3, 1e-2, 1e-1, 1, 10]
    result = []
    for lr in lr_list:
      for reg in reg_list:
        LR = LogisticRegression1(784, reg)
        acc = LR.gradDescent(train_X, train_Y, lr, 500, val_X, val_Y, test_X, test_Y)
        result.append(acc[1])
    result = np.array(result)
    print('max accuracy:', np.amax(result), 'index:', np.argmax(result))
    
    train_labels = LR.predict(train_X)
    trainaccuracy = ((train_labels == train_Y)).mean()
    
    val_labels = LR.predict(val_X)
    valaccuracy = ((val_labels == val_Y)).mean()
    
    test_labels = LR.predict(test_X)
    testaccuracy = ((test_labels == test_Y)).mean()
    print("Train accuracy : ", trainaccuracy, "Test accuracy : ", testaccuracy, "Val accuracy : ", valaccuracy)
    
    
if __name__ == '__main__':
    main()



