import sys
import random

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import numba as nb
from scipy.ndimage.filters import gaussian_filter1d
from sklearn.preprocessing import OneHotEncoder


START = 'S'
END = 'E'
NAW = 'N'

class Vectorizer:
  def fit(self, text):
    words = [START + w + END for w in text.split()]
    max_wordlen = max(len(x) for x in words)
    words = [w + ''.join([NAW]*(max_wordlen - len(w))) for w in words]
    mapping = {NAW: 0}

    for word in words:
      for l in word:
        if l not in mapping:
          mapping[l] = len(mapping)
    
    inverse_mapping = {v: k for k, v in mapping.items()}

    self.max_wordlen = max_wordlen
    self.mapping = mapping
    self.inverse_mapping = inverse_mapping
    self.alphabet = list(set(text.replace(' ', '')))

  def transform(self, text):

    data, labels = self._transform(text)
    x_train = np.zeros(shape=data.shape + (len(self.mapping), ))
    y_train = np.zeros_like(x_train)

    x_train = to_one_hot(data, x_train)
    y_train = to_one_hot(labels, y_train)
    return x_train, y_train

  def _transform(self, text, words=None):
    x_train, y_train = [], []
    try:
       mapping = self.mapping
    except AttributeError:
      err_msg = ("Call `fit()` method first!")
      raise ValueError(err_msg)

    if words is None:
      words = [START + w + END for w in text.split()]
      words = [w + ''.join([NAW]*(self.max_wordlen - len(w))) for w in words]


    for word in words:
      item = []
      for l in word:
        item.append(mapping[l])
      x_train.append(np.array(item))
      y_train.append(np.array(item))

    x_train = np.stack(x_train)
    y_train = np.stack(y_train)

    return x_train, y_train

  def _transform_corrupted(self, text):
    
    words = [START + w + END for w in text.split()]
    words = [w + ''.join([NAW]*(self.max_wordlen - len(w))) for w in words]
    corrupted_words = []
    start_idx = 1
    for word in words:
      word = list(word)
      end_idx = word.index(END) - 1

      if np.random.uniform() > 0.6:
        idx1 = np.round(np.random.uniform(low=1, high=end_idx))
        idx2 = np.round(np.random.uniform(low=1, high=end_idx))
        idx1, idx2 = int(idx1), int(idx2)
        word[idx1], word[idx2] = word[idx2], word[idx1]
      if np.random.uniform() > 0.5:

        idx = np.round(np.random.uniform(low=1, high=end_idx))
        idx = int(idx)
        word[idx] = random.choice(self.alphabet)
        if np.random.uniform() > 0.5:
          idx = np.round(np.random.uniform(low=1, high=end_idx))
          idx = int(idx)
          word[idx] = random.choice(self.alphabet)

      corrupted_words.append(''.join(word))

    labels, _ = self._transform(text, corrupted_words)
    data, _ = self._transform(text)

    x_train = np.zeros(shape=data.shape + (len(self.mapping), ))
    y_train = np.zeros_like(x_train)

    x_train = to_one_hot(data, x_train)
    y_train = to_one_hot(labels, y_train)
    return x_train, y_train

  def inverse_transform(self, arys):
    if len(arys.shape) > 2:
      res = np.zeros(shape=arys.shape[:-1], dtype=np.int32)
      res = from_one_hot(arys, res)
    else:
      res = arys
    words = []
    for ary in res:
        item = [self.inverse_mapping[int(a)] for a in ary]
        item = [i for i in item if i not in [START, END, NAW]]
        words.append(''.join(item))
    return words


@nb.njit(parallel=True, fastmath=True)
def to_one_hot(ary, res):
  for i in nb.prange(ary.shape[0]):
    for j in nb.prange(ary.shape[1]):
      res[i][j][int(ary[i][j])] = 1.

  return res

@nb.njit(parallel=True, fastmath=True)
def from_one_hot(ary, res):
  for i in nb.prange(ary.shape[0]):
    for j in nb.prange(ary.shape[1]):
      res[i][j] = np.int32(np.argmax(ary[i][j]))

  return res

def print_log(epoch, step, test_err, train_err, test_loss, train_loss):
  """Helper for logs during training."""
  buff = ("\r[epoch:{0}]|[step:{1}]|[test-error:{2:.3f}]|[train-error:{3:.3f}]|"
          "[test-loss:{4:.3f}]|[train-loss:{5:.3f}]")
  buff = buff.format(
      epoch, step, test_err, train_err, test_loss, train_loss)
  sys.stdout.write(buff)
  sys.stdout.flush()

def generate_plots(log):

  def apply_filter(x, y, sigma=1):
    ynew = gaussian_filter1d(y, sigma=sigma)
    return x, ynew

  FONTSIZE = 23
  
  n_epochs = log['n_epochs']
  colors = plt.rcParams['axes.prop_cycle'].by_key()['color']

  fig, ax = plt.subplots(figsize=(12, 8))

  # train-test loss
  train_loss = log['train_loss']
  test_loss = log['test_loss']
  x = np.linspace(1, n_epochs, len(train_loss))
  x, train_loss = apply_filter(x, train_loss)
  ax.plot(x, train_loss, color=colors[0], linestyle='--', linewidth=2,
    label='Train Loss')
  x = np.linspace(1, n_epochs, len(test_loss))
  x, test_loss = apply_filter(x, test_loss)
  ax.plot(x, test_loss, color=colors[0], linewidth=4, label='Test Loss')

  plt.yticks(fontsize=FONTSIZE)
  plt.xticks(fontsize=FONTSIZE)
  plt.xlabel('Epochs', fontsize=FONTSIZE + 3)
  plt.ylabel('Loss', fontsize=FONTSIZE + 3)
  plt.rcParams['legend.loc'] = 'upper right'
  leg = plt.legend(fancybox=True, prop={'size': 20})
  leg.get_frame().set_edgecolor('black')
  leg.get_frame().set_linewidth(3)
  plt.show()
  #plt.savefig('train-test-loss.png', bbox_inches='tight')

  # test error
  fig, ax = plt.subplots(figsize=(12, 8))

  # train-test loss
  test_error = log['test_error']
  train_error = log['train_error']

  x = np.linspace(1, n_epochs, len(test_error))
  x, test_error = apply_filter(x, test_error, sigma=3)
  ax.plot(x, test_error, color=colors[0], linewidth=4, label='Test Error')

  x = np.linspace(1, n_epochs, len(train_error))
  x, train_error = apply_filter(x, train_error, sigma=3)
  ax.plot(x, train_error, color=colors[0], linewidth=3, linestyle='--', label='Train Error')
  
  min_val = min(test_error)
  ax.plot([1, n_epochs], 2*[min_val], linestyle='--', color='black', linewidth=3)
  xticks = [0.85, 0.8, 0.75, 0.7, float("{0:.3f}".format(min_val)), float("{0:.3f}".format(min_val - 0.05))]
  plt.yticks(xticks, fontsize=FONTSIZE)
  plt.xticks(fontsize=FONTSIZE)
  plt.xlabel('Epochs', fontsize=FONTSIZE + 3)
  plt.ylabel('Error', fontsize=FONTSIZE + 3)
  plt.ylim(min_val - 0.055, 0.855)
  plt.rcParams['legend.loc'] = 'upper right'
  leg = plt.legend(fancybox=True, prop={'size': 20})
  leg.get_frame().set_edgecolor('black')
  leg.get_frame().set_linewidth(3)
  
  #plt.savefig('test-error.png', bbox_inches='tight')
  plt.show()
