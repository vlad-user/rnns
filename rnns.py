import os
os.environ["CUDA_VISIBLE_DEVICES"]="-1"

import tensorflow as tf
import numpy as np

import utils

class SimpleRNNCell:

  def __init__(self, units, n_outputs, dim=1):
    self.w_xh = tf.Variable(tf.random.normal((dim, units)))
    self.w_hh = tf.Variable(tf.random.normal((units, units)))
    self.w_hy = tf.Variable(tf.random.normal((units, n_outputs)))

  def __call__(self, inputs, state):
    next_state = tf.tanh((tf.matmul(inputs, self.w_xh)
                          + tf.matmul(state, self.w_hh)))
    output = tf.matmul(next_state, self.w_hy)
    return output, next_state

class LSTMCell:
  def __init__(self, units, n_outputs, dim):
    self.units = units
    self.w_xi = tf.Variable(tf.random.normal((dim, units)))
    self.w_hi = tf.Variable(tf.random.normal((units, units)))
    self.w_xf = tf.Variable(tf.random.normal((dim, units)))
    self.w_hf = tf.Variable(tf.random.normal((units, units)))
    self.w_xo = tf.Variable(tf.random.normal((dim, units)))
    self.w_ho = tf.Variable(tf.random.normal((units, units)))
    self.w_xz = tf.Variable(tf.random.normal((dim, units)))
    self.w_hz = tf.Variable(tf.random.normal((units, units)))
    self.w_hy = tf.Variable(tf.random.normal((units, n_outputs)))


  def __call__(self, inputs, state):
    h = state[0]
    c = state[1]

    i = tf.nn.sigmoid((tf.matmul(inputs, self.w_xi)
                      + tf.matmul(h, self.w_hi)))
    f = tf.nn.sigmoid((tf.matmul(inputs, self.w_xf)
                      + tf.matmul(h, self.w_hf)))
    o = tf.nn.sigmoid((tf.matmul(inputs, self.w_xo)
                      + tf.matmul(h, self.w_ho)))
    z = tf.nn.tanh((tf.matmul(inputs, self.w_xz)
                    + tf.matmul(h, self.w_hz)))
    next_c = f*c + i*z
    next_h = o*tf.nn.tanh(next_c)
    output = tf.matmul(next_h, self.w_hy)
    return output, (next_h, next_c)

class GRUCell:

  def __init__(self, units, n_outputs, dim):
    self.w_xz = tf.Variable(tf.random.normal((dim, units)))
    self.w_sz = tf.Variable(tf.random.normal((units, units)))
    self.w_xr = tf.Variable(tf.random.normal((dim, units)))
    self.w_sr = tf.Variable(tf.random.normal((units, units)))
    self.w_xs = tf.Variable(tf.random.normal((dim, units)))
    self.w_sg = tf.Variable(tf.random.normal((units, units)))
    self.w_sy = tf.Variable(tf.random.normal((units, n_outputs)))

  def __call__(self, inputs, state):

    z = tf.nn.sigmoid((tf.matmul(inputs, self.w_xz)
                      + tf.matmul(state, self.w_sz)))
    r = tf.nn.sigmoid((tf.matmul(inputs, self.w_xr)
                      + tf.matmul(state, self.w_sr)))
    s = tf.nn.tanh((tf.matmul(inputs, self.w_xs)
                   + tf.matmul(r*state, self.w_sg)))
    next_state = z*state + (1 - z)*s
    output = tf.matmul(next_state, self.w_sy)
    return output, next_state

def build_rnn(x, RNN, units, n_outputs, embed_dim, next_state=None):
  next_state = next_state or [tf.zeros(shape=(1, units))]
  outputs = []
  x = _embedding(x, embed_dim)
  unstacked_x = tf.unstack(x, axis=1)
  rnncell = RNN(units, n_outputs, embed_dim)

  for input_ in unstacked_x:
    output, next_state = rnncell(input_, next_state)
    outputs.append(output)

  logits = tf.concat([output[:, None, ...] for output in outputs], 1)
  return logits

def lstm(x, units=64, n_outputs=29, embed_dim=32):
  return build_rnn(x,
                   LSTMCell,
                   units,
                   n_outputs,
                   embed_dim,
                   [tf.zeros((1, units)), tf.zeros((1, units))])

def gru(x, units=128, n_outputs=29, embed_dim=32):
  return build_rnn(x, GRUCell, units, n_outputs, embed_dim)

def simple_rnn(x, units=32, n_outputs=29, embed_dim=32):
  return build_rnn(x, SimpleRNNCell, units, n_outputs, embed_dim)

def _embedding(x, embed_dim=32):
  shape = (x.get_shape().as_list()[-1], embed_dim)
  w = tf.Variable(tf.random.normal(shape))
  b = tf.Variable(tf.zeros(shape[1:]))
  res = tf.tensordot(x, w, axes=(2, 0)) + b
  return res









