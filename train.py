import pickle

import tensorflow as tf
import numpy as np

import utils
import rnns

class Trainer:

  def __init__(self,
               n_epochs=250,
               batch_size=8,
               n_outputs=29,
               model=None,
               x_shape=(None, 17, 29),
               y_shape=(None, 17, 29),
               optimizer=None,
               lr=0.01,
               initializer_fn=None):

    self.n_epochs = n_epochs
    self.batch_size = batch_size
    self.n_outputs = n_outputs
    self.model = model or rnns.simple_rnn
    self.x_shape = x_shape
    self.y_shape = y_shape
    self.optimizer = optimizer or tf.train.GradientDescentOptimizer
    self.lr = lr
    self.initializer_fn = initializer_fn
    self.graph = tf.Graph()
    
    if self.initializer_fn is None:
      with self.graph.as_default():
        self.istrain = tf.placeholder_with_default(False, shape=())
        self._x = tf.placeholder(tf.float32, x_shape)
        self._y = tf.placeholder(tf.int32, y_shape)

        self._logits = self.model(self._x, n_outputs=y_shape[-1])

        
        self._loss_tensor = cross_entropy_loss(
            self._y, self._logits)

        y_pred = tf.nn.softmax(self._logits, axis=-1)
        y_pred = tf.argmax(y_pred, axis=-1, output_type=tf.int32)
        self.y_pred = y_pred
        equals = tf.equal(y_pred,
                          tf.argmax(self._y, axis=-1, output_type=tf.int32))
        equals = tf.cast(equals, tf.float32)
        self._error_tensor = 1.0 - tf.reduce_mean(equals)

        optimizer = self.optimizer(self.lr)
        self._update_op = optimizer.minimize(self._loss_tensor)

    else:
      self.initializer_fn(self.graph)

  def fit(self, x_train, y_train, x_test, y_test):
    """Fits the model.

    Args:
      x/y_train/test: Data and labels.

    Returns:
      A dictionary, train-test loss and accuracy.

    """
    # datasets and iterators
    with self.graph.as_default():
      dataset = tf.data.Dataset.from_tensor_slices({
          'x': x_train,
          'y': y_train
      }).shuffle(x_train.shape[0]).batch(self.batch_size)
      train_iter = dataset.make_initializable_iterator()
      dataset = tf.data.Dataset.from_tensor_slices({
          'x': x_test,
          'y': y_test
      }).batch(self.batch_size)
      test_iter = dataset.make_initializable_iterator()
      config = tf.ConfigProto(allow_soft_placement=True)
      init = [tf.global_variables_initializer(),
              train_iter.initializer,
              test_iter.initializer]

    train_log = {'train_loss': [],
                 'test_loss': [],
                 'train_error': [],
                 'test_error': []}

    with tf.Session(graph=self.graph, config=config) as sess:
      _ = sess.run(init)
      next_batch_train = train_iter.get_next()
      next_batch_test = test_iter.get_next()
      step = 0

      for epoch in range(self.n_epochs):
        batch_loss, batch_err = [], []
        while True:
          try:
            batch = sess.run(next_batch_train)
            step += 1
            if epoch >= 1:
                utils.print_log(epoch + 1,
                                step,
                                train_log['test_error'][-1],
                                train_log['train_error'][-1],
                                train_log['test_loss'][-1],
                                train_log['train_loss'][-1])

            feed_dict = {
                self._x: batch['x'],
                self._y: batch['y'],
                self.istrain: True
            }

            _ = sess.run(self._update_op, feed_dict=feed_dict)

            tensors = [self._loss_tensor, self._error_tensor]
            del feed_dict[self.istrain]
            loss_val, err_val = sess.run(tensors, feed_dict=feed_dict)
            batch_loss.append(loss_val)
            batch_err.append(err_val)
          except tf.errors.OutOfRangeError:
            sess.run(train_iter.initializer)
            loss_val, err_val = self._compute_test_loss_and_error(
                sess, test_iter, next_batch_test)
            train_log['test_error'].append(err_val)
            train_log['test_loss'].append(loss_val)
            train_log['train_error'].append(np.mean(batch_err))
            train_log['train_loss'].append(np.mean(batch_loss))
            break

      variables = tf.trainable_variables()
      variables = sess.run(variables)
      with open('variables.pkl', 'wb') as fo:
        pickle.dump(variables, fo, protocol=pickle.HIGHEST_PROTOCOL)
        self.variables = variables
    train_log['n_epochs'] = self.n_epochs
    return train_log

  def inference(self, ary):
    with tf.Session(graph=self.graph) as sess:
      sess.run(tf.global_variables_initializer())
      _ = [var.load(val) for var, val in zip(tf.trainable_variables(), self.variables)]
      predicted = sess.run(self.y_pred, feed_dict={self._x: ary})
      return predicted

  def _compute_test_loss_and_error(self, sess, iterator, next_batch):
    batch_loss = []
    batch_err = []
    while True:
      try:
        batch = sess.run(next_batch)
        err_val, loss_val = sess.run([self._error_tensor,
                                      self._loss_tensor],
                                     feed_dict={self._x: batch['x'],
                                                self._y: batch['y']})
        batch_loss.append(loss_val)
        batch_err.append(err_val)

      except tf.errors.OutOfRangeError:
        sess.run(iterator.initializer)
        break
    return np.mean(batch_loss), np.mean(batch_err)


def cross_entropy_loss(labels, logits, softmax_axis=-1):
    labels = tf.cast(labels, tf.float32)
    softmaxed = tf.nn.softmax(logits, axis=softmax_axis)
    xentropy = -labels * tf.math.log(softmaxed)
    return tf.reduce_mean(xentropy)


def corrupt_inputs(self, x, istrain, thresh=tf.constant(0.6)):

  def true_fn(x):
    N = x.get_shape().as_list()[-2]


    n_items = tf.random.uniform(shape=(), minval=0, maxval=N)
    n_items = tf.cast(tf.round(n_items), tf.int32) # number of items to shuffle

    indices = tf.round(
        tf.random.uniform(shape=(n_items, ), minval=0, maxval=N))
    indices = tf.cast(indices, tf.int32)


    
    


  return tf.cond(istrain,
                 true_fn=lambda: true_fn(x),
                 false_fn=lambda: tf.identity(x))