import tensorflow as tf
import numpy as np

class Layer(object):
    def __init__(self, name, shape, activation=None):
        """
        shape - Layer shape
        activation - current supported: None, 'relu', 'sigmoid', 'tanh'
        """
        self.__name = name
        self.__shape = shape
        self.__W = None
        self.__b = None
        self.__activation = activation

    @property
    def shape(self):
        return self.__shape
    
    @property
    def name(self):
        return self.__name

    @property
    def weights(self):
        return self.__W
    
    @property
    def bias_vector(self):
        return self.__b

    def __call__(self, X):
        self.init_weight_variable()
        self.init_bias_variable()
        return self.__op(X)
        
    def __op(self, X):
        result = self.op()
        if self.__activation == 'relu':
            result = tf.nn.relu(result, name="%s_relu" % self.name)
        elif self.__activation == 'sigmoid':
            result = tf.nn.sigmoid(result, name="%s_sigmoid" % self.name)
        elif self.__activation == 'tanh':
            result = tf.nn.tanh(result, name="%s_tanh" % self.name)
        return result

    def op(self, X):
        x = tf.reshape(X, [-1, self.__shape[0]], name="%s_reshape" % self.name)
        result = tf.add(tf.matmul(x, self.__W, name="%s_mul" % self.__name), self.__b, name="%s_add" % self.name)
        return self.apply_activation(result)

    def init_weight_variable(self, stddev=0.2):
        if self.__W is None:
            initial = tf.truncated_normal(self.__shape, stddev=stddev)
            self.__W = tf.Variable(initial, name='%s_W' % self.name)

    def init_bias_variable(self):
        if self.__b is None:
            initial = tf.constant(0.1, shape=[self.__shape[-1]])
            self.__b = tf.Variable(initial, name='%s_b' % self.name)


class Convolution(Layer):
    def __init__(self, name, img_shape, field_shape, strides_shape, filters, input_channels=1, padding='SAME', activation=None):
        """
        field_shape - field Height x field Width
        strides_shape - strides Height x strides Width
        filters - number of filters
        input_channels - number of input channels
        padding - padding type
        """
        super(Convolution, self).__init__(name, [field_shape[0], field_shape[1], input_channels, filters], activation=activation)
        self.__strides_shape = strides_shape
        self.__img_shape = img_shape
        self.__input_channels = input_channels
        self.__padding = padding

    def op(self, X):
        x = tf.reshape(X, [-1, self.__img_shape[0], self.__img_shape[1], self.__input_channels],
                       name="%s_reshape" % self.name)
        return tf.add(tf.nn.conv2d(x, self.weights, strides=[1, self.__strides_shape[0], self.__strides_shape[1],
                                                             1],
                                   padding=self.__padding, name=self.name),
                      self.bias_vector, name="%s_add" % self.name)


class Relu(object):
    def __init__(self, name):
        self.__name = name

    def __call__(self, X):
        return tf.nn.relu(X, name=self.__name)


class Model(object):
    def __init__(self, num_features, num_classes, name):
        self.__X = self.__last = self.__y = None
        self.__K = num_classes
        self.__num_features = num_features
        self.__name = name
        self.__graph = tf.Graph()

        self.__trained = False
        self.__train_loss = self.__valid_loss = None
        
        self.__output = None
        self.__prediction = None
        
        self.__test_accuracy = None
        self.__test_confusion_matrix = None
    
    @property
    def filename(self):
        return "%s.ckpt" % self.__name

    @property
    def final_train_loss(self):
        return self.__train_loss

    @property
    def final_validation_loss(self):
        return self.__valid_loss
    
    @property
    def final_test_accuracy(self):
        return self.__test_accuracy
    
    @property
    def final_confusion_matrix(self):
        return self.__test_confusion_matrix

    def append(self, tensor_op):
        with self.__graph.as_default() as graph:
            if self.__X is None:
                self.__X = tf.placeholder(tf.float32, shape=[None, self.__num_features])
                self.__last = self.__X
            self.__last = tensor_op(self.__last)
    
    def train(self, batches, validation_set, test_set, steps, print_every=None):
        """
        batches - a generator that yields a tuple of two iterators on each next cycle: one for samples and another for the corresponding labels
        validation_set - a 2-element tuple: first is are the validation samples, and second are the corresponding labels
        test_set - a 2-element tuple: first is are the test samples, and second are the corresponding labels
        steps - how many training steps to perform
        print_every - print loss value each given number of steps. If None, it will be set to steps / 10
        """
        print_every = print_every or steps // 10
        if not self.__trained:
            with self.__graph.as_default():
                batchX, batchY = next(batches)
                num_classes = batchY.shape[1]

                self.__output = tf.nn.softmax(self.__last)
                self.__prediction = tf.argmax(self.__output, 1)

                self.__y = tf.placeholder(tf.float32, shape=[None, num_classes])
                loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=self.__y, logits=self.__last))
                train_step = tf.train.AdamOptimizer(1e-4).minimize(loss)

                labels = tf.argmax(self.__y, 1)
                correct_prediction = tf.equal(self.__prediction, labels)
                self.__accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
                self.__confusion_matrix = tf.confusion_matrix(labels, self.__prediction, num_classes)

                init = tf.global_variables_initializer()
                with tf.Session(graph=self.__graph) as sess:
                    sess.run(init)
                    for i in range(1, steps + 1):
                        self.__train_loss, _ = sess.run([loss, train_step], feed_dict={self.__X: batchX,
                                                                                       self.__y: batchY})
                        if i % print_every == 0:
                            self.__valid_loss = loss.eval(feed_dict={self.__X: validation_set[0],
                                                                     self.__y: validation_set[1]})
                            print('step %d, train loss %g, validation loss %g' % (i, self.__train_loss, self.__valid_loss))
                        batchX, batchY = next(batches)
                    self.__test_accuracy = self.__accuracy.eval(feed_dict={self.__X: test_set[0],
                                                                           self.__y: test_set[1]})
                    print("Test accuracy: %g" % self.__test_accuracy)

                    self.__test_confusion_matrix = self.__confusion_matrix.eval(feed_dict={self.__X: test_set[0],
                                                                                 self.__y: test_set[1]})

                    saver = tf.train.Saver()
                    saver.save(sess, self.filename)
            self.__trained = True

    def output(self, X):
        with self.__graph.as_default():
            if self.__output is None:
                self.__output = tf.nn.softmax(self.__last)
                self.__prediction = tf.argmax(self.__output, 1)

            with tf.Session(graph=self.__graph) as sess:
                saver = tf.train.Saver()
                saver.restore(sess, self.filename)
                output, prediction = sess.run([self.__output, self.__prediction], feed_dict={self.__X: X})
                return output, prediction
    
    def prediction(self, X):
        return self.output(X)[1]
