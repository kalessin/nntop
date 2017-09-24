import tensorflow as tf
import numpy as np

class Layer(object):
    def __init__(self, name, shape, activation=None, dropout=None):
        """
        shape - Layer shape
        activation - current supported: None, 'relu', 'sigmoid', 'tanh'
        """
        self.__name = name
        self.__shape = shape
        self.__W = None
        self.__b = None
        self.__activation = activation
        self.__dropout = dropout

    @property
    def shape(self):
        return self.__shape

    @property
    def name(self):
        return self.__name

    def __call__(self, X):
        result = self.op(X)
        if self.__activation == 'relu':
            result = tf.nn.relu(result, name="%s_relu" % self.name)
        elif self.__activation == 'sigmoid':
            result = tf.nn.sigmoid(result, name="%s_sigmoid" % self.name)
        elif self.__activation == 'tanh':
            result = tf.nn.tanh(result, name="%s_tanh" % self.name)
        if self.__dropout:
            result = tf.nn.dropout(result, self.__dropout, name="%s_dropout" % self.name)
        return result

    @property
    def weights_matrix(self):
        if self.__W is None and self.__shape is not None:
            initial = tf.truncated_normal(self.__shape, stddev=0.2)
            self.__W = tf.Variable(initial, name='%s_W' % self.name)
        return self.__W

    @property
    def bias_vector(self):
        if self.__b is None and self.__shape is not None:
            initial = tf.constant(0.1, shape=[self.__shape[-1]])
            self.__b = tf.Variable(initial, name='%s_b' % self.name)
        return self.__b

    def op(self, X):
        # first convert to vector the input tensor, where only first dimension (number of samples)
        # is conserved
        input_shape = [-1, np.prod([s.value for s in X.shape[1:]])]
        x = tf.reshape(X, input_shape, name="%s_reshape" % self.name)

        if self.__shape[0] is None:
            self.__shape = [input_shape[1], self.__shape[1]]

        return tf.add(tf.matmul(x, self.weights_matrix, name="%s_mul" % self.__name),
                      self.bias_vector, name="%s_add" % self.name)


class Convolution(Layer):
    def __init__(self, name, img_shape, field_shape, strides_shape, filters, input_channels=1,
                 padding='SAME', activation=None):
        """
        img_shape - input img shape. If not None, input vector will be reshaped to the given shape.
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
        if self.__img_shape is not None:
            X = tf.reshape(X, [-1, self.__img_shape[0], self.__img_shape[1], self.__input_channels],
                           name="%s_reshape" % self.name)
        return tf.add(tf.nn.conv2d(X, self.weights_matrix, strides=[1, self.__strides_shape[0],
                                                                    self.__strides_shape[1], 1],
                                   padding=self.__padding, name=self.name),
                      self.bias_vector, name="%s_add" % self.name)


class Relu(object):
    def __init__(self, name):
        self.__name = name

    def __call__(self, X):
        return tf.nn.relu(X, name=self.__name)


class Model(object):
    def __init__(self, name, num_features, num_classes):
        self.__compiled = False
        self.__train_epochs = 0

        graph = None
        if num_features is None or num_classes is None: # restore model mode
            sess = tf.Session()
            saver = tf.train.import_meta_graph('%s.meta' % name)
            saver.restore(sess, name)
            graph = tf.get_default_graph()
            num_features = graph.get_tensor_by_name("num_features:0")
            num_classes = graph.get_tensor_by_name("num_classes:0")
            self.__last_name = str(sess.run(["last_name:0"])[0].decode())

        self.__num_classes = num_classes
        self.__num_features = num_features
        self.__name = name

        self.__train_loss = self.__valid_loss = None

        self.__output = None
        self.__prediction = None

        self.__loss = self.__train_step = None
        self.__accuracy = self.__confusion_matrix = None
        self.__test_accuracy = None
        self.__test_confusion_matrix = None

        self.__graph = graph or tf.Graph()
        with self.__graph.as_default():
            if graph: # we are restoring model
                self.__X = self.__graph.get_tensor_by_name("X:0")
                self.__y = self.__graph.get_tensor_by_name("y:0")
                self.__last = self.__graph.get_tensor_by_name(self.__last_name)
                self._compile(restore=True)
            else: # we are creating a new model
                self.__X = tf.placeholder(tf.float32, shape=[None, self.__num_features], name='X')
                self.__y = tf.placeholder(tf.float32, shape=[None, self.__num_classes], name='y')

                # just to save these initialization parameters
                tf.constant(self.__num_classes, name="num_classes")
                tf.constant(self.__num_features, name="num_features")
                self.__last = self.__X
                self.__last_name = self.__last.name

    @property
    def train_epochs(self):
        return self.__train_epochs

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
    def final_test_confusion_matrix(self):
        return self.__test_confusion_matrix

    def append(self, tensor_op):
        with self.__graph.as_default():
            self.__last = tensor_op(self.__last)
            self.__last_name = self.__last.name

    def train(self, batches, validation_set, test_set, steps, print_every=None):
        """
        batches - a generator that yields a tuple of two iterators on each next cycle: one for samples and another for the corresponding labels
        validation_set - a 2-element tuple: first is are the validation samples, and second are the corresponding labels
        test_set - a 2-element tuple: first is are the test samples, and second are the corresponding labels
        steps - how many training steps to perform
        print_every - print loss value each given number of steps. If None, it will be set to steps / 10
        """
        print_every = print_every or steps // 10
        with self.__graph.as_default():
            batchX, batchY = next(batches)

            self._compile()

            with tf.Session(graph=self.__graph) as sess:
                if self.__train_epochs == 0:
                    init = tf.global_variables_initializer()
                    sess.run(init)
                else:
                    saver = tf.train.import_meta_graph('%s.meta' % self.__name)
                    saver.restore(sess, self.__name)
                self.__train_epochs += 1
                for i in range(1, steps + 1):
                    self.__train_loss, _ = sess.run([self.__loss, self.__train_step],
                                                    feed_dict={self.__X: batchX,
                                                               self.__y: batchY})
                    if i % print_every == 0:
                        self.__valid_loss = self.__loss.eval(feed_dict={self.__X: validation_set[0],
                                                                        self.__y: validation_set[1]})
                        print('step %d, train loss %g, validation loss %g' % (i, self.__train_loss, self.__valid_loss))
                    batchX, batchY = next(batches)
                self.__test_accuracy = self.__accuracy.eval(feed_dict={self.__X: test_set[0],
                                                                       self.__y: test_set[1]})
                print("Test accuracy: %g" % self.__test_accuracy)

                self.__test_confusion_matrix = self.__confusion_matrix.eval(feed_dict={self.__X: test_set[0],
                                                                                       self.__y: test_set[1]})

                saver = tf.train.Saver()
                saver.save(sess, self.__name)

    def _compile(self, restore=False):
        if not self.__compiled:
            if restore:
                self.__output = self.__graph.get_tensor_by_name("modeloutput:0")
                self.__prediction = self.__graph.get_tensor_by_name("modelprediction:0")
                self.__loss = self.__graph.get_tensor_by_name("modelloss:0")
                self.__train_step = self.__graph.get_tensor_by_name("modeltrainstep:0")
                self.__accuracy = self.__graph.get_tensor_by_name("modelaccuracy:0")
                self.__confusion_matrix = self.__graph.get_tensor_by_name("modelconfusionmatrix:0")
            else:
                tf.Variable(self.__last_name, name="last_name")
                self.__output = tf.nn.softmax(self.__last, name="modeloutput")
                self.__prediction = tf.argmax(self.__output, 1, name='modelprediction')
                self.__loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=self.__y, logits=self.__last), name="modelloss")
                self.__train_step = tf.train.AdamOptimizer(1e-4).minimize(self.__loss, name='modeltrainstep')

                labels = tf.argmax(self.__y, 1)
                correct_prediction = tf.equal(self.__prediction, labels)
                self.__accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32), name='modelaccuracy')
                self.__confusion_matrix = tf.confusion_matrix(labels, self.__prediction, self.__num_classes, name='modelconfusionmatrix')
            self.__compiled = True

    def output(self, X):
        with self.__graph.as_default():
            with tf.Session(graph=self.__graph) as sess:
                saver = tf.train.Saver()
                saver.restore(sess, self.__name)
                output, prediction = sess.run([self.__output, self.__prediction], feed_dict={self.__X: X})
                return output, prediction

    def prediction(self, X):
        return self.output(X)[1]
