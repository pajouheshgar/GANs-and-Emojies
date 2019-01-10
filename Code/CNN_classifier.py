import os
import datetime
import tensorflow as tf
import numpy as np
from Code.Models.BaseModel import BaseModel

flags = tf.app.flags
FLAGS = flags.FLAGS
flags.DEFINE_float('ilr', 0.01, 'initial_learning_rate')
flags.DEFINE_integer('decay_steps', 1000, 'steps to halve the learning rate')
flags.DEFINE_integer('epochs', 10, 'Number of epochs to train.')

from Code.Dataloaders import Conditional_GAN_Dataloader


class CNN(BaseModel):
    NAME = "CNN"
    SAVE_DIR = "../../Dataset/Models/"
    FILTERS_LIST = [16, 32, 64]
    STRIDES_LIST = [[2, 2], [2, 2], [2, 2]]
    KERNEL_SIZE_LIST = [[5, 5], [5, 5], [5, 5]]
    PADDING_LIST = ['s', 's', 's', 'v', 'v']
    DESCRIPTION = ""

    def __init__(self, name=NAME, save_dir=SAVE_DIR, summary=True, filters_list=FILTERS_LIST,
                 strides_list=STRIDES_LIST, kernel_size_list=KERNEL_SIZE_LIST, padding_list=PADDING_LIST):
        super(CNN, self).__init__(name, save_dir)
        self.MODEL_PATH = os.path.join(self.SAVE_DIR, self.NAME)
        try:
            os.mkdir(self.MODEL_PATH)
        except:
            pass
        init_time = datetime.datetime.utcnow().strftime("%Y%m%d%H%M%S")
        self.LOG_PATH = os.path.join(self.SAVE_DIR, "log/" + self.NAME + "-run-" + init_time)

        self.filters_list = filters_list
        self.strides_list = strides_list
        self.kernel_size_list = kernel_size_list
        self.padding_list = padding_list
        assert len(self.filters_list) == len(self.strides_list)
        assert len(self.filters_list) == len(self.kernel_size_list)
        assert len(self.filters_list) == len(self.padding_list)
        self.build_graph(summary)
        self.ses = tf.InteractiveSession()

    def build_graph(self, summary):
        tf.set_random_seed(42)
        np.random.seed(42)
        with tf.variable_scope(self.NAME):
            with tf.name_scope("Dataset"):
                self.dataloader = Conditional_GAN_Dataloader(flatten=False)
                self.X_train, self.Y_train = self.dataloader.train_batch
                self.X_validation, self.Y_validation = self.dataloader.validation_batch
                self.X_test, self.Y_test = self.dataloader.test_batch

            with tf.name_scope("Placeholders"):
                self.X_placeholder = tf.placeholder_with_default(self.X_train,
                                                                 shape=[None, FLAGS.image_height, FLAGS.image_width,
                                                                        1],
                                                                 name="X_placeholder")
                self.Y_placeholder = tf.placeholder_with_default(self.Y_train,
                                                                 shape=[None, FLAGS.num_classes],
                                                                 name="Y_placeholder")
                self.is_training_placeholde = tf.placeholder_with_default(True, [], name='is_training_placeholder')

            with tf.variable_scope("Inference"):
                self.conv_layers = []
                last_layer = self.X_placeholder
                for i in range(len(self.filters_list)):
                    filters = self.filters_list[i]
                    strides = self.strides_list[i]
                    kernel_size = self.kernel_size_list[i]
                    padding = 'valid' if self.padding_list[i] == 'v' else 'same'
                    conv_layer = tf.layers.conv2d(last_layer,
                                                  filters,
                                                  kernel_size=kernel_size,
                                                  strides=strides,
                                                  padding=padding,
                                                  # padding="valid",
                                                  activation=tf.nn.relu,
                                                  name='conv{}_layer'.format(i))
                    last_layer = conv_layer
                    print(last_layer.shape)
                    self.conv_layers.append(last_layer)

                conv_flatten = tf.layers.flatten(last_layer, name='conv_flatten')
                # hidden_layer1 = tf.layers.dense(conv_flatten, units=5 * FLAGS.num_classes, activation=tf.nn.relu,
                #                                 name='hidden_layer1')
                # hidden_layer1_dropped_out = tf.layers.dropout(hidden_layer1, rate=0.5,
                #                                               training=self.is_training_placeholde,
                #                                               name='hidden_layer1_dropped_out')
                #
                # hidden_layer2 = tf.layers.dense(hidden_layer1_dropped_out, units=50 * FLAGS.num_classes,
                #                                 activation=tf.nn.relu,
                #                                 name='hidden_layer2')
                # hidden_layer2_dropped_out = tf.layers.dropout(hidden_layer2, rate=0.5,
                #                                               training=self.is_training_placeholde,
                #                                               name='hidden_layer2_dropped_out')

                logits = tf.layers.dense(conv_flatten, units=FLAGS.num_classes, name='logits')
                self.estimated_class_probabilities = tf.nn.softmax(logits, name='estimated_class_probabilities')

            with tf.name_scope("Optimization"):
                self.loss = tf.reduce_mean(
                    tf.nn.softmax_cross_entropy_with_logits(labels=self.Y_placeholder, logits=logits), name='loss')

                correct_prediction = tf.equal(tf.argmax(self.estimated_class_probabilities, axis=1),
                                              tf.argmax(self.Y_placeholder, axis=1))
                self.accuracy = tf.reduce_mean(tf.cast(correct_prediction, dtype=tf.float32))

                self.global_step = tf.train.get_or_create_global_step()
                learning_rate = tf.train.exponential_decay(FLAGS.ilr, self.global_step, FLAGS.decay_steps,
                                                           0.5,
                                                           name='learning_rate')
                optimizer = tf.train.AdamOptimizer(learning_rate)
                self.train_operation = optimizer.minimize(self.loss, global_step=self.global_step)

            self.init_node = tf.global_variables_initializer()
            self.save_node = tf.train.Saver()

            if summary:
                with tf.name_scope("Summary"):
                    self.loss_summary = tf.summary.scalar(name="loss", tensor=self.loss)
                    self.accuracy_summary = tf.summary.scalar(name="accuracy", tensor=self.accuracy)
                    self.scalar_summaries = tf.summary.merge(
                        [
                            self.loss_summary,
                            self.accuracy_summary,
                        ])

                    self.train_summary_writer = tf.summary.FileWriter(self.LOG_PATH + "train",
                                                                      tf.get_default_graph())
                    self.validation_summary_writer = tf.summary.FileWriter(self.LOG_PATH + "validation")

    def add_code_summary(self):
        code_string = "\n".join(open(os.path.basename(__file__), 'r').readlines())
        text_tensor = tf.make_tensor_proto(self.DESCRIPTION + "\n\n" + code_string, dtype=tf.string)
        meta = tf.SummaryMetadata()
        meta.plugin_data.plugin_name = "text"
        summary = tf.Summary()
        summary.value.add(tag="Hyper parameters", metadata=meta, tensor=text_tensor)
        self.train_summary_writer.add_summary(summary)

    def init_variables(self):
        self.ses.run(self.init_node)

    def save(self):
        self.save_node.save(self.ses, save_path=self.MODEL_PATH + "/" + self.NAME + '.ckpt')

    def load(self):
        self.save_node.restore(self.ses, save_path=self.MODEL_PATH + "/" + self.NAME + '.ckpt')

    def get_loss_accuracy(self, type='train'):
        assert type == 'train' or type == 'test' or type == 'validation'
        if type == 'train':
            initializer = self.dataloader.train_initializer
            x, y = self.X_train, self.Y_train
        elif type == 'test':
            initializer = self.dataloader.test_initializer
            x, y = self.X_test, self.Y_test
        else:
            initializer = self.dataloader.validation_initializer
            x, y = self.X_validation, self.Y_validation

        self.ses.run(initializer)
        loss = 0
        accuracy = 0
        n = 0
        while True:
            try:
                x_fetched, y_fetched = self.ses.run([x, y])
                feed_dict = {self.X_placeholder: x_fetched, self.Y_placeholder: y_fetched}
                b = len(x_fetched)
                n += b
                l, acc = self.ses.run([self.loss, self.accuracy], feed_dict=feed_dict)
                loss += l * b
                accuracy += acc * b

            except tf.errors.OutOfRangeError:
                self.ses.run(initializer)
                break

        loss /= n
        accuracy /= n
        return loss, accuracy

    def train(self, code_summary=True):
        self.init_variables()
        if code_summary:
            self.add_code_summary()

        self.ses.run(self.dataloader.train_initializer)

        for epoch in range(FLAGS.epochs):
            while True:
                try:
                    _, scalar_summaries, step = self.ses.run(
                        [self.train_operation, self.scalar_summaries, self.global_step])
                    self.train_summary_writer.add_summary(scalar_summaries, step)

                except tf.errors.OutOfRangeError:
                    self.ses.run(self.dataloader.train_initializer)
                    break
            train_loss, train_accuracy = self.get_loss_accuracy('train')
            validation_loss, validation_accuracy = self.get_loss_accuracy('validation')
            print("Epoch = {}\n\tTrain Loss = {}\n\tTrain Accuracy = {}".format(epoch, train_loss, train_accuracy))
            print("Epoch = {}\n\tValidation Loss = {}\n\tValidation Accuracy = {}".format(epoch, validation_loss,
                                                                                          validation_accuracy))
            train_summary = tf.Summary(value=[
                tf.Summary.Value(tag="train_val_loss", simple_value=train_loss),
                tf.Summary.Value(tag="train_val_accuracy", simple_value=train_accuracy),
            ])
            self.train_summary_writer.add_summary(train_summary, epoch)

            validation_summary = tf.Summary(value=[
                tf.Summary.Value(tag="train_val_loss", simple_value=validation_loss),
                tf.Summary.Value(tag="train_val_accuracy", simple_value=validation_accuracy),
            ])
            self.validation_summary_writer.add_summary(validation_summary, epoch)

    def get_n_params(self):
        total_parameters = 0
        for variable in tf.trainable_variables():
            shape = variable.get_shape()
            variable_parameters = 1
            for dim in shape:
                variable_parameters *= dim.value
            total_parameters += variable_parameters
        print("Number of Parameters in model = {}".format(total_parameters))


if __name__ == "__main__":
    model = CNN("CNN_Test", summary=False, filters_list=[16, 32, 64, 128, 256],
                strides_list=[[2, 2], [2, 2], [1, 1], [1, 1], [1, 1]],
                kernel_size_list=[[5, 5], [3, 3], [3, 3], [3, 3], [2, 2]],
                padding_list=['v', 'v', 'v', 'v', 'v'])
    # model = CNN("CNN_Test", summary=True, filters_list=[20, 40], strides_list=[[2, 2], [2, 2]])
    model.get_n_params()
    # model.train()
    # model.save()

    model.load()
    test_loss, test_accuracy = model.get_loss_accuracy('test')
    print("Epoch = {}\n\tTest Loss = {}\n\tTest Accuracy = {}".format(1, test_loss, test_accuracy))
