import os
import datetime
import numpy as np
import tensorflow as tf
from Code.ops import *

flags = tf.app.flags
FLAGS = flags.FLAGS
flags.DEFINE_float('ilr', 0.001, 'initial_learning_rate')
flags.DEFINE_integer('decay_steps', 10000, 'steps to halve the learning rate')
flags.DEFINE_integer('epochs', 5000, 'Number of epochs to train.')

flags.DEFINE_float('z_std', 1.0, 'Standard deviation of Z')
flags.DEFINE_float('beta1', 0.5, 'beta1 of Adam optimizer')
flags.DEFINE_float('beta2', 0.99, 'beta2 of Adam optimizer')

flags.DEFINE_integer('z_dim', 256, 'Dimension of z')
flags.DEFINE_integer('n_params', 64, 'Number of parameters for Conv and Deconv layers')
flags.DEFINE_integer('dis_steps', 1, 'Number of steps to train discriminator')
flags.DEFINE_bool('use_batch_norm', False, 'Whether to use Batch Normalization or not')
flags.DEFINE_integer('kernel_size', 5, 'Kernel size for Convolution and  Deconvolution layers')

flags.DEFINE_integer('image_summary_max', 5, 'Maximum images to show in tensorboard')

from Code.Dataloaders import Conditional_GAN_Dataloader, Parallel_Conditional_GAN_Dataloader


class GAN():
    NAME = "GAN"
    SAVE_DIR = "../Dataset/Models/"
    FILTERS_LIST = [16, 32, 64]
    STRIDES_LIST = [[2, 2], [2, 2], [2, 2]]
    KERNEL_SIZE_LIST = [[5, 5], [5, 5], [5, 5]]
    PADDING_LIST = ['s', 's', 's', 'v', 'v']
    DESCRIPTION = ""

    def __init__(self, dataloader, name=NAME, save_dir=SAVE_DIR, summary=True, ):
        self.NAME = name
        self.SAVE_DIR = save_dir
        self.MODEL_PATH = os.path.join(self.SAVE_DIR, self.NAME)
        try:
            os.mkdir(self.MODEL_PATH)
        except:
            pass
        init_time = datetime.datetime.utcnow().strftime("%Y%m%d%H%M%S")
        self.LOG_PATH = os.path.join(self.SAVE_DIR, "log/" + self.NAME + "-run-" + init_time)

        self.dataloader = dataloader
        self.build_graph(summary)
        self.ses = tf.InteractiveSession()

    def DCGAN_generator(self, z_img, word_vector=None, name="generator", use_batch_norm=True, reuse=False,
                        is_training=False, nparams=64,
                        kernel_size=[5, 5]):
        activation = tf.nn.relu
        output_shape = [FLAGS.image_width, FLAGS.image_height, FLAGS.image_channels]
        n_channels = output_shape[-1]
        s_h, s_w = output_shape[0], output_shape[1]
        s_h2, s_w2 = (s_h + 1) // 2, (s_w + 1) // 2
        s_h4, s_w4 = (s_h2 + 1) // 2, (s_w2 + 1) // 2
        s_h8, s_w8 = (s_h4 + 1) // 2, (s_w4 + 1) // 2
        s_h16, s_w16 = (s_h8 + 1) // 2, (s_w8 + 1) // 2

        with tf.variable_scope(name, reuse=reuse):
            net = tf.layers.dense(z_img, units=nparams * 8 * s_h16 * s_w16, name="fc1")
            net = tf.reshape(net, shape=[-1, s_h16, s_w16, nparams * 8], name='reshape')
            deconv = tf.layers.conv2d_transpose(
                inputs=net,
                kernel_size=kernel_size,
                filters=nparams * 4,
                strides=[2, 2],
                padding="SAME",
                activation=activation,
                name='dc1',
            )
            net = tf.layers.batch_normalization(
                inputs=deconv,
                training=is_training,
                name='bn1'
            ) if use_batch_norm else deconv

            deconv = tf.layers.conv2d_transpose(
                inputs=net,
                kernel_size=kernel_size,
                filters=nparams * 2,
                strides=[2, 2],
                padding="SAME",
                activation=activation,
                name='dc2',
            )
            net = tf.layers.batch_normalization(
                inputs=deconv,
                training=is_training,
                name='bn2'
            ) if use_batch_norm else deconv

            deconv = tf.layers.conv2d_transpose(
                inputs=net,
                kernel_size=kernel_size,
                filters=nparams * 1,
                strides=[2, 2],
                padding="SAME",
                activation=activation,
                name='dc3',
            )
            net = tf.layers.batch_normalization(
                inputs=deconv,
                training=is_training,
                name='bn3'
            ) if use_batch_norm else deconv

            deconv = tf.layers.conv2d_transpose(
                inputs=net,
                kernel_size=kernel_size,
                filters=n_channels,
                strides=[2, 2],
                padding="SAME",
                # activation=activation,
                name='dc4',
            )

            generated_image = tf.nn.sigmoid(deconv)

        return generated_image

    def DCGAN_discriminator(self, img, use_batch_norm=True, name="discriminator", reuse=False,
                            is_training=False,
                            nparams=64, kernel_size=[5, 5]):

        activation = tf.nn.relu
        with tf.variable_scope(name, reuse=reuse):
            conv = tf.layers.conv2d_transpose(
                inputs=img,
                kernel_size=kernel_size,
                filters=nparams * 1,
                strides=[2, 2],
                padding="SAME",
                activation=activation,
                name='cn1',
            )
            net = tf.layers.batch_normalization(
                inputs=conv,
                training=is_training,
                name='bn1'
            ) if use_batch_norm else conv

            conv = tf.layers.conv2d_transpose(
                inputs=net,
                kernel_size=kernel_size,
                filters=nparams * 2,
                strides=[2, 2],
                padding="SAME",
                activation=activation,
                name='cn2',
            )
            net = tf.layers.batch_normalization(
                inputs=conv,
                training=is_training,
                name='bn2'
            ) if use_batch_norm else conv

            conv = tf.layers.conv2d_transpose(
                inputs=net,
                kernel_size=kernel_size,
                filters=nparams * 4,
                strides=[2, 2],
                padding="SAME",
                activation=activation,
                name='cn3',
            )
            net = tf.layers.batch_normalization(
                inputs=conv,
                training=is_training,
                name='bn3'
            ) if use_batch_norm else conv

            conv = tf.layers.conv2d_transpose(
                inputs=net,
                kernel_size=kernel_size,
                filters=nparams * 8,
                strides=[2, 2],
                padding="SAME",
                activation=activation,
                name='cn4',
            )
            net = tf.layers.batch_normalization(
                inputs=conv,
                training=is_training,
                name='bn4'
            ) if use_batch_norm else conv

            net = tf.layers.flatten(net, name='flatten')
            logits = tf.layers.dense(net, units=1, name='logits')
            probs = tf.nn.sigmoid(logits, name='probs')

        return probs, logits

    def build_graph(self, summary):
        tf.set_random_seed(42)
        np.random.seed(42)
        use_batch_norm = FLAGS.use_batch_norm
        kernel_size = [FLAGS.kernel_size, FLAGS.kernel_size]
        n_params = FLAGS.n_params
        with tf.variable_scope(self.NAME):
            with tf.name_scope("Dataset"):
                self.X_train, self.Y_train_company, self.Y_train_category, self.X_train_wordvector = self.dataloader.train_batch

            with tf.name_scope("Placeholders"):
                self.Z = tf.random_normal(shape=[FLAGS.batch_size, FLAGS.z_dim], mean=0.0, stddev=FLAGS.z_std,
                                          dtype=tf.float32, name='Z')
                self.X_placeholder = tf.placeholder_with_default(self.X_train,
                                                                 shape=[None, FLAGS.image_height, FLAGS.image_width,
                                                                        FLAGS.image_channels],
                                                                 name="X_placeholder")
                self.Y_category_placeholder = tf.placeholder_with_default(self.Y_train_category,
                                                                          shape=[None],
                                                                          name="Y_category_placeholder")
                self.Y_company_placeholder = tf.placeholder_with_default(self.Y_train_company,
                                                                         shape=[None],
                                                                         name="Y_company_placeholder")
                self.X_wordvector_placeholder = tf.placeholder_with_default(self.X_train_wordvector,
                                                                            shape=[None, FLAGS.word2vec_embedding_size],
                                                                            name='X_wordvector_placeholder')
                self.Z_placeholder = tf.placeholder_with_default(self.Z, shape=[None, FLAGS.z_dim],
                                                                 name='Z_placeholder')

                self.is_training_placeholder = tf.placeholder_with_default(True, [], name='is_training_placeholder')

            with tf.variable_scope("Inference"):
                # generator_seed = tf.concat([self.Z_placeholder, self.X_wordvector_placeholder], axis=1,
                #                            name='generator_seed')
                generator_seed = self.Z_placeholder
                self.generated_images = self.DCGAN_generator(
                    z_img=generator_seed,
                    reuse=False,
                    use_batch_norm=use_batch_norm,
                    kernel_size=kernel_size,
                    nparams=n_params,
                    is_training=self.is_training_placeholder,
                    name='generator',
                )

                self.generated_images_placeholder = tf.placeholder_with_default(self.generated_images,
                                                                                shape=[None, FLAGS.image_width,
                                                                                       FLAGS.image_height,
                                                                                       FLAGS.image_channels],
                                                                                name='generated_images_placeholder')
                self.fake_dis_prob, self.fake_dis_logits = self.DCGAN_discriminator(
                    img=self.generated_images_placeholder,
                    reuse=False,
                    use_batch_norm=use_batch_norm,
                    nparams=n_params,
                    kernel_size=kernel_size,
                    is_training=self.is_training_placeholder,
                    name="discriminator")

                self.real_dis_prob, self.real_dis_logits = self.DCGAN_discriminator(
                    img=self.X_placeholder,
                    reuse=True,
                    use_batch_norm=use_batch_norm,
                    nparams=n_params,
                    kernel_size=kernel_size,
                    is_training=self.is_training_placeholder,
                    name="discriminator",
                )

            with tf.name_scope("Optimization"):
                self.update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
                with tf.control_dependencies(self.update_ops):
                    self.global_step = tf.train.get_or_create_global_step()
                    learning_rate = tf.train.exponential_decay(FLAGS.ilr, self.global_step, FLAGS.decay_steps,
                                                               0.5,
                                                               name='learning_rate')

                    self.dis_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,
                                                      scope="{}/{}/{}".format(self.NAME, "Inference", "discriminator"))
                    self.gen_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,
                                                      scope="{}/{}/{}".format(self.NAME, "Inference", "generator"))

                    self.real_dis_loss = tf.reduce_mean(
                        tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.ones_like(self.real_dis_logits),
                                                                logits=self.real_dis_logits), name='real_dis_loss')
                    self.fake_dis_loss = tf.reduce_mean(
                        tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.zeros_like(self.fake_dis_logits),
                                                                logits=self.fake_dis_logits), name='fake_dis_loss')
                    self.dis_loss = self.real_dis_loss + self.fake_dis_loss
                    self.gen_loss = -self.fake_dis_loss

                    gen_optimizer = tf.train.AdamOptimizer(learning_rate, beta1=FLAGS.beta1, beta2=FLAGS.beta2)
                    dis_optimizer = tf.train.AdamOptimizer(learning_rate, beta1=FLAGS.beta1, beta2=FLAGS.beta2)
                    self.dis_train_operation = dis_optimizer.minimize(self.dis_loss, var_list=self.dis_vars)
                    self.gen_train_operation = gen_optimizer.minimize(self.gen_loss, global_step=self.global_step,
                                                                      var_list=self.gen_vars)
                    self.train_operation = [self.gen_train_operation] + [self.dis_train_operation for _ in
                                                                         range(FLAGS.dis_steps)]

            self.init_node = tf.global_variables_initializer()
            self.save_node = tf.train.Saver()

            if summary:
                with tf.name_scope("Summary"):
                    self.gen_loss_summary = tf.summary.scalar(name="gen_loss", tensor=self.gen_loss)
                    self.dis_loss_summary = tf.summary.scalar(name="dis_loss", tensor=self.dis_loss)
                    self.scalar_summaries = tf.summary.merge(
                        [
                            self.gen_loss_summary,
                            self.dis_loss_summary,
                        ])
                    self.image_summary = tf.summary.image(name='generated images',
                                                          tensor=self.generated_images_placeholder,
                                                          max_outputs=FLAGS.image_summary_max)
                    self.merged_summaries = tf.summary.merge_all()
                    self.summary_writer = tf.summary.FileWriter(self.LOG_PATH,
                                                                tf.get_default_graph())

    def add_code_summary(self):
        code_string = "\n".join(open(os.path.basename(__file__), 'r').readlines())
        text_tensor = tf.make_tensor_proto(self.DESCRIPTION + "\n\n" + code_string, dtype=tf.string)
        meta = tf.SummaryMetadata()
        meta.plugin_data.plugin_name = "text"
        summary = tf.Summary()
        summary.value.add(tag="Hyper parameters", metadata=meta, tensor=text_tensor)
        self.summary_writer.add_summary(summary)

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
            x, y_company, y_category = self.X_train, self.Y_train_company, self.Y_train_category
        elif type == 'test':
            initializer = self.dataloader.test_initializer
            x, y_company, y_category = self.X_test, self.Y_test_company, self.Y_test_category
        else:
            initializer = self.dataloader.validation_initializer
            x, y_company, y_category = self.X_validation, self.Y_validation_company, self.Y_validation_category

        self.ses.run(initializer)
        loss = 0
        accuracy = 0
        n = 0
        while True:
            try:
                x_fetched, y_company_fetched, y_category_fetched = self.ses.run([x, y_company, y_category])
                feed_dict = {self.X_placeholder: x_fetched, self.Y_company_placeholder: y_company_fetched,
                             self.Y_category_placeholder: y_category_fetched}
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
                    for _ in range(FLAGS.dis_steps):
                        self.ses.run(self.dis_train_operation)

                    _, summaries, step = self.ses.run(
                        [self.gen_train_operation, self.merged_summaries, self.global_step])

                    if step % 10 == 0:
                        print(step)
                        self.summary_writer.add_summary(summaries, step)
                except tf.errors.OutOfRangeError:
                    self.ses.run(self.dataloader.train_initializer)
                    break

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
    dataloader = Parallel_Conditional_GAN_Dataloader(word2vec_flag=True)
    model = GAN(dataloader, "GAN", summary=True)
    print(model.update_ops)
    # model.init_variables()
    # model.train()
    # model.ses.close()

    # model = CNN_Classifier("CNN_Test2", summary=False, filters_list=[16, 32, 64, 128, 256],
    #                        strides_list=[[2, 2], [2, 2], [1, 1], [1, 1], [1, 1]],
    #                        kernel_size_list=[[5, 5], [3, 3], [3, 3], [3, 3], [2, 2]],
    #                        padding_list=['v', 'v', 'v', 'v', 'v'])
    # model.init_variables()
    # model.ses.close()
    # model.get_n_params()
    # model.train()
    # model.save()

    # model.load()
    # test_loss, test_accuracy = model.get_loss_accuracy('validation')
    # print(
    #     "Category = {}\n\tTest Loss = {}\n\tValidation Accuracy = {}".format(model.categories_to_include[0], test_loss,
    #                                                                          test_accuracy))
