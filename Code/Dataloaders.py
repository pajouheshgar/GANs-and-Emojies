import os
import imageio
import numpy as np
from PIL import Image
from scipy import misc
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

flags = tf.app.flags
FLAGS = flags.FLAGS

flags.DEFINE_string('dataset_dir', "../Dataset/", 'Directory of dataset')
flags.DEFINE_integer('batch_size', 100, 'Number of instances in each batch')
flags.DEFINE_integer('max_deg', 15, 'Maximum angle of rotation in degree')
flags.DEFINE_integer('shuffle_buffer_size', 4096, 'Shuffle buffer size for dataloader')
flags.DEFINE_integer('num_classes', 10, 'Number of classes in dataset')
flags.DEFINE_integer('image_height', 64, 'Height of images')
flags.DEFINE_integer('image_width', 64, 'Width of images')


class classification_dataloader():
    DIRECTORIES_TO_INCLUDE = ['apple', 'facebook', 'google', 'twitter', 'messenger']

    def __init__(self, directories_to_include=DIRECTORIES_TO_INCLUDE):
        self.directories_to_include = directories_to_include
        self.files_name_list = []
        for directory in self.directories_to_include:
            images_in_directory = list(filter(lambda s: ".png" in s,
                                              os.listdir(FLAGS.dataset_dir + "emoji-images/" + directory + "/")))
            labels = [directory for _ in images_in_directory]
            self.files_name_list += (list(zip(images_in_directory, labels)))

        train_dataset = self._create_dataset(self._generator)
        train_iterator = train_dataset.make_initializable_iterator()
        self.train_initializer = train_iterator.initializer
        self.train_batch = train_iterator.get_next()

    def _generator(self):
        for image_file_name, label in self.files_name_list:
            emoji_image = Image.open(FLAGS.dataset_dir + "emoji-images/{}/{}".format(label, image_file_name)).convert(
                "RGBA")
            emoji_image = emoji_image.resize([FLAGS.image_height, FLAGS.image_width], Image.ANTIALIAS)
            yield emoji_image, self.directories_to_include.index(label)

    def _create_dataset(self, generator):
        output_shapes = (tf.TensorShape([FLAGS.image_height, FLAGS.image_width, 4]), tf.TensorShape([]))
        dataset = tf.data.Dataset.from_generator(generator=generator,
                                                 output_types=(tf.float32, tf.int32),
                                                 output_shapes=output_shapes
                                                 )

        # dataset = dataset.shuffle(buffer_size=FLAGS.shuffle_buffer_size, seed=42).repeat(1).batch(
        dataset = dataset.batch(FLAGS.batch_size)
        return dataset


if __name__ == "__main__":
    dataloader = classification_dataloader(directories_to_include=['apple'])
    x_train, y_train = dataloader.train_batch
    ses = tf.InteractiveSession()
    ses.run(dataloader.train_initializer)

    a = ses.run(x_train)
    print(a.shape)
    # plt.imshow(np.array(a[0], dtype=np.uint8))
    plt.imshow(a[0] / 255)
    print(a[0].mean())
    plt.show()

    # dataloader = noisy_classification_dataloader(flatten=False)
    # x_test, y_test = dataloader.test_batch
    # x_validation, y_validation = dataloader.validation_batch
    # i = 0
    #
    # while True:
    #     try:
    #         a, b = ses.run([x_train, y_train])
    #         print(a.shape)
    #         plt.figure()
    #         plt.imshow(a[0, :, :, 0])
    #         i += 1
    #         if i == 10:
    #             break
    #
    #     except tf.errors.OutOfRangeError:
    #         print("reinitializing dataset")
    #         ses.run(dataloader.train_initializer)
    #         i = 0
    #
    # plt.show()
