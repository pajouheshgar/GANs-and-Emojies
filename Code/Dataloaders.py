import os
import json
import numpy as np
from PIL import Image
from random import seed
import tensorflow as tf
from random import shuffle
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

flags = tf.app.flags
FLAGS = flags.FLAGS

flags.DEFINE_string('dataset_dir', "../Dataset/", 'Directory of dataset')
flags.DEFINE_integer('batch_size', 32, 'Number of instances in each batch')
flags.DEFINE_integer('shuffle_buffer_size', 512, 'Shuffle buffer size for dataloader')
flags.DEFINE_integer('image_height', 64, 'Height of images')
flags.DEFINE_integer('image_width', 64, 'Width of images')
# flags.DEFINE_integer('image_channels', 4, 'Width of images')
flags.DEFINE_integer('image_channels', 3, 'Width of images')


class GAN_Dataloader():
    COMPANIES = ['apple', 'facebook', 'google', 'twitter', 'messenger']
    CATEGORIES = ['Food & Drink', 'Smileys & People', 'Symbols', 'Travel & Places', 'Skin Tones', 'Activities',
                  'Objects', 'Animals & Nature', 'Flags']
    company2id = {c: i for i, c in enumerate(COMPANIES)}
    id2company = {i: c for i, c in enumerate(COMPANIES)}
    category2id = {c: i for i, c in enumerate(CATEGORIES)}
    id2category = {i: c for i, c in enumerate(CATEGORIES)}

    COMPANIES_TO_INCLUDE = ['apple', 'facebook', 'google', 'twitter', 'messenger']
    CATEGORIES_TO_INCLUDE = ['Smileys & People']

    def __init__(self, companies_to_include=COMPANIES_TO_INCLUDE):
        self.companies_to_include = companies_to_include
        self.files_name_list = []

        for directory in self.companies_to_include:
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
            emoji_image = np.array(emoji_image.resize([FLAGS.image_height, FLAGS.image_width], Image.BICUBIC)) / 255.0
            yield emoji_image, self.company2id[label]

    def _create_dataset(self, generator):
        output_shapes = (tf.TensorShape([FLAGS.image_height, FLAGS.image_width, 4]), tf.TensorShape([]))
        dataset = tf.data.Dataset.from_generator(generator=generator,
                                                 output_types=(tf.float32, tf.int64),
                                                 output_shapes=output_shapes
                                                 )

        dataset = dataset.shuffle(buffer_size=FLAGS.shuffle_buffer_size, seed=42)
        dataset = dataset.batch(FLAGS.batch_size)
        return dataset


class Conditional_GAN_Dataloader():
    COMPANIES = ['apple', 'facebook', 'google', 'twitter', 'messenger']
    CATEGORIES = ['Food & Drink', 'Smileys & People', 'Symbols', 'Travel & Places', 'Skin Tones', 'Activities',
                  'Objects', 'Animals & Nature', 'Flags']

    COMPANIES_TO_INCLUDE = ['apple', 'facebook', 'google', 'twitter', 'messenger']
    CATEGORIES_TO_INCLUDE = ['Smileys & People']

    def __init__(self, companies_to_include=COMPANIES_TO_INCLUDE, categories_to_include=CATEGORIES_TO_INCLUDE):
        seed(42)
        self.companies_to_include = companies_to_include
        self.categories_to_include = categories_to_include
        self.category_flag = True
        if categories_to_include is None:
            self.category_flag = False

        self.company2id = {c: i for i, c in enumerate(self.companies_to_include)}
        self.id2company = {i: c for i, c in enumerate(self.companies_to_include)}
        if self.category_flag:
            self.category2id = {c: i for i, c in enumerate(self.categories_to_include)}
            self.id2category = {i: c for i, c in enumerate(self.categories_to_include)}
        self.files_list = []

        with open(FLAGS.dataset_dir + "emoji_pretty.json") as f:
            self.additional_data = json.load(f)
        self.img_name_2_info = {d['image']: d for d in self.additional_data}

        category_list = []
        for directory in self.companies_to_include:
            images_in_directory = list(
                filter(lambda s: (".png" in s)
                                 and ((not self.category_flag) or (s in self.img_name_2_info))
                                 and (not self.category_flag or (
                        self.img_name_2_info[s]['category'] in self.categories_to_include)),
                       os.listdir(FLAGS.dataset_dir + "emoji-images/" + directory + "/")))
            labels = [directory for _ in images_in_directory]
            self.files_list += (list(zip(images_in_directory, labels)))

        train_dataset = self._create_dataset(self._generator)
        train_iterator = train_dataset.make_initializable_iterator()
        self.train_initializer = train_iterator.initializer
        self.train_batch = train_iterator.get_next()

    def _generator(self):
        for image_file_name, label in self.files_list:
            emoji_image = Image.open(FLAGS.dataset_dir + "emoji-images/{}/{}".format(label, image_file_name)).convert(
                "RGBA")
            emoji_image = np.array(emoji_image.resize([FLAGS.image_height, FLAGS.image_width], Image.BICUBIC)) / 255.0
            if FLAGS.image_channels == 3:
                emoji_image[:, :, :3] = emoji_image[:, :, :3] * emoji_image[:, :, 3:]
                emoji_image = emoji_image[:, :, :FLAGS.image_channels]

            if FLAGS.image_channels == 1:
                emoji_image = emoji_image[:, :, 3:]


            company_id = self.company2id[label]
            if self.category_flag:
                category_id = self.category2id[self.img_name_2_info[image_file_name]['category']]
            else:
                category_id = 0
            yield emoji_image, company_id, category_id

    def _create_dataset(self, generator):
        output_shapes = (
            tf.TensorShape([FLAGS.image_height, FLAGS.image_width, FLAGS.image_channels]), tf.TensorShape([]), tf.TensorShape([]))
        dataset = tf.data.Dataset.from_generator(generator=generator,
                                                 output_types=(tf.float32, tf.int64, tf.int64),
                                                 output_shapes=output_shapes
                                                 )

        dataset = dataset.shuffle(buffer_size=FLAGS.shuffle_buffer_size, seed=42)
        dataset = dataset.batch(FLAGS.batch_size)
        return dataset


class Classification_Dataloader():
    COMPANIES = ['apple', 'facebook', 'google', 'twitter', 'messenger']
    CATEGORIES = ['Food & Drink', 'Smileys & People', 'Symbols', 'Travel & Places', 'Skin Tones', 'Activities',
                  'Objects', 'Animals & Nature', 'Flags']

    COMPANIES_TO_INCLUDE = ['apple', 'facebook', 'google', 'twitter', 'messenger']
    # CATEGORIES_TO_INCLUDE = ['Smileys & People']
    CATEGORIES_TO_INCLUDE = ['Food & Drink', 'Smileys & People', 'Symbols', 'Travel & Places', 'Skin Tones',
                             'Activities',
                             'Objects', 'Animals & Nature', 'Flags']

    def __init__(self, companies_to_include=COMPANIES_TO_INCLUDE, categories_to_include=CATEGORIES_TO_INCLUDE):
        seed(42)
        self.companies_to_include = companies_to_include
        self.categories_to_include = categories_to_include
        self.category_flag = True
        if categories_to_include is None:
            self.category_flag = False

        self.company2id = {c: i for i, c in enumerate(self.companies_to_include)}
        self.id2company = {i: c for i, c in enumerate(self.companies_to_include)}
        if self.category_flag:
            self.category2id = {c: i for i, c in enumerate(self.categories_to_include)}
            self.id2category = {i: c for i, c in enumerate(self.categories_to_include)}
        self.files_list = []

        with open(FLAGS.dataset_dir + "emoji_pretty.json") as f:
            self.additional_data = json.load(f)
        self.img_name_2_info = {d['image']: d for d in self.additional_data}

        category_list = []
        for directory in self.companies_to_include:
            images_in_directory = list(
                filter(lambda s: (".png" in s)
                                 and ((not self.category_flag) or (s in self.img_name_2_info))
                                 and (not self.category_flag or (
                        self.img_name_2_info[s]['category'] in self.categories_to_include)),
                       os.listdir(FLAGS.dataset_dir + "emoji-images/" + directory + "/")))
            labels = [directory for _ in images_in_directory]
            self.files_list += (list(zip(images_in_directory, labels)))

        for img, company in self.files_list:
            category_list.append(self.img_name_2_info[img]['category'])

        n = len(self.files_list)
        if self.category_flag:
            self.train_files_list, self.test_files_list, self.train_category_list, self.test_category_list = train_test_split(
                self.files_list,
                category_list,
                train_size=int(n * 0.8),
                random_state=42,
                shuffle=True,
                stratify=category_list)
            self.train_files_list, self.validation_files_list, self.train_category_list, self.validation_category_list = train_test_split(
                self.train_files_list,
                self.train_category_list,
                train_size=int(n * 0.6),
                random_state=42,
                shuffle=True,
                stratify=self.train_category_list)

        if not self.category_flag:
            shuffle(self.files_list)
            self.train_files_list = self.files_list[n // 5:]
            self.test_files_list = self.files_list[:n // 5]

            self.n_train = len(self.train_files_list)
            self.validation_files_list = self.train_files_list[:self.n_train // 5]

            self.train_files_list = self.train_files_list[self.n_train // 5:]

        self.n_test = len(self.test_files_list)
        self.n_validation = len(self.validation_files_list)
        self.n_train = len(self.train_files_list)

        train_dataset = self._create_dataset(self._train_generator)
        train_iterator = train_dataset.make_initializable_iterator()
        self.train_initializer = train_iterator.initializer
        self.train_batch = train_iterator.get_next()

        test_dataset = self._create_dataset(self._test_generator)
        test_iterator = test_dataset.make_initializable_iterator()
        self.test_initializer = test_iterator.initializer
        self.test_batch = test_iterator.get_next()

        validation_dataset = self._create_dataset(self._validation_generator)
        validation_iterator = validation_dataset.make_initializable_iterator()
        self.validation_initializer = validation_iterator.initializer
        self.validation_batch = validation_iterator.get_next()

        self.n_companies = len(self.companies_to_include)
        self.n_categories = len(self.categories_to_include) if self.category_flag else 0

    def _generator(self, files_list):
        for image_file_name, label in files_list:
            emoji_image = Image.open(FLAGS.dataset_dir + "emoji-images/{}/{}".format(label, image_file_name)).convert(
                "RGBA")
            emoji_image = np.array(emoji_image.resize([FLAGS.image_height, FLAGS.image_width], Image.BICUBIC)) / 255.0
            if FLAGS.image_channels == 3:
                emoji_image[:, :, :3] = emoji_image[:, :, :3] * emoji_image[:, :, 3:]
                emoji_image = emoji_image[:, :, :FLAGS.image_channels]

            if FLAGS.image_channels == 1:
                emoji_image = emoji_image[:, :, 3:]

            company_id = self.company2id[label]
            if self.category_flag:
                category_id = self.category2id[self.img_name_2_info[image_file_name]['category']]
            else:
                category_id = 0
            yield emoji_image, company_id, category_id

    def _train_generator(self):
        return self._generator(self.train_files_list)

    def _test_generator(self):
        return self._generator(self.test_files_list)

    def _validation_generator(self):
        return self._generator(self.validation_files_list)

    def _create_dataset(self, generator):
        output_shapes = (
            tf.TensorShape([FLAGS.image_height, FLAGS.image_width, FLAGS.image_channels]), tf.TensorShape([]),
            tf.TensorShape([]))
        dataset = tf.data.Dataset.from_generator(generator=generator,
                                                 output_types=(tf.float32, tf.int64, tf.int64),
                                                 output_shapes=output_shapes
                                                 )

        dataset = dataset.shuffle(buffer_size=FLAGS.shuffle_buffer_size, seed=42)
        dataset = dataset.batch(FLAGS.batch_size)
        return dataset


# if __name__ == "__main__":
#     dataloader = GAN_Dataloader(companies_to_include=['apple'])
#     x_train, y_train = dataloader.train_batch
#     ses = tf.InteractiveSession()
#     ses.run(dataloader.train_initializer)
#
#     a = ses.run(x_train)
#     print(a.shape)
#     # plt.imshow(np.array(a[0], dtype=np.uint8))
#     plt.imshow(a[0] / 255)
#     print(a[0].mean())
#     plt.show()
#
#     # dataloader = noisy_classification_dataloader(flatten=False)
#     # x_test, y_test = dataloader.test_batch
#     # x_validation, y_validation = dataloader.validation_batch
#     # i = 0
#     #
#     # while True:
#     #     try:
#     #         a, b = ses.run([x_train, y_train])
#     #         print(a.shape)
#     #         plt.figure()
#     #         plt.imshow(a[0, :, :, 0])
#     #         i += 1
#     #         if i == 10:
#     #             break
#     #
#     #     except tf.errors.OutOfRangeError:
#     #         print("reinitializing dataset")
#     #         ses.run(dataloader.train_initializer)
#     #         i = 0
#     #
#     # plt.show()

# if __name__ == "__main__":
#     dataloader = Conditional_GAN_Dataloader(companies_to_include=['apple', 'google'],
#                                             categories_to_include=['Smileys & People', 'Symbols'])
#     x_train, y_train, z_train = dataloader.train_batch
#     ses = tf.InteractiveSession()
#     ses.run(dataloader.train_initializer)
#     ses.run(dataloader.train_initializer)
#
#     a, b, c = ses.run([x_train, y_train, z_train])
#     a, b, c = ses.run([x_train, y_train, z_train])
#     i = 1
#     print(b, c)
#     plt.imshow(a[i])
#     print(b[i], c[i])
#     plt.show()
#
#     # dataloader = noisy_classification_dataloader(flatten=False)
#     # x_test, y_test = dataloader.test_batch
#     # x_validation, y_validation = dataloader.validation_batch
#     # i = 0
#     #
#     # while True:
#     #     try:
#     #         a, b = ses.run([x_train, y_train])
#     #         print(a.shape)
#     #         plt.figure()
#     #         plt.imshow(a[0, :, :, 0])
#     #         i += 1
#     #         if i == 10:
#     #             break
#     #
#     #     except tf.errors.OutOfRangeError:
#     #         print("reinitializing dataset")
#     #         ses.run(dataloader.train_initializer)
#     #         i = 0
#     #
#     # plt.show()

if __name__ == "__main__":
    dataloader = Conditional_GAN_Dataloader(categories_to_include=None)
    x_train, y_train, z_train = dataloader.train_batch
    ses = tf.InteractiveSession()
    ses.run(dataloader.train_initializer)
    ses.run(dataloader.train_initializer)

    a, b, c = ses.run([x_train, y_train, z_train])
    a, b, c = ses.run([x_train, y_train, z_train])
    i = 31
    print(b, c)
    plt.imshow(a[i][:, :, -1])
    # plt.subplot(131)
    # plt.imshow(a[i][:, :, :3] * a[i][:, :, 3:])
    # plt.subplot(132)
    # plt.imshow(a[i][:, :, :4])
    # plt.subplot(133)
    # plt.imshow(a[i][:, :, :3])

    # plt.figure()
    # plt.imshow(a[i][:, :, 3], cmap='gray')
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
