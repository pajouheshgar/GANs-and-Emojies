import os
import json

from collections import defaultdict
import tensorflow as tf

flags = tf.app.flags
FLAGS = flags.FLAGS
flags.DEFINE_string('dataset_dir', "../Dataset/", 'Directory of dataset')

with open(FLAGS.dataset_dir + "emoji_pretty.json") as f:
    data = json.load(f)

COMPANIES_TO_INCLUDE = ['apple', 'facebook', 'google', 'twitter', 'messenger']
CATEGORIES_TO_INCLUDE = ['Food & Drink', 'Smileys & People', 'Symbols', 'Travel & Places', 'Skin Tones', 'Activities',
                         'Objects', 'Animals & Nature', 'Flags']



print(data[0])
print(len(data))
files_name_list = []
img_name_2_inf = {d['image']: d for d in data}


for directory in COMPANIES_TO_INCLUDE:
    images_in_directory = list(filter(lambda s: ".png" in s,
                                      os.listdir(FLAGS.dataset_dir + "emoji-images/" + directory + "/")))
    # for img in images_in_directory:
    #     if img not in img_name_2_inf:
    #         print(directory + "/" + img)
    labels = [directory for _ in images_in_directory]
    files_name_list += (list(zip(images_in_directory, labels)))

print(files_name_list)
