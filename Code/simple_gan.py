import tensorflow as tf

import Code.GANBlocks as GANBlocks
from Code.Dataloaders import GAN_Dataloader, Conditional_GAN_Dataloader

flags = tf.app.flags
FLAGS = flags.FLAGS

spgen = GANBlocks.simple_generator
spdis = GANBlocks.simple_discriminator
dcgen = GANBlocks.DCGAN_generator
dcdis = GANBlocks.DCGAN_discriminator
bedis = GANBlocks.BEGAN_discriminator

all_gan_types = ["GAN", "DCGAN", "BEGAN", "DRAGAN", "MIXGAN"]
gan_type = "MIXGAN"

# model_name = "all_emojies"
save_every = 10
z_len = 250
image_size = [FLAGS.image_width, FLAGS.image_width]

train_config_init = {"batch_size": FLAGS.batch_size,
                     "num_steps": 30000,
                     "z_sd": 1,
                     "model_name": None,
                     "save_every": save_every}

config = {"batch_size": FLAGS.batch_size,
          "beta1": 0.5, "beta2": 0.99, "lambda": 0.001, "gamma": 0.75, "dis_iters": 5,
          'clamp_lower': -0.01, 'clamp_upper': 0.01, 'lambda_gp': 6
          }
train_config = {"learning_rate": [0.001, 0.9, 1000]}
train_config.update(train_config_init)
# test_optims, test_fd = GANBlocks.testGAN(dcgen, dcdis, bedis, config, z_len=z_len,
#                                          image_shape=image_size + [FLAGS.image_channels],
#                                          minimax=False)

test_optims, test_fd = GANBlocks.WGAN(dcgen, dcdis, config, z_len=z_len,
                                      image_shape=image_size + [FLAGS.image_channels],
                                      mode='gp')

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.Session(config=config)
sess.run(tf.global_variables_initializer())

directories_to_include = ['apple', 'facebook', 'google', 'twitter', 'messenger']
# data_loader = GAN_Dataloader(directories_to_include)
# data_loader = Conditional_GAN_Dataloader(directories_to_include, categories_to_include=None)
# data_loader = Conditional_GAN_Dataloader(directories_to_include, categories_to_include=['Flags'])
# data_loader = Conditional_GAN_Dataloader(directories_to_include, categories_to_include=['Animals & Nature'])
# data_loader = Conditional_GAN_Dataloader(directories_to_include, categories_to_include=['Symbols'])
data_loader = Conditional_GAN_Dataloader(directories_to_include, categories_to_include=['Smileys & People'])
print("TRAINING MODEL")
GANBlocks.train_wgan_dataloader(sess, test_optims, test_fd, data_loader, train_config)
