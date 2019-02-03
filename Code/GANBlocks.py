import tensorflow as tf
import numpy as np
from Code.ops import *


######## GENERATORS
def simple_generator(z, output_shape=[64, 64, 3], name="SIMPLE_generator", reuse=False, is_training=False):
    with tf.variable_scope(name, reuse=reuse):
        net = linear(z, 1024, scope="g_fc1")
        net = batch_norm(net, is_training=is_training, scope="g_bn1")
        net = lrelu(net)
        net = linear(net, 128 * (output_shape[0] // 4) * (output_shape[1] // 4), scope="g_fc2")
        net = batch_norm(net, is_training=is_training, scope="g_bn2")
        net = lrelu(net)
        net = tf.reshape(net, [-1, output_shape[0] // 4, output_shape[1] // 4, 128])
        net = deconv2d(net, [-1, output_shape[0] // 2, output_shape[1] // 2, 64],
                       4, 4, 2, 2, name="g_dc3")
        net = batch_norm(net, is_training=is_training, scope="g_bn3")
        net = lrelu(net)
        net = deconv2d(net, [-1, output_shape[0], output_shape[1], output_shape[2]],
                       4, 4, 2, 2, name="g_dc4")
        out = tf.nn.sigmoid(net)

    info = {}
    info["variables"] = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=name)
    return out, info


def DCGAN_generator(z, output_shape=[64, 64, 3], name="DCGAN_generator", reuse=False, is_training=False, gf_dim=64):
    with tf.variable_scope(name, reuse=reuse):
        s_h, s_w = output_shape[0], output_shape[1]
        s_h2, s_w2 = int(np.ceil(float(s_h) / float(2))), int(np.ceil(float(s_w) / float(2)))
        s_h4, s_w4 = int(np.ceil(float(s_h2) / float(2))), int(np.ceil(float(s_w2) / float(2)))
        s_h8, s_w8 = int(np.ceil(float(s_h4) / float(2))), int(np.ceil(float(s_w4) / float(2)))
        s_h16, s_w16 = int(np.ceil(float(s_h8) / float(2))), int(np.ceil(float(s_w8) / float(2)))

        net = linear(z, gf_dim * 8 * s_h16 * s_w16, scope="g_fc1")
        net = tf.reshape(net, [-1, s_h16, s_w16, gf_dim * 8])
        output_c_dim = output_shape[-1]
        net = lrelu(batch_norm(net, is_training, scope="g_bn1"), leak=0.1)
        # net = tf.nn.relu(batch_norm(net, is_training, scope="g_bn1"))
        net = deconv2d(net, [-1, s_h8, s_w8, gf_dim * 4], 5, 5, 2, 2, name="g_dc1")
        net = lrelu(batch_norm(net, is_training, scope="g_bn2"), leak=0.1)
        # net = tf.nn.relu(batch_norm(net, is_training, scope="g_bn2"))
        net = deconv2d(net, [-1, s_h4, s_w4, gf_dim * 2], 5, 5, 2, 2, name="g_dc2")
        net = lrelu(batch_norm(net, is_training, scope="g_bn3"), leak=0.1)
        # net = tf.nn.relu(batch_norm(net, is_training, scope="g_bn3"))
        net = deconv2d(net, [-1, s_h2, s_w2, gf_dim * 1], 5, 5, 2, 2, name="g_dc3")
        net = lrelu(batch_norm(net, is_training, scope="g_bn4"), leak=0.1)
        # net = tf.nn.relu(batch_norm(net, is_training, scope="g_bn4"))
        net = deconv2d(net, [-1, s_h, s_w, output_c_dim], 5, 5, 2, 2, name="g_dc4")
        net = 0.5 * tf.nn.tanh(net) + 0.5

    info = {}
    info["variables"] = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=name)
    return net, info


######## DISCRIMINATORS
def simple_discriminator(x, batch_normalization=True, name="SIMPLE_discriminator", reuse=False, is_training=False):
    with tf.variable_scope(name, reuse=reuse):
        net = conv2d(
            x, 64, 4, 4, 2, 2, name="d_conv1")  # [bs, h/2, w/2, 64]
        net = lrelu(net)
        net = conv2d(
            net, 128, 4, 4, 2, 2, name="d_conv2")  # [bs, h/4, w/4, 128]
        if batch_normalization:
            net = batch_norm(net, is_training=True, scope="d_bn2")
        net = lrelu(net)
        net = tf.reshape(net, [-1, int(x.shape[1] * x.shape[2] * 8)])  # [bs, h * w * 8]
        net = linear(net, 1024, scope="d_fc3")  # [bs, 1024]
        if batch_normalization:
            net = batch_norm(net, is_training=True, scope="d_bn3")
        net = lrelu(net)
        out_logit = linear(net, 1, scope="d_fc4")  # [bs, 1]
        out = tf.nn.sigmoid(out_logit)

    info = {}
    info["variables"] = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=name)

    return out, out_logit, info


def DCGAN_discriminator(x, batch_normalization=True, name="DCGAN_dicriminator", reuse=False, is_training=False,
                        df_dim=64):
    with tf.variable_scope(name, reuse=reuse):
        net = lrelu(conv2d(x, df_dim, 5, 5, 2, 2, name="d_conv1"))
        net = conv2d(net, df_dim * 2, 5, 5, 2, 2, name="d_conv2")

        if batch_normalization:
            net = batch_norm(net, is_training, scope="d_bn1")
        net = lrelu(net)
        net = conv2d(net, df_dim * 4, 5, 5, 2, 2, name="d_conv3")

        if batch_norm:
            net = batch_norm(net, is_training, scope="d_bn2")
        net = lrelu(net)
        net = conv2d(net, df_dim * 8, 5, 5, 2, 2, name="d_conv4")

        if batch_norm:
            net = batch_norm(net, is_training, scope="d_bn3")
        net = lrelu(net)
        out_logit = linear(
            tf.reshape(net, [-1, int(net.shape[1] * net.shape[2] * net.shape[3])]), 1, scope="d_fc4")
        out = tf.nn.sigmoid(out_logit)
    info = {}
    info["variables"] = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=name)

    return out, out_logit, info


def BEGAN_discriminator(x, batch_normalization=True, name="BEGAN_discriminator", reuse=False, is_training=False):
    with tf.variable_scope(name, reuse=reuse):
        net = conv2d(
            x, 64, 4, 4, 2, 2, name="d_conv1")  # [bs, h/2, w/2, 64]
        net = lrelu(net)
        net = conv2d(
            net, 128, 4, 4, 2, 2, name="d_conv2")  # [bs, h/4, w/4, 128]
        net = tf.reshape(net, [-1, int(x.shape[1] * x.shape[2] * 8)])  # [bs, h * w * 8]
        code = linear(net, 64, scope="d_fc6")  # [bs, 64]
        if batch_normalization:
            code = batch_norm(code, is_training=is_training, scope="d_bn1")
        code = lrelu(code)

        # Decoding step (Mapping from [bs, 64] to [bs, h, w, c])

        net = linear(
            code, 128 * (int(x.get_shape()[1]) // 4) * (int(x.get_shape()[2]) // 4),
            scope="d_fc1")  # [bs, h/4 * w/4 * 128]
        if batch_normalization:
            net = batch_norm(net, is_training=is_training, scope="d_bn2")
        net = lrelu(net)
        net = tf.reshape(net, [
            -1, int(x.get_shape()[1] // 4), int(x.get_shape()[2] // 4), 128])  # [bs, h/4, w/4, 128]
        net = deconv2d(net, [-1, int(x.get_shape()[1] // 2), int(x.get_shape()[2] // 2), 64],
                       4, 4, 2, 2, name="d_deconv1")  # [bs, h/2, w/2, 64]
        if batch_normalization:
            net = batch_norm(net, is_training=is_training, scope="d_bn3")
        net = lrelu(net)
        net = deconv2d(net, [-1, int(x.get_shape()[1]), int(x.get_shape()[2]), x.get_shape()[3]],
                       4, 4, 2, 2, name="d_deconv2")  # [bs, h, w, c]
        out = tf.nn.sigmoid(net)

        # Reconstruction loss.
        recon_error = tf.reduce_mean(tf.abs(out - x))

    info = {}
    info["variables"] = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=name)
    return out, recon_error, code, info


######## GAN_TYPES
def simpleGAN(generator, discriminator, config, z_len=100, image_shape=[64, 64, 3], minimax=False):
    z_ph = tf.placeholder(tf.float32, shape=[None, z_len], name="z_placeholder")
    z_eval = tf.placeholder(tf.float32, shape=[None, z_len], name="z_placeholder_eval")
    real_images = tf.placeholder(tf.float32, shape=[None, image_shape[0], image_shape[1], image_shape[2]],
                                 name="real_images")
    learning_rate = tf.placeholder(tf.float32, name="learning_rate")

    generated_images, gen_info = generator(z_ph, output_shape=image_shape, name="GENERATOR", reuse=False,
                                           is_training=True)
    generated_images_eval, gen_info_eval = generator(z_eval, output_shape=image_shape, name="GENERATOR", reuse=True,
                                                     is_training=False)
    real_out, real_logits, dis_info = discriminator(real_images, name="DISCRIMINATOR", reuse=False, is_training=True)
    fake_out, fake_logits, _ = discriminator(generated_images, name="DISCRIMINATOR", reuse=True, is_training=True)

    real_dis_loss = tf.reduce_mean(
        tf.nn.sigmoid_cross_entropy_with_logits(logits=real_logits, labels=tf.ones_like(real_logits)))
    fake_dis_loss = tf.reduce_mean(
        tf.nn.sigmoid_cross_entropy_with_logits(logits=fake_logits, labels=tf.zeros_like(fake_logits)))
    dis_loss = tf.add(real_dis_loss, fake_dis_loss, name="discriminator_loss")

    if minimax:
        gen_loss = tf.multiply(fake_dis_loss, -1, name="generator_loss")
    else:
        gen_loss = tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(logits=fake_logits, labels=tf.ones_like(fake_logits)),
            name="generator_loss")

    dis_vars = dis_info["variables"]
    gen_vars = gen_info["variables"]

    gen_optim = tf.train.AdamOptimizer(learning_rate=learning_rate, beta1=config["beta1"],
                                       beta2=config["beta2"]).minimize(gen_loss, var_list=gen_vars)
    dis_optim = tf.train.AdamOptimizer(learning_rate=learning_rate, beta1=config["beta1"],
                                       beta2=config["beta2"]).minimize(dis_loss, var_list=dis_vars)

    feed_dict = {"z_ph": z_ph,
                 "real_images": real_images,
                 "z_eval": z_eval,
                 "fake_images": generated_images_eval,
                 "real_out": real_out,
                 "fake_out": fake_out,
                 "learning_rate": learning_rate,
                 "gen_loss": gen_loss,
                 "dis_loss": dis_loss}

    optimizers = [gen_optim, dis_optim]

    return optimizers, feed_dict


def testGAN(generator, discriminator, began_discriminator, config, z_len=100, image_shape=[64, 64, 3], minimax=False):
    z_ph = tf.placeholder(tf.float32, shape=[None, z_len], name="z_placeholder")
    z_eval = tf.placeholder(tf.float32, shape=[None, z_len], name="z_placeholder_eval")
    real_images = tf.placeholder(tf.float32, shape=[None, image_shape[0], image_shape[1], image_shape[2]],
                                 name="real_images")
    learning_rate = tf.placeholder(tf.float32, name="learning_rate")

    generated_images, gen_info = generator(z_ph, output_shape=image_shape, name="GENERATOR", reuse=False,
                                           is_training=True)
    tf.identity(generated_images, "generated_images")
    generated_images_eval, gen_info_eval = generator(z_eval, output_shape=image_shape, name="GENERATOR", reuse=True,
                                                     is_training=False)
    tf.identity(generated_images_eval, "generated_images_eval")
    real_out, real_logits, dis_info = discriminator(real_images, name="DISCRIMINATOR", reuse=False, is_training=True)
    tf.identity(real_out, "real_out")
    fake_out, fake_logits, _ = discriminator(generated_images, name="DISCRIMINATOR", reuse=True, is_training=True)
    tf.identity(real_out, "fake_out")

    real_out_be, real_error_be, real_code_be, dis_info_be = began_discriminator(real_images, name="DISCRIMINATOR_BE",
                                                                                reuse=False, is_training=True)
    tf.identity(real_out_be, "real_out_be")
    tf.identity(real_error_be, "real_error_be")
    tf.identity(real_code_be, "real_code_be")
    fake_out_be, fake_error_be, fake_code_be, _ = began_discriminator(generated_images, name="DISCRIMINATOR_BE",
                                                                      reuse=True, is_training=True)
    tf.identity(fake_out_be, "fake_out_be")
    tf.identity(fake_error_be, "fake_error_be")
    tf.identity(fake_code_be, "fake_code_be")

    k_value = tf.Variable(0., trainable=False)
    dis_loss_be = tf.add(real_error_be, - k_value * fake_error_be, name="dis_loss_be")

    gen_loss_init = fake_error_be
    update_k = k_value.assign(k_value + config["lambda"] *
                              (config["gamma"] * real_error_be - fake_error_be))

    real_dis_loss = tf.reduce_mean(
        tf.nn.sigmoid_cross_entropy_with_logits(logits=real_logits, labels=tf.ones_like(real_logits)))
    fake_dis_loss = tf.reduce_mean(
        tf.nn.sigmoid_cross_entropy_with_logits(logits=fake_logits, labels=tf.zeros_like(fake_logits)))
    dis_loss = tf.add(real_dis_loss, fake_dis_loss, name="dis_loss")

    if minimax:
        gen_loss = tf.add(gen_loss_init, tf.multiply(fake_dis_loss, -1), name="gen_loss")
    else:
        gen_loss = tf.add(gen_loss_init, tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(logits=fake_logits, labels=tf.ones_like(fake_logits))),
                          name="gen_loss")

    dis_vars = dis_info["variables"]
    dis_vars_be = dis_info_be["variables"]
    gen_vars = gen_info["variables"]

    gen_optim = tf.train.AdamOptimizer(learning_rate=learning_rate, beta1=config["beta1"],
                                       beta2=config["beta2"]).minimize(gen_loss, var_list=gen_vars, name="gen_optim")
    dis_optim = tf.train.AdamOptimizer(learning_rate=learning_rate, beta1=config["beta1"],
                                       beta2=config["beta2"]).minimize(dis_loss, var_list=dis_vars, name="dis_optim")
    dis_optim_be = tf.train.AdamOptimizer(learning_rate=learning_rate, beta1=config["beta1"],
                                          beta2=config["beta2"]).minimize(dis_loss_be, var_list=dis_vars_be,
                                                                          name="dis_optim_be")

    feed_dict = {"z_ph": z_ph,
                 "real_images": real_images,
                 "z_eval": z_eval,
                 "fake_images": generated_images_eval,
                 "real_out": [real_out, real_out_be],
                 "fake_out": [fake_out, fake_out_be],
                 "learning_rate": learning_rate,
                 "gen_loss": gen_loss,
                 "dis_loss": dis_loss}

    optimizers = [gen_optim, dis_optim, dis_optim_be, dis_optim_be, dis_optim_be, dis_optim_be, dis_optim_be, update_k]

    return optimizers, feed_dict


def WGAN(generator, discriminator, config, z_len=100, image_shape=[64, 64, 3], mode='regular'):
    assert mode in ['gp', 'regular']
    z_ph = tf.placeholder(tf.float32, shape=[None, z_len], name="z_placeholder")
    z_eval = tf.placeholder(tf.float32, shape=[None, z_len], name="z_placeholder_eval")
    real_images = tf.placeholder(tf.float32, shape=[None, image_shape[0], image_shape[1], image_shape[2]],
                                 name="real_images")
    learning_rate = tf.placeholder(tf.float32, name="learning_rate")

    generated_images, gen_info = generator(z_ph, output_shape=image_shape, name="GENERATOR", reuse=False,
                                           is_training=True)
    tf.identity(generated_images, "generated_images")
    generated_images_eval, gen_info_eval = generator(z_eval, output_shape=image_shape, name="GENERATOR", reuse=True,
                                                     is_training=False)
    tf.identity(generated_images_eval, "generated_images_eval")
    real_out, real_logits, dis_info = discriminator(real_images, batch_normalization=False, name="DISCRIMINATOR",
                                                    reuse=False, is_training=True)
    tf.identity(real_out, "real_out")
    fake_out, fake_logits, _ = discriminator(generated_images, batch_normalization=False, name="DISCRIMINATOR",
                                             reuse=True, is_training=True)
    tf.identity(real_out, "fake_out")

    dis_loss = tf.reduce_mean(tf.subtract(fake_logits, real_logits), name='dis_loss')
    gen_loss = tf.reduce_mean(-fake_logits, name='gen_loss')
    if mode == 'gp':
        alpha_dist = tf.contrib.distributions.Uniform(low=0., high=1.)
        alpha = alpha_dist.sample((config['batch_size'], 1, 1, 1))
        interpolated = real_images + alpha * (generated_images - real_images)
        interpolated_out, interpolated_logits, _ = discriminator(interpolated, batch_normalization=False,
                                                                 name="DISCRIMINATOR",
                                                                 reuse=True, is_training=True)

        gradients = tf.gradients(interpolated_logits, [interpolated, ])[0]
        grad_l2 = tf.sqrt(tf.reduce_sum(tf.square(gradients), axis=[1, 2, 3]))
        gradient_penalty = tf.reduce_mean((grad_l2 - 1) ** 2)
        dis_loss += config['lambda_gp'] * gradient_penalty

    dis_vars = dis_info["variables"]
    gen_vars = gen_info["variables"]

    gen_optim = tf.train.AdamOptimizer(learning_rate=learning_rate, beta1=config["beta1"],
                                       beta2=config["beta2"]).minimize(gen_loss, var_list=gen_vars, name="gen_optim")
    dis_optim = tf.train.AdamOptimizer(learning_rate=learning_rate, beta1=config["beta1"],
                                       beta2=config["beta2"]).minimize(dis_loss, var_list=dis_vars, name="dis_optim")

    if mode is 'regular':
        clipped_var_c = [tf.assign(var, tf.clip_by_value(var, config['clamp_lower'], config['clamp_upper'])) for var in
                         dis_vars]
        with tf.control_dependencies([dis_optim]):
            dis_optim = tf.tuple(clipped_var_c)

    feed_dict = {"z_ph": z_ph,
                 "real_images": real_images,
                 "z_eval": z_eval,
                 "fake_images": generated_images_eval,
                 "learning_rate": learning_rate,
                 "gen_loss": gen_loss,
                 "dis_loss": dis_loss}

    optimizers = [gen_optim] + [dis_optim for _ in range(config["dis_iters"])]

    return optimizers, feed_dict


def BEGAN(generator, began_discriminator, config, z_len=100, image_shape=[64, 64, 3]):
    z_ph = tf.placeholder(tf.float32, shape=[None, z_len], name="z_placeholder")
    z_eval = tf.placeholder(tf.float32, shape=[None, z_len], name="z_placeholder_eval")
    real_images = tf.placeholder(tf.float32, shape=[None, image_shape[0], image_shape[1], image_shape[2]],
                                 name="real_images")
    learning_rate = tf.placeholder(tf.float32, name="learning_rate")

    k_value = tf.Variable(0., trainable=False)

    generated_images, gen_info = generator(z_ph, output_shape=image_shape, name="GENERATOR", reuse=False,
                                           is_training=True)
    generated_images_eval, _ = generator(z_eval, output_shape=image_shape, name="GENERATOR", reuse=True,
                                         is_training=False)
    real_out, real_error, real_code, dis_info = began_discriminator(real_images, name="DISCRIMINATOR", reuse=False,
                                                                    is_training=True)
    fake_out, fake_error, fake_code, _ = began_discriminator(generated_images, name="DISCRIMINATOR", reuse=True,
                                                             is_training=True)

    dis_loss = real_error - k_value * fake_error
    gen_loss = fake_error

    update_k = k_value.assign(k_value + config["lambda"] *
                              (config["gamma"] * real_error - fake_error))

    dis_vars = dis_info["variables"]
    gen_vars = gen_info["variables"]

    gen_optim = tf.train.AdamOptimizer(learning_rate=learning_rate, beta1=config["beta1"],
                                       beta2=config["beta2"]).minimize(gen_loss, var_list=gen_vars)
    dis_optim = tf.train.AdamOptimizer(learning_rate=learning_rate, beta1=config["beta1"],
                                       beta2=config["beta2"]).minimize(dis_loss, var_list=dis_vars)

    feed_dict = {"z_ph": z_ph,
                 "real_images": real_images,
                 "z_eval": z_eval,
                 "fake_images": generated_images_eval,
                 "real_out": real_out,
                 "fake_out": fake_out,
                 "learning_rate": learning_rate,
                 "gen_loss": gen_loss,
                 "dis_loss": dis_loss}

    optimizers = [gen_optim]
    optimizers += [dis_optim for _ in range(config["dis_iters"])]
    optimizers += [update_k]

    return optimizers, feed_dict


def DRAGAN(generator, discriminator, config, z_len=100, image_shape=[64, 64, 3]):
    z_ph = tf.placeholder(tf.float32, shape=[None, z_len], name="z_placeholder")
    z_eval = tf.placeholder(tf.float32, shape=[None, z_len], name="z_placeholder_eval")
    real_images = tf.placeholder(tf.float32, shape=[None, image_shape[0], image_shape[1], image_shape[2]],
                                 name="real_images")
    real_images_p = tf.placeholder(tf.float32, shape=[None, image_shape[0], image_shape[1], image_shape[2]],
                                   name="real_images_perturbed")
    learning_rate = tf.placeholder(tf.float32, name="learning_rate")

    generated_images, gen_info = generator(z_ph, output_shape=image_shape, name="GENERATOR", reuse=False,
                                           is_training=True)
    generated_images_eval, gen_info_eval = generator(z_eval, output_shape=image_shape, name="GENERATOR", reuse=True,
                                                     is_training=False)
    real_out, real_logits, dis_info = discriminator(real_images, name="DISCRIMINATOR", reuse=False, is_training=True)
    fake_out, fake_logits, _ = discriminator(generated_images, name="DISCRIMINATOR", reuse=True, is_training=True)

    real_dis_loss = tf.reduce_mean(
        tf.nn.sigmoid_cross_entropy_with_logits(logits=real_logits, labels=tf.ones_like(real_logits)))
    fake_dis_loss = tf.reduce_mean(
        tf.nn.sigmoid_cross_entropy_with_logits(logits=fake_logits, labels=tf.zeros_like(fake_logits)))
    dis_loss = tf.add(real_dis_loss, fake_dis_loss, name="discriminator_loss")
    gen_loss = tf.reduce_mean(
        tf.nn.sigmoid_cross_entropy_with_logits(logits=fake_logits, labels=tf.ones_like(fake_logits)),
        name="generator_loss")

    alpha = tf.random_uniform(shape=[1], minval=0., maxval=1.)
    differences = real_images_p - real_images
    interpolates = real_images + (alpha * differences)
    D_inter, _, _ = discriminator(interpolates, name="DISCRIMINATOR", is_training=True, reuse=True)
    gradients = tf.gradients(D_inter, [interpolates])[0]
    slopes = tf.sqrt(0.0001 + tf.reduce_sum(
        tf.square(gradients), reduction_indices=[1, 2, 3]))
    gradient_penalty = tf.reduce_mean(tf.square(slopes - 1.))
    dis_loss += config["lambda"] * gradient_penalty

    dis_vars = dis_info["variables"]
    gen_vars = gen_info["variables"]

    gen_optim = tf.train.AdamOptimizer(learning_rate=learning_rate, beta1=config["beta1"],
                                       beta2=config["beta2"]).minimize(gen_loss, var_list=gen_vars)
    dis_optim = tf.train.AdamOptimizer(learning_rate=learning_rate, beta1=config["beta1"],
                                       beta2=config["beta2"]).minimize(dis_loss, var_list=dis_vars)

    feed_dict = {"z_ph": z_ph,
                 "real_images": real_images,
                 "real_images_perturbed": real_images_p,
                 "z_eval": z_eval,
                 "fake_images": generated_images_eval,
                 "real_out": real_out,
                 "fake_out": fake_out,
                 "gen_loss": gen_loss,
                 "dis_loss": dis_loss}

    optimizers = [gen_optim]
    optimizers += [dis_optim for _ in range(config["dis_iters"])]

    elements = {}
    elements["z_eval"] = tf.get_default_graph().get_tensor_by_name("z_placeholder_eval:0")
    elements["output_eval"] = tf.get_default_graph().get_tensor_by_name(gen_info_eval["output_name"])
    elements["output_eval"] = tf.get_default_graph().get_tensor_by_name(gen_info_eval["output_name"])
    return optimizers, feed_dict, elements


######## TRAIN_GAN
def make_config(config):
    if not config.get("batch_size"): config["batch_size"] = 32
    if not config.get("z_sd"): config["batch_size"] = 64
    if not config.get("num_steps"): config["num_steps"] = 10000
    if not config.get("learning_rate"): config["num_steps"] = 0.0005
    if not config.get("DRAGAN"): config["DRAGAN"] = False
    if not config.get("save_every"): config["save_every"] = 100


def train_gan(sess, optimizers, feed_dict, data_content, config):
    import matplotlib.pyplot as plt
    make_config(config)

    import matplotlib
    if not config["new_tf"]:
        matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    if config["new_tf"]:
        dataset = data_content.batch(config["batch_size"])
        images, _ = dataset.make_one_shot_iterator().get_next()

    model_name = config["model_name"]

    if model_name is not None:
        saver = tf.train.Saver()
        if os.path.exists(model_name):
            raise Exception("Model name already exists")
        os.mkdir(model_name)
        os.mkdir(os.path.join(model_name, "outputs"))

    z_for_eval = np.random.normal(0, config["z_sd"], [100, feed_dict["z_ph"].shape[1]])

    lr_decay = 1
    decay_step = 1000
    if config["learning_rate"].__class__ == list and len(config["learning_rate"]) == 3:
        lr_decay = config["learning_rate"][1]
        decay_step = config["learning_rate"][2]
        config["learning_rate"] = config["learning_rate"][0]

    for step in range(config["num_steps"]):
        if config["new_tf"]:
            step_images = sess.run(images)
        else:
            step_images = data_content.next_batch(config["batch_size"])

        step_z = np.random.normal(0, config["z_sd"], [config["batch_size"], feed_dict["z_ph"].shape[1]])
        step_fd = {feed_dict["z_ph"]: step_z,
                   feed_dict["real_images"]: step_images,
                   feed_dict["learning_rate"]: config["learning_rate"]}
        if config["DRAGAN"]:
            real_perturbed = step_images + 0.5 * step_images.std() * np.random.random(step_images.shape)
            step_fd[feed_dict["real_images_perturbed"]] = real_perturbed

        step_gen_loss, step_dis_loss, *rest = sess.run([feed_dict["gen_loss"],
                                                        feed_dict["dis_loss"]] +
                                                       feed_dict["real_out"] +
                                                       feed_dict["fake_out"],
                                                       feed_dict=step_fd)
        step_real_outs = rest[:int(len(rest) / 2)]
        step_fake_outs = rest[int(len(rest) / 2):]
        step_real_accs = [np.sum(x) / config["batch_size"] for x in step_real_outs]
        step_fake_accs = [1 - np.sum(x) / config["batch_size"] for x in step_fake_outs]
        sess.run(optimizers, feed_dict=step_fd)

        print("STEP: " + str(step), "   ",
              "Learning_rate: " + str(config["learning_rate"]), "   ",
              "Generator_loss: " + str(step_gen_loss), "   ",
              "Discriminator_loss: " + str(step_dis_loss), "   ",
              "Real ACC: " + str(step_real_accs), "   ",
              "Fake ACC: " + str(step_fake_accs), "   ")

        if step % decay_step == 0 and step > 0:
            config["learning_rate"] *= lr_decay

        if step % config["save_every"] == 0:
            fake_images = sess.run(feed_dict["fake_images"], feed_dict={feed_dict["z_eval"]: z_for_eval})

            images_gallery = gallery(array=fake_images, ncols=10)
            if model_name is not None:
                saver.save(sess, os.path.join(model_name, "GANModel"), global_step=step)
                plt.imsave(os.path.join(model_name, "outputs", "RESULT" + str(step) + ".png"), images_gallery)
            else:
                plt.imsave("temp/RESULT" + str(step) + ".png", images_gallery)
        # if step % 100 == 0:
        #     config["learning_rate"] /= 2


def train_gan_dataloader(sess, optimizers, feed_dict, dataloader, config):
    import matplotlib.pyplot as plt
    make_config(config)

    import matplotlib
    import matplotlib.pyplot as plt

    sess.run(dataloader.train_initializer)
    x_train, y_train, _ = dataloader.train_batch

    model_name = config["model_name"]

    if model_name is not None:
        saver = tf.train.Saver()
        if os.path.exists(model_name):
            raise Exception("Model name already exists")
        os.mkdir(model_name)
        os.mkdir(os.path.join(model_name, "outputs"))

    z_for_eval = np.random.normal(0, config["z_sd"], [100, feed_dict["z_ph"].shape[1]])

    lr_decay = 1
    decay_step = 1000
    if config["learning_rate"].__class__ == list and len(config["learning_rate"]) == 3:
        lr_decay = config["learning_rate"][1]
        decay_step = config["learning_rate"][2]
        config["learning_rate"] = config["learning_rate"][0]

    for step in range(config["num_steps"]):
        try:
            step_images = sess.run(x_train)
        except Exception as e:
            sess.run(dataloader.train_initializer)

        step_z = np.random.normal(0, config["z_sd"], [config["batch_size"], feed_dict["z_ph"].shape[1]])
        step_fd = {feed_dict["z_ph"]: step_z,
                   feed_dict["real_images"]: step_images,
                   feed_dict["learning_rate"]: config["learning_rate"]}
        if config["DRAGAN"]:
            real_perturbed = step_images + 0.5 * step_images.std() * np.random.random(step_images.shape)
            step_fd[feed_dict["real_images_perturbed"]] = real_perturbed

        step_gen_loss, step_dis_loss, *rest = sess.run([feed_dict["gen_loss"],
                                                        feed_dict["dis_loss"]] +
                                                       feed_dict["real_out"] +
                                                       feed_dict["fake_out"],
                                                       feed_dict=step_fd)
        step_real_outs = rest[:int(len(rest) / 2)]
        step_fake_outs = rest[int(len(rest) / 2):]
        step_real_accs = [np.sum(x) / config["batch_size"] for x in step_real_outs]
        step_fake_accs = [1 - np.sum(x) / config["batch_size"] for x in step_fake_outs]
        sess.run(optimizers, feed_dict=step_fd)

        print("STEP: " + str(step), "   ",
              "Learning_rate: " + str(config["learning_rate"]), "   ",
              "Generator_loss: " + str(step_gen_loss), "   ",
              "Discriminator_loss: " + str(step_dis_loss), "   ",
              "Real ACC: " + str(step_real_accs), "   ",
              "Fake ACC: " + str(step_fake_accs), "   ")

        if step % decay_step == 0 and step > 0:
            config["learning_rate"] *= lr_decay

        if step % config["save_every"] == 0:
            z_for_eval_ = np.random.normal(0, config["z_sd"], [100, feed_dict["z_ph"].shape[1]])
            z_for_eval_[:50] = z_for_eval[:50]
            fake_images = sess.run(feed_dict["fake_images"], feed_dict={feed_dict["z_eval"]: z_for_eval_})

            images_gallery = gallery(array=fake_images, ncols=10)
            if images_gallery.shape[-1] == 1:
                images_gallery = images_gallery[:, :, 0]
            if model_name is not None:
                saver.save(sess, os.path.join(model_name, "GANModel"), global_step=step)
                plt.imsave(os.path.join(model_name, "outputs", "RESULT" + str(step) + ".png"), images_gallery)
            else:
                plt.imsave("temp/RESULT" + str(step) + ".png", images_gallery)
        # if step % 100 == 0:
        #     config["learning_rate"] /= 2


def train_wgan_dataloader(sess, optimizers, feed_dict, dataloader, config):
    import matplotlib.pyplot as plt
    make_config(config)

    import matplotlib
    import matplotlib.pyplot as plt

    sess.run(dataloader.train_initializer)
    x_train, y_train, _ = dataloader.train_batch

    model_name = config["model_name"]

    if model_name is not None:
        saver = tf.train.Saver()
        if os.path.exists(model_name):
            raise Exception("Model name already exists")
        os.mkdir(model_name)
        os.mkdir(os.path.join(model_name, "outputs"))

    z_for_eval = np.random.normal(0, config["z_sd"], [100, feed_dict["z_ph"].shape[1]])

    lr_decay = 1
    decay_step = 1000
    if config["learning_rate"].__class__ == list and len(config["learning_rate"]) == 3:
        lr_decay = config["learning_rate"][1]
        decay_step = config["learning_rate"][2]
        config["learning_rate"] = config["learning_rate"][0]

    for step in range(config["num_steps"]):
        try:
            step_images = sess.run(x_train)
        except Exception as e:
            sess.run(dataloader.train_initializer)

        step_z = np.random.normal(0, config["z_sd"], [config["batch_size"], feed_dict["z_ph"].shape[1]])
        step_fd = {feed_dict["z_ph"]: step_z,
                   feed_dict["real_images"]: step_images,
                   feed_dict["learning_rate"]: config["learning_rate"]}
        if config["DRAGAN"]:
            real_perturbed = step_images + 0.5 * step_images.std() * np.random.random(step_images.shape)
            step_fd[feed_dict["real_images_perturbed"]] = real_perturbed

        step_gen_loss, step_dis_loss = sess.run([feed_dict["gen_loss"],
                                                 feed_dict["dis_loss"]],
                                                feed_dict=step_fd)

        sess.run(optimizers, feed_dict=step_fd)

        print("STEP: " + str(step), "   ",
              "Learning_rate: " + str(config["learning_rate"]), "   ",
              "Generator_loss: " + str(step_gen_loss), "   ",
              "Discriminator_loss: " + str(step_dis_loss), "   ")

        if step % decay_step == 0 and step > 0:
            config["learning_rate"] *= lr_decay

        if step % config["save_every"] == 0:
            z_for_eval_ = np.random.normal(0, config["z_sd"], [100, feed_dict["z_ph"].shape[1]])
            z_for_eval_[:50] = z_for_eval[:50]
            fake_images = sess.run(feed_dict["fake_images"], feed_dict={feed_dict["z_eval"]: z_for_eval_})

            images_gallery = gallery(array=fake_images, ncols=10)
            if images_gallery.shape[-1] == 1:
                images_gallery = images_gallery[:, :, 0]
            if model_name is not None:
                saver.save(sess, os.path.join(model_name, "GANModel"), global_step=step)
                plt.imsave(os.path.join(model_name, "outputs", "RESULT" + str(step) + ".png"), images_gallery)
            else:
                plt.imsave("temp/RESULT" + str(step) + ".png", images_gallery)
        # if step % 100 == 0:
        #     config["learning_rate"] /= 2


######## LOAD AND GENERATE RESULTS
def load_and_generate_results(sess, model_name, num_transforms=50):
    meta_files = [os.path.join(model_name, x) for x in os.listdir(model_name) if x.endswith(".meta")]
    meta_files = sorted(meta_files)
    saver = tf.train.import_meta_graph(meta_files[-1])
    saver.restore(sess, meta_files[-1].split(".")[0])

    z_ph = tf.get_default_graph().get_tensor_by_name("z_placeholder:0")
    z_eval = tf.get_default_graph().get_tensor_by_name("z_placeholder_eval:0")
    real_images = tf.get_default_graph().get_tensor_by_name("real_images:0")
    learning_rate = tf.get_default_graph().get_tensor_by_name("learning_rate:0")

    # for op in tf.get_default_graph().get_operations():
    #     print(str(op.name))

    # generated_images_eval = tf.get_default_graph().get_tensor_by_name("generated_images_eval:0")
    generated_images_eval = tf.get_default_graph().get_tensor_by_name("GENERATOR_1/add:0")

    import matplotlib.pyplot as plt

    ###Saving Some Generated Pictures
    z_feed = np.random.normal(size=[100, int(z_eval.shape[1])])
    gen_images = sess_run_several_images(sess, generated_images_eval, {z_eval: z_feed})
    # gen_images = bgr2rgb(gen_images)
    images_gallery = gallery(array=gen_images, ncols=10)
    plt.imsave(os.path.join(model_name, "generated.png"), images_gallery)

    ###Animation between states
    import imageio
    import cv2
    z = np.random.normal(size=[num_transforms, int(z_eval.shape[1])])
    z[-1, :] = z[0, :].copy()
    video = animate_list_of_latents(z, 20)
    gen_images = sess_run_several_images(sess, generated_images_eval, {z_eval: video})
    if os.path.exists(os.path.join(model_name, "frames")):
        import shutil
        shutil.rmtree(os.path.join(model_name, "frames"))
    os.mkdir(os.path.join(model_name, "frames"))
    images_for_gif = []
    for iter in range(video.shape[0]):
        this_image = gen_images[iter, :, :, :]
        plt.imsave(os.path.join(model_name, "frames/frame" + str(iter) + ".jpg"), this_image)
        images_for_gif.append(imageio.imread(os.path.join(model_name, "frames/frame" + str(iter) + ".jpg")))

    imageio.mimsave(os.path.join(model_name, "Linear_Transformations.gif"), images_for_gif)

    print("DONE")


def load_visualization_form(sess, model_name):
    meta_files = [os.path.join(model_name, x) for x in os.listdir(model_name) if x.endswith(".meta")]
    meta_files = sorted(meta_files)
    saver = tf.train.import_meta_graph(meta_files[-1])
    saver.restore(sess, meta_files[-1].split(".")[0])

    z_ph = tf.get_default_graph().get_tensor_by_name("z_placeholder:0")
    z_eval = tf.get_default_graph().get_tensor_by_name("z_placeholder_eval:0")
    real_images = tf.get_default_graph().get_tensor_by_name("real_images:0")
    learning_rate = tf.get_default_graph().get_tensor_by_name("learning_rate:0")

    # for op in tf.get_default_graph().get_operations():
    #     print(str(op.name))

    # generated_images_eval = tf.get_default_graph().get_tensor_by_name("generated_images_eval:0")
    generated_images_eval = tf.get_default_graph().get_tensor_by_name("GENERATOR_1/add:0")

    def gen_fn(z_list):
        z_stacked = np.stack(z_list, axis=0)
        gen_images_bgr = sess.run(generated_images_eval, feed_dict={z_eval: z_stacked})
        # gen_images = bgr2rgb(gen_images_bgr)
        gen_images = gen_images_bgr
        return [gen_images[i, :, :, :] for i in range(gen_images.shape[0])]

    try:
        import Visualization_Form as VF
        VF.Visualization_Form(GAN_config={"z_sd": 1.0, "z_len": int(z_eval.shape[1])}, gen_fn=gen_fn, num_sliders=20)
    except Exception as e:
        print("YOU DONT HAVE VISUALIZATION SCRIPT. SORRY")


def bgr2rgb(images):
    images_2 = images.copy()
    images_2[:, :, :, 0] = images[:, :, :, 2].copy()
    images_2[:, :, :, 2] = images[:, :, :, 0].copy()
    return images_2
