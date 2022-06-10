__author__ = 'Christopher Sweet'
"""
Various Machine Learning Architectures with the intent of recreating cyber-event
data. Includes GAN, WGAN, WGAN_GP, and more
"""
import os
import sys
import time
import json
from pprint import pprint
from copy import deepcopy

import tensorflow as tf
import numpy as np
import pandas as pd
from pandas.plotting import table as tab
import matplotlib
import matplotlib.pyplot as plt

from utilities import *


# Settings for matplotlib and tensorflow
font = {'size': 34}
matplotlib.rc('font', **font)
matplotlib.rc('xtick', labelsize=30)
matplotlib.rc('ytick', labelsize=30)
matplotlib.rc('legend', fontsize=28)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


class BaseModel():
    '''
    Base Model for attributes and functions shared by generative models
        __init__()
        load_model()
        save_model()
    '''
    def __init__(self, sess, sample_dim:int, h_dim:int, num_classes:int,
                 unique_values:list=None, checkpoint_dir:str='./checkpoints',
                 log_dir:str='./logs', name="Base"):
        """
        Creates a new generic generative model
        :param sess: TensorFlow Session
        :param sample_dim: Features in the data container
        :param h_dim: Hidden Dimension used for NNs
        :param num_classes:  Dimension of the generator/gt output
        :param unique_values: List of Lists containing unique values for the dataset
        """
        self.sess = sess
        self.sample_dim = sample_dim
        self.h_dim = h_dim
        self.num_classes = num_classes
        self.unique_values = unique_values
        self.checkpoints = checkpoint_dir
        self.logs = log_dir
        self.name = name

    def load_model(self):
        '''
        Attempts to load the given model returning it's status as a boolean
        :return: Boolean indicating load status
        '''
        ckpt = tf.train.get_checkpoint_state(self.checkpoints)
        if ckpt and ckpt.model_checkpoint_path:
            self.saver.restore(self.sess, ckpt.model_checkpoint_path)
            return True
        else:
            return False

    def save_model(self, counter:int):
        '''
        Saves the given model at the step counter given
        :param counter: Step of training
        '''
        if not os.path.exists(self.checkpoints):
            os.makedirs(self.checkpoints)
        self.saver.save(self.sess, os.path.join(self.checkpoints, self.name),
            global_step=counter)


class GAN(BaseModel):
    """
    Generative Adversarial Network:

    Game Theory Model for data generation. Pits two networks against each other
    a generator and a discriminator. As each improves the other must do so as
    well.

    https://arxiv.org/pdf/1406.2661.pdf
    https://arxiv.org/pdf/1701.07875.pdf
    """

    def __init__(self, sess, sample_dim:int, h_dim:int, num_classes:int, unique_values:list=None, checkpoint_dir:str='./checkpoints', name:str='GAN'):
        """
        Creates a new Generative Adversarial Network
        :param sess: TensorFlow Session
        :param sample_dim: Features in the data container
        :param h_dim: Hidden Dimension used for NNs
        :param num_classes:  Dimension of the generator/gt output
        :param unique_values: List of lists containing the unique_values of the dataset on a per feature basis
        """
        super().__init__(sess, sample_dim, h_dim, num_classes, unique_values, checkpoint_dir=checkpoint_dir, name=name)
        self.noise_dim = 64

    def model_loss(self):
        '''
        Specify the loss function to use for the model.

        :notes: Makes inheritance easier for WGAN model
        '''
        d_real = tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(
                logits = self.D_logits,
                labels = tf.ones_like(self.D)
            )
        )
        d_fake = tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(
                logits = self.D_logits_,
                labels = tf.zeros_like(self.D_)
            )
        )
        self.d_loss = 0.5 * (d_real + d_fake)
        self.g_loss = tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(
                logits = self.D_logits_,
                labels = tf.ones_like(self.D_)
            )
        )

        #Add in regularization to loss
        self.reg = tf.contrib.layers.apply_regularization(
            tf.contrib.layers.l1_regularizer(2.5e-5),
            weights_list=[var for var in tf.global_variables() if 'weights' in var.name]
        )
        self.g_loss_reg = self.g_loss + self.reg
        self.d_loss_reg = self.d_loss + self.reg

    def build_model(self):
        """
        Build Generative Adversarial Network architecture
        """
        self.d = tf.placeholder(tf.float32, shape=[None, self.sample_dim],
            name='dis_data')
        self.g = tf.placeholder(tf.float32, shape=[None, self.noise_dim],
            name='gen_data')

        #Generative Network
        self.G_logits = self.generator(self.g)
        #Discrminator Network - Ground-truth input
        self.D, self.D_logits = self.discriminator(self.d)
        #Discrminator Network - Generator input
        self.D_, self.D_logits_ = self.discriminator(self.G_logits, reuse=True)

        #Define the loss functions for the model - In this case DCGAN Loss
        self.model_loss()

        #Summary Histograms
        for var in tf.trainable_variables():
            tf.summary.histogram(var.name, var)
        self.merged_summary = tf.summary.merge_all()

        # Collect all variables for generator and discriminator. TF will perform a gradient check if given explicitely to optimizer
        self.g_vars = [var for var in tf.global_variables() if 'generator' in var.name]
        self.d_vars = [var for var in tf.global_variables() if 'discriminator' in var.name]

        #Saver and File Writer
        self.saver = tf.train.Saver(max_to_keep=1)
        self.writer = tf.summary.FileWriter(self.logs, self.sess.graph)

    def train_model(self, config:dict, samples:list, labels:list):
        """
        Trains the network
        :param config: Configuration dictionary for tuned hyperparameters
        :param samples: Samples to input for training
        :param: labels: List of labels to feed in as conditioning
        """
        d_opt = tf.train.RMSPropOptimizer(
            learning_rate=config['learning_rate'],
            name='d_opt').minimize(
                self.d_loss,
                var_list=self.d_vars
            )
        g_opt = tf.train.RMSPropOptimizer(
            learning_rate=config['learning_rate'],
            name='g_opt').minimize(
                self.g_loss,
                var_list=self.g_vars
            )

        tf.global_variables_initializer().run()

        #Adversarial Training
        counter = 0
        for epoch in range(config['epoch']):
            is_new_epoch = True
            steps = len(samples) // config['batch_size']
            for step in range(steps):
                r_sample = np.random.choice(len(samples), size=config['batch_size'],
                    replace=False)
                #Feed random noise into GAN to procude results after pretraining
                # noise_z = np.random.random([len(r_sample), self.sample_dim])
                batch_z = np.random.normal(-1, 1, [config['batch_size'], config['noise_dims']]).astype(np.float32)
                sample = samples[r_sample]
                label = labels[r_sample]
                _, d_loss, var_save = self.sess.run(
                    [
                        d_opt,
                        self.d_loss,
                        self.merged_summary
                    ],
                    feed_dict = {
                        self.d: sample,
                        self.g: batch_z
                    }
                )
                _, g_loss, var_save = self.sess.run(
                    [
                        g_opt,
                        self.g_loss,
                        self.merged_summary
                    ],
                    feed_dict = {
                        self.g: batch_z
                    }
                )

                #Every 100 steps save the weights and model state
                if counter % 100 == 0:
                    d_loss, g_loss = self.sess.run(
                    [
                        self.d_loss,
                        self.g_loss
                    ],
                    feed_dict = {
                            self.d: sample,
                            self.g: batch_z
                        }
                    )
                    self.writer.add_summary(var_save, counter)
                    self.save_model(counter)

                #Show updated results, append a header if it's a new epoch
                if is_new_epoch:
                    sys.stdout.write("\nEpoch: %d\n" % (epoch))
                    sys.stdout.flush()
                    is_new_epoch = False
                sys.stdout.write(
                    "    Step %d    Discriminator: %g    Generator: %g        \r" %
                    (counter, d_loss, g_loss)
                )
                sys.stdout.flush()

                counter += 1
        print('\n')

    def generator(self, x):
        """
        Generates synthetic data through a series of transforms
        :param x: Input seed for generation
        :return gen_y: Synthetic Data
        """
        with tf.variable_scope("generator") as scope:
            gen_l1 = tf.contrib.layers.fully_connected(
                inputs = x,
                num_outputs = self.h_dim
            )
            gen_y_logits = []
            for i, _ in enumerate(self.unique_values):
                size = len(self.unique_values[i])
                gen_y_logits.append(
                    tf.contrib.layers.fully_connected(
                        inputs = gen_l1,
                        num_outputs = size,
                        activation_fn = None
                    )
                )

            gen_y_logits = tf.concat(gen_y_logits, axis=1)
            return gen_y_logits

    def discriminator(self, x, reuse=False):
        """
        Discriminates between real and synthetic data
        :param x: Input data to discriminate
        :param y: Input label to condition on
        :param reuse: (Optional) Used when Trained Variables must be reused
        :return dis_y, dis_y_logits: Discriminated output and logits
        """
        with tf.variable_scope("discriminator") as scope:
            if reuse:
                scope.reuse_variables()

            dis_l1 = tf.contrib.layers.fully_connected(
                inputs = x,
                num_outputs = self.h_dim
            )
            dis_y_logits = tf.contrib.layers.fully_connected(
                inputs = dis_l1,
                num_outputs = 1,
                activation_fn = None
            )
            dis_y = tf.nn.sigmoid(dis_y_logits)
            return dis_y, dis_y_logits

    def generate_sample(self, x):
        """
        Reconstructs the sample from the given data
        :param x: Input noise to create samples from
        :return: Sample recreated from latent space
        """
        predictions = self.sess.run(self.G_logits,
            feed_dict = {
                self.g: x
            }
        )
        # Take the argmax per feature value to get real valued output
        start_ind = 0
        prediction_indices = np.zeros((predictions.shape[0], len(self.unique_values)))
        for i in range(len(self.unique_values)):
            size = len(self.unique_values[i])
            slice = predictions[:, start_ind:start_ind+size]
            prediction_indices[:,i] = (np.argmax(slice, axis=1))
            start_ind += size
        return prediction_indices

class WGAN(GAN):
    """
    Wasserstien Generative Adversarial Network:

    An expansion of the GAN which uses the Earthmover Distance rather than
    minimax loss between the two networks. Employs manual gradient clipping to
    avoid gradient explosion

    """

    def model_loss(self):
        '''
        Specify WGAN loss and add in gradient clipping for discriminator
        '''
        #WGAN Loss with clipping applied after
        self.d_loss = tf.reduce_mean((self.D_) - tf.reduce_mean(self.D))
        self.g_loss = -tf.reduce_mean(self.D_)

        #Manually clip discriminator values for wgan
        clip_ops = []

        for var in tf.contrib.framework.get_variables('discriminator'):
            clip_bounds = [-0.01, 0.01]
            clip_ops.append(
                tf.assign(
                    var,
                    tf.clip_by_value(var, clip_bounds[0], clip_bounds[1])
                )
            )
        self.clip_dis_weights = tf.group(*clip_ops)

        #Add in regularization to loss
        self.reg = tf.contrib.layers.apply_regularization(
            tf.contrib.layers.l1_regularizer(2.5e-5),
            weights_list=[var for var in tf.global_variables() if 'weights' in var.name]
        )

        # Collect all variables for generator and discriminator. TF will perform a gradient check if given explicitely to optimizer
        self.g_vars = [var for var in tf.global_variables() if 'generator' in var.name]
        self.d_vars = [var for var in tf.global_variables() if 'discriminator' in var.name]

        self.g_loss_reg = self.g_loss + self.reg
        self.d_loss_reg = self.d_loss + self.reg

    def train_model(self, config:dict, samples:list, labels:list):
        """
        Trains the network - redefined from parent class to add in gradient clipping
        :param config: Configuration dictionary for tuned hyperparameters
        :param samples: Samples to input for training
        :param: labels: List of labels to feed in as conditioning
        """
        d_opt = tf.train.RMSPropOptimizer(
            learning_rate=config['learning_rate'],
            name='d_opt').minimize(
                self.d_loss_reg,
                var_list=self.d_vars
            )
        g_opt = tf.train.RMSPropOptimizer(
            learning_rate=config['learning_rate'],
            name='g_opt').minimize(
                self.g_loss_reg,
                var_list=self.g_vars
            )

        tf.global_variables_initializer().run()

        #Adversarial Training
        counter = 0
        for epoch in range(config['epoch']):
            is_new_epoch = True
            steps = len(samples) // config['batch_size']
            for step in range(steps):
                r_sample = np.random.choice(len(samples), size=config['batch_size'],
                    replace=False)

                #Discriminator Training
                for _ in range(config['discriminator_iterations']):
                    batch_z = np.random.normal(-1, 1, [config['batch_size'], config['noise_dims']]).astype(np.float32)
                    sample = samples[r_sample]

                    #One hot encode each feature an concatenate into a single input of dim (batch_size, sum(encodings))
                    sample_one_hot = []
                    for i, s in enumerate(sample.T):
                        sample_one_hot.append(one_hot(s, len(self.unique_values[i])))
                    sample_one_hot = np.concatenate(sample_one_hot, axis=1)
                    sample = sample_one_hot

                    label = labels[r_sample]

                    _, d_loss, var_save = self.sess.run(
                        [
                            d_opt,
                            self.d_loss_reg,
                            self.merged_summary
                        ],
                        feed_dict = {
                            self.d: sample,
                            self.g: batch_z
                        }
                    )

                    #Clip Discr\minator weights manually
                    _ = self.sess.run(
                        [self.clip_dis_weights]
                    )

                #Generator Training
                _, g_loss, var_save = self.sess.run(
                    [
                        g_opt,
                        self.g_loss_reg,
                        self.merged_summary
                    ],
                    feed_dict = {
                        self.g: batch_z
                    }
                )

                #Every 100 steps save the weights and model state
                if counter % 100 == 0:
                    d_loss, g_loss = self.sess.run(
                    [
                        self.d_loss_reg,
                        self.g_loss_reg
                    ],
                    feed_dict = {
                            self.d: sample,
                            self.g: batch_z
                        }
                    )
                    self.writer.add_summary(var_save, counter)
                    self.save_model(counter)

                #Show updated results, append a header if it's a new epoch
                if is_new_epoch:
                    sys.stdout.write("\nEpoch: %d\n" % (epoch))
                    sys.stdout.flush()
                    is_new_epoch = False
                sys.stdout.write(
                    "    Step %d    Discriminator: %g    Generator: %g        \r" %
                    (counter, d_loss, g_loss)
                )
                sys.stdout.flush()

                counter += 1
        print('\n')



class WGAN_GP(GAN):
    """
    Wasserstien Generative Adversarial Network with Gradient Penalty:

    An expansion of the GAN which uses the Earthmover Distance rather than
    minimax loss between the two networks. Removes need for gradient clipping
    by adding in a gradient penalty term to the loss

    """

    def model_loss(self):
        '''
        Specify WGAN loss and add in gradient clipping for discriminator
        '''
        #WGAN Loss with clipping applied after
        self.d_loss = tf.reduce_mean((self.D_) - tf.reduce_mean(self.D))
        self.g_loss = -tf.reduce_mean(self.D_)

        # Collect all variables for generator and discriminator. TF will perform a gradient check if given explicitely to optimizer
        self.g_vars = [var for var in tf.global_variables() if 'generator' in var.name]
        self.d_vars = [var for var in tf.global_variables() if 'discriminator' in var.name]

    def train_model(self, config:dict, samples:list):
        """
        Trains the network - redefined from parent class to add in gradient clipping
        :param config: Configuration dictionary for tuned hyperparameters
        :param samples: Samples to input for training
        """
        # Process for adding in Gradient Penalty to the Discriminator
        alpha = tf.random_uniform(
            shape=[config['batch_size'], 1],
            minval=0.,
            maxval=1.
        )
        interpolates = alpha*self.d + ((1-alpha)*self.D_)
        disc_interpolates = self.discriminator(interpolates, reuse=True)
        gradients = tf.gradients(disc_interpolates, [interpolates])[0]
        slopes = tf.sqrt(tf.reduce_sum(tf.square(gradients), reduction_indices=[1]))
        gradient_penalty = tf.reduce_mean((slopes-1)**2)
        self.d_loss += config["lambda"]*gradient_penalty

        d_opt = tf.train.AdamOptimizer(
                learning_rate=config['learning_rate'],
                beta1=config['beta1'],
                beta2=config['beta2'],
                name='d_opt'
            ).minimize(
                self.d_loss,
                var_list=self.d_vars
            )
        g_opt = tf.train.AdamOptimizer(
                learning_rate=config['learning_rate'],
                beta1=config['beta1'],
                beta2=config['beta2'],
                name='g_opt'
            ).minimize(
                self.g_loss,
                var_list=self.g_vars
            )

        tf.global_variables_initializer().run()

        #Adversarial Training
        counter = 0
        for epoch in range(config['epoch']):
            is_new_epoch = True
            steps = len(samples) // config['batch_size']
            for step in range(steps):

                #Discriminator Training
                for _ in range(config['discriminator_iterations']):
                    r_sample = np.random.choice(len(samples), size=config['batch_size'],
                        replace=True)

                    batch_z = np.random.normal(-1, 1, [config['batch_size'], config['noise_dims']]).astype(np.float32)
                    sample = samples[r_sample]

                    #One hot encode each feature an concatenate into a single input of dim (batch_size, sum(encodings))
                    sample_one_hot = []
                    for i, s in enumerate(sample.T):
                        sample_one_hot.append(one_hot(s, len(self.unique_values[i])))
                    sample_one_hot = np.concatenate(sample_one_hot, axis=1)
                    sample = sample_one_hot

                    _, d_loss, var_save = self.sess.run(
                        [
                            d_opt,
                            self.d_loss,
                            self.merged_summary
                        ],
                        feed_dict = {
                            self.d: sample,
                            self.g: batch_z
                        }
                    )

                r_sample = np.random.choice(len(samples),
                    size=config['batch_size'],
                    replace=False
                )
                #Generator Training
                _, g_loss, var_save = self.sess.run(
                    [
                        g_opt,
                        self.g_loss,
                        self.merged_summary
                    ],
                    feed_dict = {
                        self.g: batch_z
                    }
                )

                #Every 100 steps save the weights and model state
                if counter % 100 == 0:
                    d_loss, g_loss = self.sess.run(
                    [
                        self.d_loss,
                        self.g_loss
                    ],
                    feed_dict = {
                            self.d: sample,
                            self.g: batch_z
                        }
                    )
                    self.writer.add_summary(var_save, counter)
                    self.save_model(counter)

                #Show updated results, append a header if it's a new epoch
                if is_new_epoch:
                    sys.stdout.write("\nEpoch: %d\n" % (epoch))
                    sys.stdout.flush()
                    is_new_epoch = False
                sys.stdout.write(
                    "    Step %d    Discriminator: %g    Generator: %g        \r" %
                    (counter, d_loss, g_loss)
                )
                sys.stdout.flush()

                counter += 1
        print('\n')

class MINE(BaseModel):
    '''
    Mutual Information Neural Estimation model which computes a mutual information
    term through neural network estimation. Can be used to improve GAN output
    '''
    def build_model(self):
        """
        Creates the Mutual Information Estimator Model
        """
        self.x_in = tf.placeholder(tf.float32, shape=[None, self.sample_dim],
            name='noise_data')
        self.y_in = tf.placeholder(tf.float32, shape=[None, self.sample_dim],
            name='real_data')

        # Estimator - Takes in noise and real samples and provides estimates
        self.joint_logits, self.shuffled_logits = self.estimator(self.x_in, self.y_in)

        #Define the loss functions for the model - In this case DCGAN Loss
        self.model_loss()

        #Summary Histograms
        for var in tf.trainable_variables():
            tf.summary.histogram(var.name, var)
        self.merged_summary = tf.summary.merge_all()

        self.m_vars = [var for var in tf.global_variables() if 'estimator' in var.name]

        #Saver and File Writer
        self.saver = tf.train.Saver(max_to_keep=1)
        self.writer = tf.summary.FileWriter(self.logs, self.sess.graph)

    def estimator(self, x, y):
        """
        Generates data through successive approximations of the ground truth
        probability distribution function
        :param x: Input seed for generation
        :param y: Real data distribution to learn
        :return gen_y: Synthetic Data
        """
        with tf.variable_scope("estimator") as scope:

            # Emulates the tensor product between input noise and real samples
            y_shuffled = tf.random_shuffle(y)
            x = tf.concat([x, x], axis=0)
            y = tf.concat([y, y_shuffled], axis=0)

            x_l1 = tf.contrib.layers.fully_connected(
                inputs = x,
                num_outputs = self.h_dim,
                activation_fn = None
            )
            y_l1 = tf.contrib.layers.fully_connected(
                inputs = y,
                num_outputs = self.h_dim,
                activation_fn = None

            )
            joint_l2 = tf.nn.relu(x_l1 + y_l1)
            output = tf.contrib.layers.fully_connected(
                inputs = joint_l2,
                num_outputs = 1,
                activation_fn = None
            )
            batch_size = tf.shape(x)[0]//2
            joint_logits = output[:batch_size]
            shuffled_logits = output[batch_size:]
            return joint_logits, shuffled_logits

    def model_loss(self):
        """
        Parameterizes the loss function to be directly related to the dependence
        between features values
        """
        self.loss = -(tf.reduce_mean(self.joint_logits, axis=0) - tf.log(tf.reduce_mean(tf.exp(self.shuffled_logits))))

    def train_model(self, config:dict, samples_1:list, samples_2:list):
        """
        Trains the network
        :param config: Configuration dictionary for tuned hyperparameters
        :param samples_1: Samples from the first distribution.
        :param samples_2: Samples from the second distribution.
        """
        opt = tf.train.AdamOptimizer(
            learning_rate=config['learning_rate'],
            name='m_opt').minimize(
                self.loss
            )
        tf.global_variables_initializer().run()
        counter = 0
        for epoch in range(config['epoch']):
            is_new_epoch = True
            steps = len(samples_1) // config['batch_size']
            for step in range(steps):
                r_sample = np.random.choice(len(samples_1), size=config['batch_size'],
                    replace=False)
                sample_1 = samples_1[r_sample]
                sample_2 = samples_2[r_sample]

                #One hot encode each feature an concatenate into a single input of dim (batch_size, sum(encodings))
                sample_1_one_hot = []
                for i, s in enumerate(sample_1.T):
                    sample_1_one_hot.append(one_hot(s, len(self.unique_values[i])))
                sample_1_one_hot = np.concatenate(sample_1_one_hot, axis=1)

                #One hot encode each feature an concatenate into a single input of dim (batch_size, sum(encodings))
                sample_2_one_hot = []
                for i, s in enumerate(sample_2.T):
                    sample_2_one_hot.append(one_hot(s, len(self.unique_values[i])))
                sample_2_one_hot = np.concatenate(sample_2_one_hot, axis=1)

                _, loss, var_save = self.sess.run(
                    [
                        opt,
                        self.loss,
                        self.merged_summary
                    ],
                    feed_dict = {
                        self.y_in: sample_1_one_hot,
                        self.x_in: sample_2_one_hot,
                    }
                )

                #Every 100 steps save the weights and model state
                if counter % 100 == 0:
                    loss = self.sess.run(
                        [
                            self.loss
                        ],
                        feed_dict = {
                            self.y_in: sample_1_one_hot,
                            self.x_in: sample_2_one_hot,
                        }
                    )
                    self.writer.add_summary(var_save, counter)
                    self.save_model(counter)

                #Show updated results, append a header if it's a new epoch
                if is_new_epoch:
                    sys.stdout.write("\nEpoch: %d\n" % (epoch))
                    sys.stdout.flush()
                    is_new_epoch = False
                sys.stdout.write(
                    "    Step %d    MINE: %g      \r" %
                    (counter, -loss[0])
                )
                sys.stdout.flush()

                counter += 1
        print('\n')


class WGAN_GPMI(WGAN_GP, MINE):
    '''
    Mutual Information Neural Estimation model for improved GAN operation in the
    the data-limited setting.

    Notes: Adds MI as a loss term to the generator network while also making use
    of the discriminator gradients.

    Computes Frobenius Norm for the generator gradients and weight by the
    min(g_d, g_m) to avoid overemphasizing either term.
    '''

    def build_model(self):
        """
        Creates the Mutual Information Estimator Model
        """
        self.d = tf.placeholder(tf.float32, shape=[None, self.sample_dim],
            name='dis_data')
        self.g = tf.placeholder(tf.float32, shape=[None, self.noise_dim],
            name='gen_data')

        # Generative Network
        self.G_logits = self.generator(self.g)
        # Discrminator Network - Ground-truth input
        self.D, self.D_logits = self.discriminator(self.d)
        # Discrminator Network - Generator input
        self.D_, self.D_logits_ = self.discriminator(self.G_logits, reuse=True)
        # Estimator - Takes in noise and real samples and provides estimates
        self.joint_logits, self.shuffled_logits = self.estimator(self.G_logits, self.g)

        self.model_loss()

        # Collect all variables for generator and discriminator. TF will perform a gradient check if given explicitely to optimizer
        self.g_vars = [var for var in tf.global_variables() if 'generator' in var.name]
        self.d_vars = [var for var in tf.global_variables() if 'discriminator' in var.name]
        self.m_vars = [var for var in tf.global_variables() if 'estimator' in var.name]

        #Saver and File Writer
        self.saver = tf.train.Saver(max_to_keep=1)
        self.writer = tf.summary.FileWriter(self.logs, self.sess.graph)

    def model_loss(self):
        """
        Parameterizes the loss function to be directly related to the dependence
        between features values
        """
        self.m_loss = -(tf.reduce_mean(self.joint_logits) - tf.log(tf.reduce_mean(tf.exp(self.shuffled_logits))))
        self.d_loss = tf.reduce_mean((self.D_) - tf.reduce_mean(self.D))

        self.g_loss_term_1 = -tf.reduce_mean(self.D_)
        self.g_loss_term_2 = self.m_loss
        self.g_loss = self.g_loss_term_1 + self.g_loss_term_2

    def generate_sample(self, x):
        """
        Reconstructs the sample from the given data
        :param x: Input noise to create samples from
        :return: Sample recreated from latent space
        """
        predictions = self.sess.run(self.G_logits,
            feed_dict = {
                self.g: x
            }
        )
        # Take the argmax per feature value to get real valued output
        start_ind = 0
        prediction_indices = np.zeros((predictions.shape[0], len(self.unique_values)))
        for i in range(len(self.unique_values)):
            size = len(self.unique_values[i])
            slice = predictions[:, start_ind:start_ind+size]
            prediction_indices[:,i] = (np.argmax(slice, axis=1))
            start_ind += size
        return prediction_indices

    def train_model(self, config:dict, samples:list):
        """
        Trains the model such that samples are generated from sampling noise with
        the generator, and improved via gradient feedback from the discriminator
        and the mutual information estimate.
        """
        # Process for adding in Gradient Penalty to the Discriminator
        alpha = tf.random_uniform(
            shape=[config['batch_size'], 1],
            minval=0.,
            maxval=1.
        )
        interpolates = alpha*self.d + ((1-alpha)*self.D_)
        disc_interpolates = self.discriminator(interpolates, reuse=True)
        gradients = tf.gradients(disc_interpolates, [interpolates])[0]
        slopes = tf.sqrt(tf.reduce_sum(tf.square(gradients), reduction_indices=[1]))
        gradient_penalty = tf.reduce_mean((slopes-1)**2)
        self.d_loss += config["lambda"]*gradient_penalty

        m_opt = tf.train.AdamOptimizer(
            learning_rate=config['learning_rate'],
            name='m_opt').minimize(
                self.m_loss,
                var_list=self.m_vars
            )
        d_opt = tf.train.AdamOptimizer(
                learning_rate=config['learning_rate'],
                beta1=config['beta1'],
                beta2=config['beta2'],
                name='d_opt'
            ).minimize(
                self.d_loss,
                var_list=self.d_vars
            )
        g_opt = tf.train.AdamOptimizer(
                learning_rate=config['learning_rate'],
                beta1=config['beta1'],
                beta2=config['beta2'],
                name='g_opt'
            )

        # Compute the gradients w.r.t. the generator parameters
        grad_g = g_opt.compute_gradients(self.g_loss_term_1, self.g_vars)
        grad_m = g_opt.compute_gradients(self.g_loss_term_2, self.g_vars)

        # Normalize and find the minimum
        grad_g_norm = [(tf.norm(grad), var) for grad, var in grad_g]
        grad_m_norm = [(tf.norm(grad), var) for grad, var in grad_m]
        min_term = [tf.minimum(g_norm, m_norm) for (g_norm, _), (m_norm, _) in zip(grad_g_norm, grad_m_norm)]

        # Compute the Frobenius Norm for the gradients resultant from MI term
        f_norm = [(grad / norm , var) for (grad, _), (norm, var) in zip(grad_m, grad_m_norm)]
        grad_normed = [(min_t * norm, var) for min_t, (norm, var) in zip(min_term, f_norm)]

        # Add the normalized MI gradient to the
        grad_g_fin = [(grad_g + grad_n, var_g) for (grad_g, var_g), (grad_n, var_n) in zip(grad_g, grad_normed)]

        # Apply the gradients
        g_opt = g_opt.apply_gradients(grad_g_fin)

        # # Gradient Histograms - Debugging
        # for (grad_g, var_g), (grad_m, var_m) in zip(grad_g_norm, grad_m_norm):
        #     tf.summary.histogram(var_g.name + '_trad_gradient', grad_g)
        #     tf.summary.histogram(var_m.name + '_mi_gradient', grad_m)
        #
        # for grad_n,var_n in grad_normed:
        #     tf.summary.histogram(var_n.name + '_norm_gradient', grad_n)

        # Summary Histograms
        for var in tf.trainable_variables():
            tf.summary.histogram(var.name, var)
        self.merged_summary = tf.summary.merge_all()

        tf.global_variables_initializer().run()

        #Adversarial Training
        counter = 0
        for epoch in range(config['epoch']):
            is_new_epoch = True
            steps = len(samples) // config['batch_size']
            for step in range(steps):

                #Discriminator Training
                for _ in range(config['discriminator_iterations']):
                    r_sample = np.random.choice(len(samples), size=config['batch_size'],
                        replace=True)
                    batch_z = np.random.normal(-1, 1, [config['batch_size'], config['noise_dims']]).astype(np.float32)
                    sample = samples[r_sample]

                    # One hot encode each feature an concatenate into a single input of dim (batch_size, sum(encodings))
                    sample_one_hot = []
                    for i, s in enumerate(sample.T):
                        sample_one_hot.append(one_hot(s, len(self.unique_values[i])))
                    sample_one_hot = np.concatenate(sample_one_hot, axis=1)
                    sample = sample_one_hot

                    _, d_loss, var_save = self.sess.run(
                        [
                            d_opt,
                            self.d_loss,
                            self.merged_summary
                        ],
                        feed_dict = {
                            self.d: sample,
                            self.g: batch_z
                        }
                    )

                # # Generator Training
                r_sample = np.random.choice(len(samples), size=config['batch_size'],
                    replace=False)
                # Random noise is now split between uniform, incompressible data.
                batch_z = np.random.normal(-1, 1, [config['batch_size'], config['noise_dims']]).astype(np.float32)
                sample = samples[r_sample]

                # One hot encode each feature an concatenate into a single input of dim (batch_size, sum(encodings))
                sample_one_hot = []
                for i, s in enumerate(sample.T):
                    sample_one_hot.append(one_hot(s, len(self.unique_values[i])))
                sample_one_hot = np.concatenate(sample_one_hot, axis=1)
                sample = sample_one_hot

                _, g_loss, var_save = self.sess.run(
                    [
                        g_opt,
                        self.g_loss,
                        self.merged_summary
                    ],
                    feed_dict = {
                        self.d: sample,
                        self.g: batch_z
                    }
                )

                # Mutual Information Training
                _, m_loss, var_save = self.sess.run(
                    [
                        m_opt,
                        self.m_loss,
                        self.merged_summary
                    ],
                    feed_dict = {
                        self.d: sample,
                        self.g: batch_z
                    }
                )

                #Every 100 steps save the weights and model state
                if counter % 100 == 0:
                    d_loss, g_loss, m_loss = self.sess.run(
                    [
                        self.d_loss,
                        self.g_loss,
                        self.m_loss
                    ],
                    feed_dict = {
                            self.d: sample,
                            self.g: batch_z
                        }
                    )
                    self.save_model(counter)
                    self.writer.add_summary(var_save, counter)

                #Show updated results, append a header if it's a new epoch
                if is_new_epoch:
                    sys.stdout.write("\nEpoch: %d\n" % (epoch))
                    sys.stdout.flush()
                    is_new_epoch = False
                sys.stdout.write(
                    "    Step %d    Discriminator: %g    Generator: %g    MINE: %g          \r" %
                    (counter, d_loss, g_loss, m_loss)
                )

                sys.stdout.flush()

                counter += 1
        print('\n')

def main():
    """
    Run the given samples through the given network
    """
    print("TensorFlow Version: %s" % tf.__version__)

    #Create hyperparameters for training/generating configuration
    config = {}
    config['epoch'] = 300
    config['batch_size'] = 100
    config['learning_rate'] = 5e-4
    config['lambda'] = 0.4
    config['beta1'] = 0.5
    config['beta2'] = 0.8
    config['noise_dims'] = 64
    config['discriminator_iterations'] = 5

    alerts, labels, unique_values, unique_dest_ips, _, cuts = preprocess_data_by_target(is_stage_conditioned=True)

    alerts_on_target = []

    for target in range(max(labels)):
        # Only train on select victims

        # CPTC 18 targets
        # if unique_dest_ips[target] in ['10.0.0.24', '10.0.1.5', '10.0.0.22', '10.0.1.46']:

        # CPTC 17 targets
        if unique_dest_ips[target] in ['10.0.0.100', '10.0.0.22', '10.0.0.27', '10.0.99.143']:
            print('\n\nTARGET: %s\n' % (unique_dest_ips[target]))

            # TODO: Could be replaced by logical indexing
            # Separates out the alerts per target IP Address
            alerts_on_target = []
            for ind, l in enumerate(labels):
                if l == target:
                    alerts_on_target.append(alerts[ind])
            feature_combos_gt = get_feature_combinations(np.asarray(alerts_on_target))
            alerts_arr = np.asarray(alerts_on_target)

            # Create the TensorFlow Session, instantiate the model object we're training with
            # It's important to change the checkpoints dir to reflect the directory where the checkpoints are stored.
            # Similarly the name is checked when loading previous models
            tf.reset_default_graph()
            with tf.Session() as gan_sess:
                gan = WGAN_GPMI(
                    gan_sess,
                    sample_dim=sum([len(uv) for uv in unique_values]),
                    h_dim=H_DIM,
                    num_classes=1,
                    unique_values=unique_values,
                    checkpoint_dir='./checkpoints/%s' % (unique_dest_ips[target]),
                    name="WGAN_GPMI"
                )

                # Build the model architecture, then try to load prevoius instance.
                # If there is a prevous checkpoint with the same architecture it will be loaded
                # If there is no prevous checkpoint or the architecture has changed, retrain the model
                gan.build_model()
                try:
                    if not gan.load_model():
                        gan.train_model(config, alerts_arr)
                # Model checkpoint was for an older architecture, overwrite it
                except (tf.errors.InvalidArgumentError, tf.errors.NotFoundError) as e:
                    gan.train_model(config, alerts_arr)

                i = 1
                target_team = [1]
                for team in target_team:
                    # The dimensionality of the input noise determines the size of the outputself.
                    # With shape [A,B] A determines the number of samples generated and B must equal self.noise_dim in the model
                    input_noise = np.random.normal(-1, 1, [len(alerts_on_target), 64])
                    generated_alerts = gan.generate_sample(input_noise)

                    ################ Insert Analysis Code ##################
                    generated_alerts = generated_alerts.astype(int)
                    feature_combos_gen = get_feature_combinations(np.asarray(generated_alerts))


if __name__ == '__main__':
    main()
