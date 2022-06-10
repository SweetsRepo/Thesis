'''
The Graveyard: Where mediocore/uneeded code goes to die
'''
####################### VAE Model ###########################
class VAE(BaseModel):
    '''
    Variational Auto Encoder:

    Encoder Decoder framework for learning latent space representations of the
    input data. Can be used as a generative model by sampling batches of points
    from this latent space.

    https://arxiv.org/pdf/1312.6114.pdf
    https://arxiv.org/pdf/1606.05908.pdf
    '''

    def __init__(self, sess, sample_dim:int, h_dim:int, num_classes:int=0, unique_values=None, name='VAE'):
        """
        Creates a new Variational Auto-Encoder
        :param sess: TensorFlow Session
        :param sample_dim: Features in the data container
        :param h_dim: Hidden Dimension used for NNs
        :param num_classes:  Dimension of the generator/gt output
        :param unique_values: List of lists containing the unique_values of the dataset on a per feature basis

        Note: num_clusters doesn't matter for unsupervised models like VAE
        """
        super().__init__(sess, sample_dim, h_dim, num_classes, name=name)
        self.unique_values = unique_values

    def build_model(self):
        """
        Builds the tensor graph, sets up logging, and defines loss functions
        for latent and reconstructed representations
        """
        #Input/Output - Full representation
        self.x = tf.placeholder(tf.float32, shape=[None, self.sample_dim],
                                name='input_sample')

        self.z = tf.placeholder(tf.float32, shape=[None, self.h_dim],
                                name='noise_z')

        self.Z_mu, self.Z_std = self.encoder(self.x)

        eps = tf.random_normal(tf.shape(self.Z_mu))

        #Intermediate - Latent representation
        z = self.Z_mu + self.Z_std * eps

        self.Y, self.Y_Logits = self.decoder(z)

        #Added in 1e-6 to avoid evaluating log(0)
        self.x_loss = tf.reduce_sum(self.x * tf.log(1e-6 + self.Y) +\
            (1 - self.x) * tf.log(1e-6 + 1 - self.Y), 1)
        #KL Divergence of the data?
        self.z_loss = 0.5 * tf.reduce_sum(tf.square(self.Z_mu) +\
            tf.square(self.Z_std) - tf.log(1e-6 + tf.square(self.Z_std)) - 1, 1)
        #Reduce mean for losses and calculate ELBO
        self.x_loss = tf.reduce_mean(self.x_loss)
        self.z_loss = tf.reduce_mean(self.z_loss)
        self.total_loss = -(self.x_loss - self.z_loss)

        #Summary Scalars & Histograms
        self.x_loss_sum = tf.summary.scalar("x_loss", self.x_loss)
        self.z_loss_sum = tf.summary.scalar("z_loss", self.z_loss)
        self.total_loss_sum = tf.summary.scalar("total_loss", self.total_loss)
        for var in tf.trainable_variables():
            tf.summary.histogram(var.name, var)
        self.merged_summary = tf.summary.merge_all()

        #Saver and File Writer
        self.saver = tf.train.Saver(max_to_keep=1)
        self.writer = tf.summary.FileWriter(self.logs, self.sess.graph)

    def train_model(self, config:dict, data:list):
        """
        Trains the given model end to end
        :param config: Configuration dictionary for tunable hyperparameters
        :param data: Training data to learn embedded representation of
        """
        opt = tf.train.AdamOptimizer(learning_rate=config['learning_rate'],
                                     beta1=config['beta1'],
                                     name='opt').minimize(self.total_loss)

        tf.global_variables_initializer().run()

        counter = 0
        for epoch in range(config['epoch']):
            step = 0
            while step+config['batch_size'] < len(data):
                # Feed sequences in order
                sequences_to_use = data[step:step+config['batch_size']]
                #Acquire cost and summary str
                _, tcost, zcost, xcost, summary_str, var_save = self.sess.run(
                    [
                        opt,
                        self.total_loss,
                        self.z_loss,
                        self.x_loss,
                        self.total_loss_sum,
                        self.merged_summary
                    ],
                    feed_dict = {
                        self.x : sequences_to_use[:][0]
                        }
                )
                counter += 1
                if counter % 100 == 0:
                    self.writer.add_summary(var_save, counter)
                    print("Step %d\n Loss: %f : %f : %f" % (
                        counter,
                        xcost,
                        zcost,
                        tcost
                    )
                )
                step += config['batch_size']

    def transform_sample(self, x):
        """
        Transforms the data by mapping it to latent space
        This maps to the mean of the distribution, could sample guassian instead
        :param X: Input to map into latent space
        :return: Sample transformed into latent space
        """
        return self.sess.run(self.Z_mu, feed_dict={self.x: x})

    def generate_sample(self, x):
        """
        Reconstructs the sample from the given data
        :param x: Random normal noise to generate sample from
        :return: Sample recreated from latent space
        """
        predictions = []
        for i, key in enumerate(FEATURE_EMBEDDING_SIZE):
            with tf.name_scope(key[0]):
                size = 1 if key == 'timestamp' else len(self.unique_values[i])
                pred = tf.contrib.layers.fully_connected(
                    inputs = self.Y_Logits,
                    num_outputs = size,
                    activation_fn = tf.nn.softmax
                )
                index = np.argmax(self.sess.run(pred, feed_dict={pred: x}), axis=1)
                sample = []
                for ind in index:
                    sample.append(self.unique_values[i][ind])
                predictions.append(sample)
        return predictions

    def encoder(self, x):
        """
        Provides the tensor mapping from input x to latent representation z
        :param x: Real valued input
        :return mu, std: Values representing latent mean and standard deviation
        """
        with tf.variable_scope("encoder") as scope:
            #Begin Encoder Network
            enc_l1 = tf.contrib.layers.fully_connected(
                inputs = x,
                num_outputs = self.h_dim
            )
            enc_l2 = tf.contrib.layers.fully_connected(
                inputs = enc_l1,
                num_outputs = self.h_dim
            )
            enc_y = tf.contrib.layers.fully_connected(
                inputs = enc_l2,
                num_outputs = 2*self.h_dim,
                activation_fn = None
            )
            mu = enc_y[:, :self.h_dim]
            std = 1e-6 + tf.nn.softplus(enc_y[:, self.h_dim:])
            return mu, std

    def decoder(self, z):
        """
        Provides the tensor mapping from latent representation z to output x
        :param tensor: Latent space input
        :param reuse: Determines if variables within scope are reused or not.
        :return: Value within the range {0,1} representing real valued mean
        """
        with tf.variable_scope("decoder") as scope:
            dec_l1 = tf.contrib.layers.fully_connected(
                inputs = z,
                num_outputs = self.h_dim
            )
            dec_l2 = tf.contrib.layers.fully_connected(
                inputs = dec_l1,
                num_outputs = self.h_dim
            )
            dec_y = tf.contrib.layers.fully_connected(
                inputs = dec_l2,
                num_outputs = self.sample_dim,
                activation_fn = None
            )
            return tf.nn.softmax(dec_y), dec_y
##############################################################

######################## RNN Model ###########################
class RNN(BaseModel):
    '''
    Recurrent Neural Network adds in temporal locality of data. Can be used to
    generate sequences given a seed. LSTM cells determine how much data to
    remember from a sequence in order to make accurate predictions.

    '''

    def __init__(self, sess, sample_dim:int, h_dim:int, num_classes:int, longest_seq:int, unique_values=None, name='VAE'):
        """
        Creates a new LSTM-based RNN
        :param sess: TensorFlow Session
        :param sample_dim: Features in the data container
        :param h_dim: Hidden Dimension used for NNs
        :param num_classes:  Dimension of the generator/gt output
        :param longest_seq: Longest sequence for RNN Cell to expect

        """
        super().__init__(sess, sample_dim, h_dim, num_classes, name=name)
        self.longest_seq = longest_seq
        self.unique_values = unique_values

    def build_model(self):
        '''
        Builds the network. Makes use of TensorFlow Embedding layers and
        BasicLSTM Cell.
        '''
        #Inputs, Outputs, and Sequence Length for Internal Usage
        self.x = tf.placeholder(tf.int32, [None, self.longest_seq, self.sample_dim], name='input')
        self.y = tf.placeholder(tf.int32, [None, WIN_SIZE, len(FEATURE_EMBEDDING_SIZE)], name = "label")
        # self.y_t = tf.transpose(self.y, perm=[0, 2, 1])
        self.seq_len = tf.placeholder(tf.int32, [None])

        #24 Timesteps, each with 7 Fully Connected Layers for Feature Output
        self.LSTM, self.LSTM_logits = self.rnn(self.x)
        # print("[%d,%d]"%(len(self.LSTM), len(self.LSTM[0])))

        self.model_loss()

        #Add in summary histogram
        for var in tf.trainable_variables():
            tf.summary.histogram(var.name, var)
        self.merged_summary = tf.summary.merge_all()

        #Saver and File Writer
        self.saver = tf.train.Saver(max_to_keep=1)
        self.writer = tf.summary.FileWriter(self.logs, self.sess.graph)

    def model_loss(self):
        #Compute loss as the sum of the loss from each feature
        self.total_loss = 0
        for t in range(WIN_SIZE):
            for i, key in enumerate(FEATURE_EMBEDDING_SIZE):
                # if key[0] == 'timestamp':
                #     self.total_loss += tf.losses.mean_squared_error(self.y[:,t,i], tf.squeeze(self.LSTM_logits[t][i]))
                # else:
                self.total_loss += tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=self.LSTM_logits[t][i], labels=self.y[:,t,i]))


    def rnn(self, x):
        '''
        LSTM-based network with embedded input from categorical features.
        :param x: Input feature sample
        :return LSTM, LSTM_logits: Feature prediction outputs and logits
        '''
        with tf.variable_scope("rnn") as scope:
            num_features = len(FEATURE_EMBEDDING_SIZE)
            #Once again, credit where it's due. Ian's code for embedding sequences
            features_squeezed = tf.reshape(tf.transpose(x, perm=[2,0,1]), [num_features, -1])
            tf_features = tf.split(features_squeezed, [1 for _ in range(num_features)], 0)
            on_val = tf.constant(1.0)
            off_val = tf.constant(0.0)

            for i, tf_feature in enumerate(tf_features):
                if FEATURE_EMBEDDING_SIZE[i][1] != -1:
                    tf_features[i] = tf.contrib.layers.embed_sequence(tf.squeeze(tf_feature), len(self.unique_values[i]), embed_dim=FEATURE_EMBEDDING_SIZE[i][1])
                else:
                    tf_features[i] = tf.one_hot(tf.squeeze(tf_feature), len(self.unique_values[i]), on_val, off_val)
            for i, tf_feature in enumerate(tf_features):
                if FEATURE_EMBEDDING_SIZE[i][1] != -1:
                    tf_features[i] = tf.reshape(tf_feature, [-1, WIN_SIZE, FEATURE_EMBEDDING_SIZE[i][1]])
                else:
                    tf_features[i] = tf.reshape(tf_feature, [-1, WIN_SIZE, len(self.unique_values[i])])

            x = tf.concat(tf_features, 2, name="embedded_concat")
            rnn_output, _ = tf.nn.dynamic_rnn(
                tf.contrib.rnn.BasicLSTMCell(self.h_dim),
                x,
                dtype = tf.float32,
                sequence_length = self.seq_len
            )

            y = []
            y_logits = []
            for t in range(WIN_SIZE):
                # Group all samples into timesteps
                y_at_t = []
                y_logits_at_t = []
                for i, key in enumerate(FEATURE_EMBEDDING_SIZE):
                    size = len(self.unique_values[i])
                    entry = tf.contrib.layers.fully_connected(
                        inputs = rnn_output[:,t],
                        num_outputs = size,
                        activation_fn = None
                    )
                    y_at_t.append(entry)
                    y_logits_at_t.append(tf.nn.softmax(entry))
                y_logits.append(y_at_t)
                y.append(y_logits_at_t)
            return y, y_logits

    def train_model(self, config:dict, data:list, labels:list):
        '''
        Train the network using settings in config
        :param config: Hyperparameter settings to use during training
        :param samples: Input sequences to train on
        '''
        opt = tf.train.AdamOptimizer(learning_rate=config['learning_rate'],
                                     beta1=config['beta1'],
                                     name='opt').minimize(self.total_loss)

        tf.global_variables_initializer().run()

        counter = 0
        for epoch in range(config['epoch']):
            is_new_epoch = True
            step = 0
            next_step = step+config['batch_size']
            while next_step < len(data):
                sequences_to_use = data[step:next_step]
                labels_to_use = labels[step:next_step]
                uniform_seq_size = [WIN_SIZE for _ in range(config['batch_size'])]
                _, loss, var_saved = self.sess.run(
                    [
                        opt,
                        self.total_loss,
                        self.merged_summary
                    ],
                    feed_dict = {
                        self.x: sequences_to_use,
                        self.y: labels_to_use,
                        self.seq_len: uniform_seq_size
                    }
                )
                if counter % 100 == 0:
                    loss, var_save = self.sess.run(
                        [
                            self.total_loss,
                            self.merged_summary
                        ],
                        feed_dict = {
                            self.x: sequences_to_use,
                            self.y: labels_to_use,
                            self.seq_len: uniform_seq_size
                        }
                    )
                    self.writer.add_summary(var_save, counter)
                    self.save_model(counter)
                #Show updated results, append a header if it's a new epoch
                if is_new_epoch:
                    sys.stdout.write("\nEpoch: %d\n" % (epoch))
                    sys.stdout.flush()
                    is_new_epoch = False
                try:
                    sys.stdout.write(
                        "    Step %d    Loss: %g        \r" %
                        (counter, loss)
                    )
                    sys.stdout.flush()
                #TODO: Why is the loss getting packed into a single value list
                except TypeError as e:
                    loss = loss[0]
                    sys.stdout.write(
                        "    Step %d    Loss: %g        \r" %
                        (counter, loss)
                    )
                    sys.stdout.flush()
                counter += 1
                step += config['batch_size']
                next_step +=config['batch_size']
        print('\n')

    def generate_sequence(self, x, length):
        '''
        Given a starting sequence x, generate a predicted sequence
        :param x: Input sequence to seed prediction with
        :param length: Number of alerts to try and generate after the initial sequence
        :return sequence: Output sequence predicted from RNN
        :notes: Can this be adapted to take in noise like a GAN?
        '''
        generated_sequences = []
        for i in range(len(self.LSTM)):
            timestep = self.sess.run(
                [
                    self.LSTM[i]
                ],
                feed_dict={
                    self.x: x,
                    self.seq_len: length
                }
            )
            #For some reason it dumps everything into an array of cardinality 1
            generated_sequences.append(timestep[0])
        return generated_sequences
###################################################################
