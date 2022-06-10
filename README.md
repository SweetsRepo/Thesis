# Sweets Thesis Repository
GANs, Info Theory, and more Statistics than I care to admit

## Notes About this Repository
This section of the Deep Learning repository consists of Christopher Sweet's work on Generative Adversarial Modeling of Cyber-Intrusion Alerts. This is meant to be a stable commit of the work at the end of 2018-2019 Academic year, and has been cleaned of miscellaneous files, scripts, etc. This repository is a clean point to begin new development from.

## Requirements
* All testing has been performed in Ubuntu 16.04 LTS
* Python 3.5.3 (or newer)
* Tensorflow 1.8 (More on this below)
* See requirements.txt file for all other packages

## Overview
First navigate to the home directory of this repository and install all required packages:
```bash
pip install -r requirements.txt
```

There are three files in this repository that are important for future experimentation:
* models.py - Contains all the machine learning models using a standardized API
* utilities.py - All preprocessing, functions to read in dataset, and useful transformations
* graveyard.py - Old code that was cluttering up any of these other files and snippets of analysis that can be added to the main at the bottom of models.py

Make any required modifications to models.py main function to change models or analysis. Note that analysis can almost always be added after the comment below:

```python
############## Insert analysis code here ###############
```

If you need to change the models themselves then review the models that have already been implemented. These models make heavy use of object orientation and inheritance and follow a structure similar to what is given below:

```
-BaseModel
 |-MINE
 |  |-MINEGAN
 |-GAN
 |  |-WGAN
 |  |-WGAN_GP
 |    |-MINEGAN
 |-VAE*
 |-RNN*
```

Note that MINEGAN is listed twice. This model makes use of multiple inheritance and is referred to as WGAN-GPMI in literature.

Models with an * have been moved to graveyard.py as they were no longer useful for experimentation, but were running when stored.

Additionally, each model contains the following standardized functions:

* load_model - Loads previous checkpoint files from the model
* save_model - Saves the current iteration of training for the model. This is called as part of training
* model_loss - Defines the loss function of the model and stores it as an object variable. (e.g. self.\*)
* build_model - Defines the model architecture at a high level (inputs, neural networks used, outputs, and variables for saving, logging, etc)
* train_model - Trains the model by first specifying the optimizer to use, setting up inputs per training step, and looping through the prescribed number of epochs. Example given below.
* generate_sample - Synthesizes samples for a given input afte the model has been trained.
* generator/discriminator - Model defining functions. These create the specific TensorFlow based architectures used in build_model.

To run the code type the following in a terminal:
```bash
python models.py
```

If you would like to view tensorboard type in the following command:
```bash
tensorboard --logdir=./logs
```

### Model Training
Model training is encapsulated in the train model function and is broken down as follows:

```python
# Creates the optimizer object and tells it to minimize the loss created in model_loss
# Additionally, rounding up all the variables that can be changed during training and feeding them into
# the var_list argument allows TF to perform a check to make sure that all modifiable variables are back-propagable.
opt = tf.train.Optimizer(learning_rate).minimize(self.loss, var_list=self.vars)

# TF initializing all variables, interfaces with the C backend
tf.global_variables_initializer().run()

# Training loops - 1 for the number of epochs and 1 for each batch
for epoch in range(config['epoch']):
    steps = len(samples) // config['batch_size']
    for step in range(steps):

      # Pull random samples for training without replacement
      r_sample = np.random.choice(len(samples), size=config['batch_size'],
                  replace=False)
      # Create noise sample for Generator
      batch_z = np.random.normal(-1, 1, [config['batch_size'], config['noise_dims']]).astype(np.float32)
      sample = samples[r_sample]

      # One hot encode each feature an concatenate into a single input of dim (batch_size, sum(encodings))
      # If guiding generator behavior based off inputs, this is where you would add your other forms of
      # noise and concatenate it
      sample_one_hot = []
      for i, s in enumerate(sample.T):
          sample_one_hot.append(one_hot(s, len(self.unique_values[i])))
      sample_one_hot = np.concatenate(sample_one_hot, axis=1)
      sample = sample_one_hot

      # Train the Discriminator
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

        # Train the Generator
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

        # Every 100 steps save the model and save to the log file
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
```


## Quirks
* Configuration of training elements are defined in the config dictionary contained in the main function under models.py
* Configuration of latent network aspects like hidden dimensions are globals at the top of the utilities file.


## Known Bugs/Design Choices
* Training will randomly stop at times via an error. Seems to be related to the host machine trying to go to sleep. May also be a memory leak that existed in earlier versions of tensorflow.
* While we're talking about that, none of this code has been verified for operation past TF 1.8.
* TF 2.0 brings about many changes to execution state, be wary of updating (though probably worth the time eventually)
* Updating will require changes to the model structure (contrib library removal/changes) and state saving/loading from checkpoints
* When preprocessing the data on a per team basis the timebins are localized (meaning each team's timebin may encompass a different range than others)
