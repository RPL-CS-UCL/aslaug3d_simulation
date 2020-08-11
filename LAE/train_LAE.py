import numpy as np    # Standard matrix operations
import random         # Randomizing functions
import sys, os
import time
import tensorflow as tf
tf.enable_eager_execution()
import argparse
#device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
device = "cpu"
sys.path.insert(0, "../")
sys.path.insert(0, "../../")


class Conv1DTranspose(tf.keras.layers.Layer):
    def __init__(self, filters, kernel_size, strides=1, padding='valid'):
        super().__init__()
        self.conv2dtranspose = tf.keras.layers.Conv2DTranspose(
          filters, (kernel_size, 1), (strides, 1), padding
        )

    def call(self, x):
        x = tf.expand_dims(x, axis=2)
        x = self.conv2dtranspose(x)
        x = tf.squeeze(x, axis=2)
        return x

class CVAE(tf.keras.Model):
  """Convolutional variational autoencoder."""

  def __init__(self, latent_dim, capacity=8):
    super(CVAE, self).__init__()
    self.latent_dim = latent_dim
    c = capacity
    n_scans = 402
    k1 = 11
    k2 = 7 

    self.encoder = tf.keras.Sequential(
        [
          tf.keras.layers.InputLayer(input_shape=(n_scans,1)),
          # Shape B x 402 x 1
          tf.keras.layers.Conv1D(
            filters=c, kernel_size=k1, 
            activation='relu'),
          # Shape B x 392 x C
          tf.keras.layers.Conv1D(
              filters=2*c, kernel_size=k2, activation='relu'),
          tf.keras.layers.Flatten(),
          # No activation
          tf.keras.layers.Dense(latent_dim + latent_dim),
        ]
    )

    self.decoder = tf.keras.Sequential(
        [
            tf.keras.layers.InputLayer(input_shape=(latent_dim,)),
            tf.keras.layers.Dense(units=2*c*(n_scans), activation=tf.nn.relu),
            tf.keras.layers.Reshape(target_shape=(n_scans, 2*c)),
            Conv1DTranspose(
                filters=2*c, kernel_size=k2, padding='same'),
            Conv1DTranspose(
                filters=c, kernel_size=k1, padding='same'),
            # No activation
            Conv1DTranspose(
                filters=1, kernel_size=3, padding='same'),
        ]
    )

  def sample(self, eps=None):
    if eps is None:
      eps = tf.random.normal(shape=(100, self.latent_dim))
    return self.decode(eps, apply_sigmoid=True)

  def encode(self, x):
    mean, logvar = tf.split(self.encoder(x), num_or_size_splits=2, axis=1)
    return mean, logvar

  def reparameterize(self, mean, logvar):
    eps = tf.random.normal(shape=mean.shape)
    return eps * tf.exp(logvar * .5) + mean

  def decode(self, z, apply_sigmoid=False):
    logits = self.decoder(z)
    if apply_sigmoid:
      probs = tf.sigmoid(logits)
      return probs
    return logits







def log_normal_pdf(sample, mean, logvar, raxis=1):
  log2pi = tf.math.log(2. * np.pi)
  return tf.reduce_sum(
      -.5 * ((sample - mean) ** 2. * tf.exp(-logvar) + logvar + log2pi),
      axis=raxis)


def compute_loss(model, x):
  mean, logvar = model.encode(x)
  z = model.reparameterize(mean, logvar)
  x_logit = model.decode(z)
  cross_ent = tf.nn.sigmoid_cross_entropy_with_logits(logits=x_logit, labels=x)
  logpx_z = -tf.reduce_sum(cross_ent, axis=[1, 2])
  logpz = log_normal_pdf(z, 0., 0.)
  logqz_x = log_normal_pdf(z, mean, logvar)
  return -tf.reduce_mean(logpx_z + logpz - logqz_x)

def train_step(model, x, optimizer):
  """Executes one training step and returns the loss.

  This function computes the loss and gradients, and uses the latter to
  update the model's parameters.
  """
  with tf.GradientTape() as tape:
    loss = compute_loss(model, x)

  gradients = tape.gradient(loss, model.trainable_variables)
  optimizer.apply_gradients(zip(gradients, model.trainable_variables))
  #optimizer.minimize(loss, var_list= model.trainable_variables)




optimizer = tf.compat.v1.train.AdamOptimizer(1e-4)
#tf.keras.optimizers.Adam(1e-4)


epochs = 1
# set the dimensionality of the latent space to a plane for visualization later
latent_dim = 128
num_examples_to_generate = 16

# keeping the random vector constant for generation (prediction) so
# it will be easier to see the improvement.
random_vector_for_generation = tf.random.normal(
    shape=[num_examples_to_generate, latent_dim])

model = CVAE(latent_dim)

dataset = np.load("./data/lidar_data_both.npy")
dataset = dataset.astype('float32')
dataset = dataset.reshape((dataset.shape[0], 402, 1 ))

split_n = dataset.shape[0]*9//10
train_images = dataset[:split_n]
test_images = dataset[split_n:]
train_size = train_images.shape[0]
test_size = test_images.shape[0]
batch_size = 25
train_dataset = (tf.data.Dataset.from_tensor_slices(
  train_images)
  .shuffle(train_size)
  .batch(batch_size))

test_dataset = (tf.data.Dataset.from_tensor_slices(test_images)
                .shuffle(test_size).batch(batch_size))
for test_batch in train_dataset.take(1):
  test_sample = test_batch


def generate_and_save_images(model, epoch, test_sample):
  mean, logvar = model.encode(test_sample)
  z = model.reparameterize(mean, logvar)
  predictions = model.sample(z)


for epoch in range(1, epochs + 1):
  start_time = time.time()
  for i, train_x in enumerate(train_dataset):
    train_step(model, train_x, optimizer)
    if i%1 == 0:
      print ("Train step {:}.{:}"
        .format(epoch, i))
  end_time = time.time()

  loss = tf.keras.metrics.Mean()
  for test_x in test_dataset:
    loss(compute_loss(model, test_x))
  elbo = -loss.result()
  print('Epoch: {}, Test set ELBO: {}, time elapse for current epoch: {}'
        .format(epoch, elbo, end_time - start_time))
  
