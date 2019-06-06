# coding=utf-
import matplotlib
matplotlib.use('Agg')
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import os
import time
import pathlib
from IPython import display

from generator import make_generator_model
from preprocessor import preprocess_image
from discriminator import make_discriminator_model
from generator import make_generator_model
from utils import parse_arguments
from utils import ensure_dirs
print(tf.__version__)

##### PARAMETERS ##### 
FLAGS, unparsed = parse_arguments()
TEST_NAME = FLAGS.name
EPOCHS = FLAGS.epochs
BATCH_SIZE = FLAGS.batch_size
DATASET_SIZE = FLAGS.dataset_size
noise_dim = 100
num_examples_to_generate = 16

# Récupération des fichiers images
images = [str(path) for path in list(pathlib.Path("datasets/datanime").glob('*'))]

# Création d'un tf.Dataset à partir des fichiers images
paths_ds = tf.data.Dataset.from_tensor_slices(images)

# Appel de la fonction de préprocessing sur le dataset
images_ds = paths_ds.map(preprocess_image, num_parallel_calls=tf.data.experimental.AUTOTUNE).shuffle(DATASET_SIZE).batch(BATCH_SIZE)


# Creation du générateur
generator = make_generator_model()

# Génération du bruit
noise = tf.random.normal([1, 100])
generated_image = generator(noise, training=False)
# plt.imshow(generated_image[0, :, :, 0], cmap='gray')

discriminator = make_discriminator_model()
decision = discriminator(generated_image)
print (decision)


# Calcul la valeur du loss
cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)

def generator_loss(fake_output):
    return cross_entropy(tf.ones_like(fake_output), fake_output)

def discriminator_loss(real_output, fake_output):
    real_loss = cross_entropy(tf.ones_like(real_output), real_output)
    fake_loss = cross_entropy(tf.zeros_like(fake_output), fake_output)
    total_loss = real_loss + fake_loss
    return total_loss

generator_optimizer = tf.keras.optimizers.Adam(1e-4)
discriminator_optimizer = tf.keras.optimizers.Adam(1e-4)

checkpoint_dir = './training_checkpoints/dcgan/{}'.format(TEST_NAME)
checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")
checkpoint = tf.train.Checkpoint(generator_optimizer=generator_optimizer,
                                 discriminator_optimizer=discriminator_optimizer,
                                 generator=generator,
                                 discriminator=discriminator)

test_output_dir = 'tests/dcgan/{}'.format(TEST_NAME)

# Check that required directories are present and created
ensure_dirs(test_output_dir, checkpoint_dir)

# We will reuse this seed overtime (so it's easier)
# to visualize progress in the animated GIF)
seed = tf.random.normal([num_examples_to_generate, noise_dim])


# Notice the use of <a href="../../../versions/r2.0/api_docs/python/tf/function"><code>tf.function</code></a>
# This annotation causes the function to be "compiled".
@tf.function
def train_step(images):
    noise = tf.random.normal([BATCH_SIZE, noise_dim])

    with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
      generated_images = generator(noise, training=True)

      real_output = discriminator(images, training=True)
      fake_output = discriminator(generated_images, training=True)

      gen_loss = generator_loss(fake_output)
      disc_loss = discriminator_loss(real_output, fake_output)

    gradients_of_generator = gen_tape.gradient(gen_loss, generator.trainable_variables)
    gradients_of_discriminator = disc_tape.gradient(disc_loss, discriminator.trainable_variables)

    generator_optimizer.apply_gradients(zip(gradients_of_generator, generator.trainable_variables))
    discriminator_optimizer.apply_gradients(zip(gradients_of_discriminator, discriminator.trainable_variables))


def generate_and_save_images(model, epoch, test_input):
  # Notice `training` is set to False.
  # This is so all layers run in inference mode (batchnorm).
  predictions = model(test_input, training=False)

  fig = plt.figure(figsize=(4,4))

  for i in range(predictions.shape[0]):
      plt.subplot(4, 4, i+1)
      plt.imshow(predictions[i, :, :, 0] * 127.5 + 127.5, cmap='gray')
      plt.axis('off')

  plt.savefig('{}/{:04d}.png'.format(test_output_dir, epoch))
 # plt.show()


def train(dataset, epochs):
  for epoch in range(epochs):
    start = time.time()
    print("Going into the {} epoch".format(epoch))
    for image_batch in dataset:
      train_step(image_batch)

    # Produce images for the GIF as we go
    display.clear_output(wait=True)
    generate_and_save_images(generator,
                             epoch + 1,
                             seed)

    # Save the model every 15 epochs
    if (epoch + 1) % 5 == 0:
      checkpoint.save(file_prefix = checkpoint_prefix)

    print ('Time for epoch {} is {} sec'.format(epoch + 1, time.time()-start))

  # Generate after the final epoch
  display.clear_output(wait=True)
  generate_and_save_images(generator, epochs, seed)

# Entraine le model
train(images_ds, EPOCHS)
