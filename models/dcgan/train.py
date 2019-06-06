import tensorflow as tf
import os
import time
import matplotlib.pyplot as plt
from IPython import display
from dcgan import discriminator
from discriminator import discriminator_optimizer
from generator import generator_optimizer

EPOCHS = 50
noise_dim = 100
num_examples_to_generate = 32
BATCH_SIZE = 256








