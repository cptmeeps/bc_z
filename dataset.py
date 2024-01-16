import numpy as np
import tqdm
from PIL import Image

import tensorflow_datasets as tfds
import tensorflow as tf

# utils

dataset_name = "bc_z"
source = f'gs://gresearch/robotics/{dataset_name}/0.1.0'

def as_gif(images, path='temp.gif'):
  images[0].save(path, save_all=True, append_images=images[1:], duration=100, loop=0)
  gif_bytes = open(path,'rb').read()
  return gif_bytes

def display_example():
  b = tfds.builder_from_directory(builder_dir=source)
  ds = b.as_dataset(split='train[:10]').shuffle(10)   # take only first 10 episodes
  episode = next(iter(ds))
  instructions = [step['observation']['natural_language_instruction'] for step in episode['steps']]
  images = [step['observation']['image'] for step in episode['steps']]
  images = [Image.fromarray(image.numpy()) for image in images]
  print(
    'episode step count: ', len(episode['steps']),
    '\n', 'instructions: ', instructions[0]
  )
  
def download_ds():
  # _ = tfds.load('bc_z', data_dir='~/bc_z_dataset')
  # load raw dataset --> replace this with tfds.load(<dataset_name>) on your local machine
  
  dataset = dataset_name
  b = tfds.builder_from_directory(builder_dir=source)
  ds = b.as_dataset(split='train[:10]')

  def episode2steps(episode):
    return episode['steps']

  def step_map_fn(step):
    return {
        'observation': {
            'image': tf.image.resize(step['observation']['image'], (128, 128)),
        },
        'action': tf.concat([
            step['action']['world_vector'],
            step['action']['rotation_delta'],
            step['action']['gripper_closedness_action'],
        ], axis=-1)
    }

  # convert RLDS episode dataset to individual steps & reformat
  ds = ds.map(
      episode2steps, num_parallel_calls=tf.data.AUTOTUNE).flat_map(lambda x: x)
  ds = ds.map(step_map_fn, num_parallel_calls=tf.data.AUTOTUNE)

  # shuffle, repeat, pre-fetch, batch
  ds = ds.cache()         # optionally keep full dataset in memory
  ds = ds.shuffle(100)    # set shuffle buffer size
  ds = ds.repeat()        # ensure that data never runs out

def batch_ds():
  for i, batch in tqdm.tqdm(
      enumerate(ds.prefetch(3).batch(4).as_numpy_iterator())):
    # here you would add your Jax / PyTorch training code
    if i == 10000: break

# 

