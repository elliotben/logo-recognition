
from __future__ import print_function
import imageio
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import os
import sys
import tarfile
from IPython.display import display, Image
from six.moves.urllib.request import urlretrieve
from six.moves import cPickle as pickle
import tensorflow as tf

IMAGE_SIZE = 100  # Pixel width and height.
PIXEL_DEPTH = 255.0  # Number of levels per pixel.
NUM_CLASSES = 5  # Number of different brands
NUM_CHANNELS = 1 # grayscale

def load_logos(folder, min_num_images):
  """Load the data for a single brand."""
  image_files = os.listdir(folder)
  dataset = np.ndarray(shape=(len(image_files), IMAGE_SIZE, IMAGE_SIZE),
                         dtype=np.float32)
  print(folder)
  num_images = 0
  for image in image_files:
    image_file = os.path.join(folder, image)
    try:
      image_data = (imageio.imread(image_file).astype(float) -
                    PIXEL_DEPTH / 2) / PIXEL_DEPTH
      if image_data.shape != (IMAGE_SIZE, IMAGE_SIZE):
        raise Exception('Unexpected image shape: %s' % str(image_data.shape))
      dataset[num_images, :, :] = image_data
      num_images = num_images + 1
    except (IOError, ValueError) as e:
      print('Could not read:', image_file, ':', e, '- it\'s ok, skipping.')

  dataset = dataset[0:num_images, :, :]
  if num_images < min_num_images:
    raise Exception('Many fewer images than expected: %d < %d' %
                    (num_images, min_num_images))

  print('Full dataset tensor:', dataset.shape)
  print('Mean:', np.mean(dataset))
  print('Standard deviation:', np.std(dataset))
  return(dataset)

def maybe_pickle(data_folders, min_num_images_per_class, force=False):
  dataset_names = []
  for folder in data_folders:
    set_filename = folder + '.pickle'
    dataset_names.append(set_filename)
    if os.path.exists(set_filename) and not force and not folder == 'tests':
      # You may override by setting force=True.
      print('%s already present - Skipping pickling.' % set_filename)
    else:
      print('Pickling %s.' % set_filename)
      dataset = load_logos(folder, min_num_images_per_class)
      try:
        with open(set_filename, 'wb') as f:
          pickle.dump(dataset, f, pickle.HIGHEST_PROTOCOL)
      except Exception as e:
        print('Unable to save data to', set_filename, ':', e)

  return(dataset_names)

data_path = ['/Users/elliotbensabat/Desktop/projectAI/Dunkin','/Users/elliotbensabat/Desktop/projectAI/McDo',
            '/Users/elliotbensabat/Desktop/projectAI/Starbucks', '/Users/elliotbensabat/Desktop/projectAI/5_guys',
            '/Users/elliotbensabat/Desktop/projectAI/ABP']
predict_path = ['/Users/elliotbensabat/Desktop/projectAI/tests']

train_datasets = maybe_pickle(data_path, 6000)
test_datasets = maybe_pickle(data_path, 1500)

def make_arrays(nb_rows, img_size):
  if nb_rows:
    dataset = np.ndarray((nb_rows, img_size, img_size), dtype=np.float32)
    labels = np.ndarray(nb_rows, dtype=np.int32)
  else:
    dataset, labels = None, None
  return(dataset, labels)

def merge_datasets(pickle_files, train_size, valid_size=0):
  NUM_CLASSES = len(pickle_files)
  valid_dataset, valid_labels = make_arrays(valid_size, IMAGE_SIZE)
  train_dataset, train_labels = make_arrays(train_size, IMAGE_SIZE)
  vsize_per_class = valid_size // NUM_CLASSES
  tsize_per_class = train_size // NUM_CLASSES

  start_v, start_t = 0, 0
  end_v, end_t = vsize_per_class, tsize_per_class
  end_l = vsize_per_class+tsize_per_class
  for label, pickle_file in enumerate(pickle_files):
    try:
      with open(pickle_file, 'rb') as f:
        letter_set = pickle.load(f)
        # let's shuffle the logos to have random validation and training set
        np.random.shuffle(letter_set)
        if valid_dataset is not None:
          valid_letter = letter_set[:vsize_per_class, :, :]
          valid_dataset[start_v:end_v, :, :] = valid_letter
          valid_labels[start_v:end_v] = label
          start_v += vsize_per_class
          end_v += vsize_per_class

        train_letter = letter_set[vsize_per_class:end_l, :, :]
        train_dataset[start_t:end_t, :, :] = train_letter
        train_labels[start_t:end_t] = label
        start_t += tsize_per_class
        end_t += tsize_per_class
    except Exception as e:
      print('Unable to process data from', pickle_file, ':', e)
      raise

  return(valid_dataset, valid_labels, train_dataset, train_labels)


train_size = 5250*5
valid_size = 1500*5
test_size = 750*5

valid_dataset, valid_labels, train_dataset, train_labels = merge_datasets(
  train_datasets, train_size, valid_size)
_, _, test_dataset, test_labels = merge_datasets(test_datasets, test_size)

print('Training:', train_dataset.shape, train_labels.shape)
print('Validation:', valid_dataset.shape, valid_labels.shape)
print('Testing:', test_dataset.shape, test_labels.shape)


def randomize(dataset, labels):
  permutation = np.random.permutation(labels.shape[0])
  shuffled_dataset = dataset[permutation,:,:]
  shuffled_labels = labels[permutation]
  return(shuffled_dataset, shuffled_labels)
train_dataset, train_labels = randomize(train_dataset, train_labels)
test_dataset, test_labels = randomize(test_dataset, test_labels)
valid_dataset, valid_labels = randomize(valid_dataset, valid_labels)

data_root = '.'
pickle_file = os.path.join(data_root, 'logos.pickle')

try:
  f = open(pickle_file, 'wb')
  save = {
    'train_dataset': train_dataset,
    'train_labels': train_labels,
    'valid_dataset': valid_dataset,
    'valid_labels': valid_labels,
    'test_dataset': test_dataset,
    'test_labels': test_labels,
    }
  pickle.dump(save, f, pickle.HIGHEST_PROTOCOL)
  f.close()
except Exception as e:
  print('Unable to save data to', pickle_file, ':', e)
  raise


statinfo = os.stat(pickle_file)
print('Compressed pickle size:', statinfo.st_size)


# load pickel file from disk
def load_datasets(pickle_filename):
  with open(pickle_filename, 'rb') as f:
    return(pickle.Unpickler(f).load())

#display a random image from dataset
def show_sample_img(dataset):
  for file in dataset:
    train_dataset_graph = load_datasets(file)
    img_num = np.random.randint(0, 1000)
    plt.imshow(train_dataset_graph[img_num])
    plt.show()


show_sample_img(test_datasets)
show_sample_img(train_datasets)


pickle_file = 'logos.pickle'

with open(pickle_file, 'rb') as f:
  save = pickle.load(f)
  train_dataset = save['train_dataset']
  train_labels = save['train_labels']
  valid_dataset = save['valid_dataset']
  valid_labels = save['valid_labels']
  test_dataset = save['test_dataset']
  test_labels = save['test_labels']
  del save  # hint to help gc free up memory
  print('Training set', train_dataset.shape, train_labels.shape)
  print('Validation set', valid_dataset.shape, valid_labels.shape)
  print('Test set', test_dataset.shape, test_labels.shape)




def reformat(dataset, labels):
  dataset = dataset.reshape(
    (-1, IMAGE_SIZE, IMAGE_SIZE, NUM_CHANNELS)).astype(np.float32)
  labels = (np.arange(NUM_CLASSES) == labels[:,None]).astype(np.float32)
  return(dataset, labels)
train_dataset, train_labels = reformat(train_dataset, train_labels)
valid_dataset, valid_labels = reformat(valid_dataset, valid_labels)
test_dataset, test_labels = reformat(test_dataset, test_labels)
print('Training set', train_dataset.shape, train_labels.shape)
print('Validation set', valid_dataset.shape, valid_labels.shape)
print('Test set', test_dataset.shape, test_labels.shape)


def accuracy(predictions, labels):
  return(100.0 * np.sum(np.argmax(predictions, 1) == np.argmax(labels, 1))
          / predictions.shape[0])


#-----------------------------------------------------------------------------
#TRAINING PHASE

BATCH_SIZE = 32
PATCH_SIZE = 5
DEPTH = 16
NUM_HIDDEN = 32
NUM_STEPS = 300000
DROPOUT_RATE = 0.8
y_minibatch = []
y_validation = []
x = []

graph = tf.Graph()

with graph.as_default():

      # Input data.
    tf_train_dataset = tf.placeholder(tf.float32, shape=(BATCH_SIZE, IMAGE_SIZE, IMAGE_SIZE, NUM_CHANNELS))
    tf_train_labels = tf.placeholder(tf.float32, shape=(BATCH_SIZE, NUM_CLASSES))
    tf_valid_dataset = tf.constant(valid_dataset)
    tf_test_dataset = tf.constant(test_dataset)
    keep_prob = tf.placeholder(tf.float32)

      # Variables.
    layer1_weights = tf.Variable(tf.truncated_normal([PATCH_SIZE, PATCH_SIZE, NUM_CHANNELS, DEPTH], stddev=0.1))
    layer1_biases = tf.Variable(tf.zeros([DEPTH]))
    layer2_weights = tf.Variable(tf.truncated_normal([PATCH_SIZE, PATCH_SIZE, DEPTH, DEPTH], stddev=0.1))
    layer2_biases = tf.Variable(tf.constant(1.0, shape=[DEPTH]))
    layer3_weights = tf.Variable(tf.truncated_normal([IMAGE_SIZE // 4 * IMAGE_SIZE // 4 * DEPTH, NUM_HIDDEN], stddev=0.1))
    layer3_biases = tf.Variable(tf.constant(1.0, shape=[NUM_HIDDEN]))
    layer4_weights = tf.Variable(tf.truncated_normal([NUM_HIDDEN, NUM_CLASSES], stddev=0.1))
    layer4_biases = tf.Variable(tf.constant(1.0, shape=[NUM_CLASSES]))


      # Model.
    def model(data, keep_prob):
        conv = tf.nn.conv2d(data, layer1_weights, [1, 2, 2, 1], padding='SAME')
        hidden = tf.nn.relu(conv + layer1_biases)
        conv = tf.nn.conv2d(hidden, layer2_weights, [1, 2, 2, 1], padding='SAME')
        hidden = tf.nn.relu(conv + layer2_biases)
        shape = hidden.get_shape().as_list()
        reshape = tf.reshape(hidden, [shape[0], shape[1] * shape[2] * shape[3]])
        hidden = tf.nn.relu(tf.matmul(reshape, layer3_weights) + layer3_biases)
        drop_out = tf.nn.dropout(hidden, keep_prob)  # DROP-OUT here
        # output layer with linear activation
        out_layer = tf.matmul(drop_out, layer4_weights) + layer4_biases
        return(out_layer)

      # Training computation.
    logits = model(tf_train_dataset, keep_prob)
    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=tf_train_labels, logits=logits))


      # Optimizer.
    optimizer = tf.train.AdamOptimizer(1e-4).minimize(loss)

      # Predictions for the training, validation, and test data.
    train_prediction = tf.nn.softmax(logits)
    valid_prediction = tf.nn.softmax(model(tf_valid_dataset, 1.0))
    test_prediction = tf.nn.softmax(model(tf_test_dataset, 1.0))

    saver = tf.train.Saver()


with tf.Session(graph=graph) as session:
  tf.global_variables_initializer().run()
  print('Initialized')
  for step in range(NUM_STEPS):
    offset = (step * BATCH_SIZE) % (train_labels.shape[0] - BATCH_SIZE)
    batch_data = train_dataset[offset:(offset + BATCH_SIZE), :, :, :]
    batch_labels = train_labels[offset:(offset + BATCH_SIZE), :]
    feed_dict = {tf_train_dataset : batch_data, tf_train_labels : batch_labels, keep_prob : DROPOUT_RATE}
    _, l, predictions = session.run(
      [optimizer, loss, train_prediction], feed_dict=feed_dict)
    if (step % 3000 == 0):
        print('Minibatch loss at step %d: %f' % (step, l))
        acc_minibatch = accuracy(predictions, batch_labels)
        acc_validation = accuracy(valid_prediction.eval(), valid_labels)
        print('Minibatch accuracy: %.1f%%' % acc_minibatch)
        print('Validation accuracy: %.1f%%' % acc_validation)
        y_minibatch.append(acc_minibatch)
        y_validation.append(acc_validation)
        x.append(step)
  saver.save(session, './logo-model-new.ckpt')
  print('Test accuracy: %.1f%%' % accuracy(test_prediction.eval(), test_labels))


fig, ax = plt.subplots()
fig, ax1 = plt.subplots()
ax.plot(x, y_minibatch)
ax1.plot(x, y_validation)

ax.set(xlabel='step', ylabel='acc %', title='acc % for minibatch')
ax1.set(xlabel='step', ylabel='acc %', title='acc % for validation')
ax.grid()
ax1.grid()
plt.show()
