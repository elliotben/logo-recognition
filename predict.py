from PIL import Image
from resizeimage import resizeimage
import imageio
import sys
import numpy as np
import tensorflow as tf

NUM_CLASSES = 5
BATCH_SIZE = 32
PATCH_SIZE = 5
DEPTH = 16
NUM_HIDDEN = 32
NUM_STEPS = 300000
DROPOUT_RATE = 0.8
y_minibatch = []
y_validation = []
x = []

IMAGE_SIZE = 100
RESIZED_WIDTH = 100
RESIZED_HEIGHT = 100
NUM_CHANNELS = 1
PIXEL_DEPTH = 255.0  # Number of levels per pixel.
logos = ['Dunkin Donuts', 'McDonald\'s', 'Starbucks', '5 Guys', 'Au Bon Pain']


graph = tf.Graph()

with graph.as_default():
    
    # Input data.
    tf_train_dataset = tf.placeholder(tf.float32, shape=(BATCH_SIZE, IMAGE_SIZE, IMAGE_SIZE, NUM_CHANNELS))
    tf_train_labels = tf.placeholder(tf.float32, shape=(BATCH_SIZE, NUM_CLASSES))
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
        return out_layer
    
    # Training computation.
    logits = model(tf_train_dataset, keep_prob)
    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=tf_train_labels, logits=logits))
    
    
    # Optimizer.
    optimizer = tf.train.AdamOptimizer(1e-4).minimize(loss)
    
    # Predictions for the training, validation, and test data.
    train_prediction = tf.nn.softmax(logits)
    
    saver = tf.train.Saver()


#returns the probability of the inputed picture being in each class
def get_proba(path):
    im1 = Image.open(path)
    im1 = im1.convert('L')
    #use nearest neighbour to resize
    im2 = im1.resize((RESIZED_WIDTH, RESIZED_HEIGHT), Image.NEAREST)
    ext = ".jpg"
    im2.save(path)
    im2 = (imageio.imread(path).astype(float) -
                    PIXEL_DEPTH / 2) / PIXEL_DEPTH
    x = im2.reshape(
        (-1, IMAGE_SIZE, IMAGE_SIZE, NUM_CHANNELS)).astype(np.float32)
    with tf.Session(graph=graph) as session:
        saver.restore(session, "./logo-model.ckpt")
        tf_predict_input = tf.constant(x)
        predict_input = tf.nn.softmax(model(tf_predict_input, 1.0))
        return(predict_input.eval())

#returns the class with the highest probability: our model predictions
def get_pred(path):
    p = get_proba(path)
    while np.isnan(p).all():
        p = get_proba(path)
    i = np.argmax(p, 1)[0]
    print(logos[i])

def main(argv):
    get_pred(argv)

if __name__ == "__main__":
    main(sys.argv[1:][0])
#get_pred('./abp_test.jpg')
