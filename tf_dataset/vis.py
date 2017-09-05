import tensorflow as tf
import numpy as np
import tensorflow.contrib.data as data
# from crohme.data import DatasetManager
from create import records_filename
import matplotlib.pyplot as plt
from dataset import get_example_parser

output_shape = (256, 64)

name = 'train2011'
parse_image = get_example_parser(output_shape)

tex_dataset = data.TFRecordDataset(
    [records_filename(name, 'tex', output_shape)]).map(parse_image)
hme_dataset = data.TFRecordDataset(
    [records_filename(name, 'hme', output_shape)]).map(parse_image)

combined = data.Dataset.zip((tex_dataset, hme_dataset))

batch_size = 4
tex_tf, hme_tf = combined.batch(batch_size).make_one_shot_iterator().get_next()

with tf.Session() as sess:
    tex, hme = sess.run([tex_tf, hme_tf])


for t, h in zip(tex, hme):
    fig, (ax0, ax1) = plt.subplots(2, 1)
    ax0.imshow(t)
    ax1.imshow(h)
    plt.show()
