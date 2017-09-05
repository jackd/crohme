import os
import tensorflow as tf
import numpy as np
import tensorflow.contrib.data as data
from create import records_filename
from crohme.data import DatasetManager, _root_dir


def get_example_parser(output_shape):
    def parse_image(example_proto):
        n = np.prod(output_shape)
        features = {
            'image': tf.FixedLenFeature(shape=n, dtype=tf.string)
        }
        parsed_features = tf.parse_single_example(example_proto, features)
        image_raw = parsed_features['image']
        image = tf.reshape(
            tf.decode_raw(image_raw, tf.uint8), output_shape[-1::-1])
        return image
    return parse_image


def get_dataset(filenames, shape):
    return data.TFRecordDataset(filenames).map(get_example_parser(shape))


def get_full_dataset(type_str, shape):
    names = DatasetManager.registered_names()
    filenames = [records_filename(name, type_str, shape) for name in names]
    filenames = [os.path.join(_root_dir, 'tf_dataset', fn) for fn in filenames]
    if len(filenames) == 0:
        raise Exception('No tfrecords for %s, %s' % (type_str, str(shape)))
    return get_dataset(filenames, shape)


def get_full_tex_dataset(shape):
    return get_full_dataset('tex', shape)


def get_full_hme_dataset(shape):
    return get_full_dataset('hme', shape)


def get_zipped_datasets(shape, types=['tex', 'hme']):
    return data.Dataset.zip([get_full_dataset(t, shape) for t in types])


if __name__ == '__main__':
    import matplotlib.pyplot as plt
    full = get_zipped_datasets((256, 64))
    tex_tf, hme_tf = full.batch(4).make_one_shot_iterator().get_next()

    with tf.Session() as sess:
        tex, hme = sess.run([tex_tf, hme_tf])

    for t, h in zip(tex, hme):
        fig, (ax0, ax1) = plt.subplots(2, 1)
        ax0.imshow(t)
        ax1.imshow(h)
        plt.show()
