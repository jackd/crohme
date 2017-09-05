from __future__ import division
import os
import numpy as np
from xml.etree import ElementTree
import tensorflow as tf
from progress.bar import IncrementalBar
import crohme.data as data
from crohme.tex2png import tex_to_image


def _bytes_features(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=value))


def _images_to_tfrecords(filename, images, n, overwrite):
    if os.path.isfile(filename) and not overwrite:
        print('%s already exists - skipping' % filename)
        return
    print('%s: Converting %d images to records' % (filename, n))
    try:
        writer = tf.python_io.TFRecordWriter(filename)
        bar = IncrementalBar('image->tfrecords', max=n)
        it = iter(images)
        n_invalid = 0

        try:
            while True:
                try:
                    image = next(it)
                    features = tf.train.Features(
                        feature={'image': _bytes_features(image.tostring())})
                    example = tf.train.Example(features=features)
                    writer.write(example.SerializeToString())
                except ElementTree.ParseError:
                    print('Invalid xml tree - skipping')
                    n_invalid += 1
                    pass
                bar.next()

        except StopIteration:
            print('n_invalid: %d' % n_invalid)
            bar.finish()
    except Exception:
        os.remove(filename)
        raise


def records_filename(name, type_str, shape):
    return '%s_%s_%d_%d.tfrecords' % (name, type_str, shape[0], shape[1])


class DatasetBuilder(object):

    def __init__(self, manager):
        self._manager = manager

    @property
    def paths(self):
        return self._manager.example_paths

    @property
    def n_examples(self):
        return self._manager.n_examples

    @property
    def examples(self):
        return (data.InkMLExample(path) for path in self.paths)

    def aspect_ratios(self):
        boxes = np.array([example.bounding_box for example in self.examples])
        return boxes[:, 2] / boxes[:, 3]

    def create_tex_rendering_records(self, output_shape, overwrite=False):
        filename = records_filename(self._manager.name, 'tex', output_shape)
        renderings = (data.resize_image(
            tex_to_image(example.tex), *output_shape)
            for example in self.examples)
        _images_to_tfrecords(filename, renderings, self.n_examples, overwrite)

    def create_hme_rendering_records(
            self, output_shape, thickness=1, overwrite=False):
        filename = records_filename(self._manager.name, 'hme', output_shape)
        renderings = (example.hme_image(output_shape, thickness=thickness)
                      for example in self.examples)
        _images_to_tfrecords(filename, renderings, self.n_examples, overwrite)


if __name__ == '__main__':
    for name in data.DatasetManager.registered_names():
        builder = DatasetBuilder(data.DatasetManager.manager(name))
        output_shape = ((64*4), 64)
        builder.create_tex_rendering_records(output_shape)
        builder.create_hme_rendering_records(output_shape)
    # aspect_ratios = builder.aspect_ratios()
    # n, bins, patches = plt.hist(
    #     aspect_ratios, 50, normed=1, facecolor='green')

    # l = plt.plot(bins)
    # plt.show()
