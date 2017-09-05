"""Provides utility functions and InkMLExample class for extracting data."""
from __future__ import division
import os
import xml.etree.ElementTree as ET
from cached_property import cached_property
import numpy as np
import cv2

_root_dir = os.path.dirname(os.path.realpath(__file__))
_data_dir = os.path.join(_root_dir, 'data')


class DatasetManager(object):
    _managers = {}

    @staticmethod
    def manager(name):
        return DatasetManager._managers[name]

    @staticmethod
    def registered_names():
        names = DatasetManager._managers.keys()
        names.sort()
        return names

    def __init__(self, name):
        if name in DatasetManager._managers:
            raise Exception('Dataset with name %s already registered' % name)
        self._name = name
        DatasetManager._managers[name] = self

    @property
    def example_paths(self):
        raise NotImplementedError()

    @property
    def n_examples(self):
        raise NotImplementedError()

    @property
    def name(self):
        return self._name


class FolderManager(DatasetManager):
    def __init__(self, name, folder):
        self._folder = folder
        super(FolderManager, self).__init__(name)

    @property
    def _filenames(self):
        return os.listdir(self._folder)

    @property
    def example_paths(self):
        return (os.path.join(
            self._folder, fn) for fn in self._filenames)

    @property
    def n_examples(self):
        return len(self._filenames)


train2011_manager = FolderManager('train2011', os.path.join(
    _data_dir, 'CROHME2011_data', 'CROHME_training', 'CROHME_training'))
train2012_manager = FolderManager('train2012', os.path.join(
    _data_dir, 'CROHME2012_data', 'trainData', 'trainData'))
train2013_managers = {
    k: FolderManager('train2013-%s' % k, os.path.join(
        _data_dir, 'CROHME2013_data', 'TrainINKML', k))
    for k in ['expressmatch', 'HAMEX', 'KAIST', 'MathBrush', 'MfrDB']}


def trace_to_numpy(trace_string):
    """Get (n, 2) ndarray of ints corresponding to the given trace_string."""
    return np.array([[float(a), float(b)] for (a, b) in [
               s.strip().split(' ')[:2]
               for s in trace_string.strip().split(',')]],
               dtype=np.float32)


def meta_bounding_box(np_boxes):
    """Get the bounding box of multiple bounding boxes."""
    lower_lims = np_boxes[:, :2]
    upper_lims = np_boxes[:, 2:] + lower_lims
    mins = np.min(lower_lims, axis=0)
    maxs = np.max(upper_lims, axis=0)
    maxs -= mins
    return mins[0], mins[1], maxs[0], maxs[1]


class Trace(object):
    """A higher level representation of the data from a trace tag."""

    def __init__(self, et):
        """Initialize with the ElementTree."""
        self._et = et

    @property
    def id(self):
        """Get the id of this path."""
        return self._et.attrib['id']

    @cached_property
    def ndarray(self):
        """Get this trace as a 2d numpy array."""
        return trace_to_numpy(self._et.text)

    @cached_property
    def bounding_box(self):
        """Get x, y, w, h for this path."""
        points = self.ndarray
        mins = np.min(points, axis=0)
        maxs = np.max(points, axis=0)
        maxs -= mins
        return mins[0], mins[1], maxs[0], maxs[1]


class TraceGroup(object):
    """A class representing high level features of a particular trace_group."""

    def __init__(self, example, et):
        """Initialize with the parent InkMLExample and ElementTree et."""
        self._example = example
        id_attr = et.attrib.items()[0]
        assert(id_attr[0].split('}')[1] == 'id')
        self._id = id_attr[1]
        self._tex = None
        self._xml_href = None
        self._trace_ids = []
        for child in et:
            TraceGroup._visits[child.tag.split('}')[1]](self, child)
        assert(self._tex is not None)
        assert(self._xml_href is not None)
        assert(len(self._trace_ids) > 0)

    def _visit_annotation(self, et):
        assert(self.tex is None)
        self._tex = '$%s$' % et.text

    def _visit_trace_view(self, et):
        self._trace_ids.append(et.attrib['traceDataRef'])

    def _visit_annotation_xml(self, et):
        assert(self._xml_href is None)
        self._xml_href = et.attrib['href']

    _visits = {
        'annotation': _visit_annotation,
        'traceView': _visit_trace_view,
        'annotationXML': _visit_annotation_xml
    }

    @property
    def traces(self):
        """Get a list of np.ndarrays, one for each trace in this group."""
        return [self._example.trace(trace_id) for trace_id in self._trace_ids]

    @property
    def bounding_box(self):
        """Get the bounding box of this trace_group."""
        return meta_bounding_box(
            np.array([t.bounding_box for t in self.traces]))

    @property
    def tex(self):
        """Get the tex."""
        return self._tex

    @property
    def xml_href(self):
        """Get the xml_href with the MathML element."""
        return self._xml_href


class InkMLExample(object):
    """A class for extracting high level features from an inkml file."""

    def __init__(self, path):
        """
        Get an example based on a inkml file path.

        Most properties are evaluated lazily.
        """
        self._root = ET.parse(path).getroot()
        assert(self._root.tag.split('}')[1] == 'ink')
        self._annotations = {}
        self._mathML_annotation = None
        self._traces = {}
        self._outer_trace_group = None
        self._trace_groups = None
        for child in self._root:
            InkMLExample._init_visits[child.tag.split('}')[1]](self, child)

    def _pass(self, et):
        pass

    @property
    def tex(self):
        """Get tex of this example."""
        tex = self._annotations['truth']
        if not(tex[0] == '$' and tex[-1] == '$'):
            tex = '$%s$' % self._annotations['truth']
        return tex

    def _init_annotation(self, et):
        ann_type = et.attrib['type']
        assert(ann_type not in self._annotations)
        self._annotations[ann_type] = et.text

    def _init_annotation_xml(self, et):
        assert(et.attrib['encoding'] in
               ['Content-MathML', 'Presentation-MathML'])
        assert(et.attrib['type'] == 'truth')
        assert(self._mathML_annotation is None)
        self._mathML_annotation = et

    def _init_trace(self, et):
        trace = Trace(et)
        self._traces[trace.id] = trace

    def _init_outer_trace_group(self, et):
        assert(self._outer_trace_group is None)
        self._outer_trace_group = et

    def _init_inner_trace_group(self, et):
        self._inner_trace_group[et.attrib['id']] = et

    _init_visits = {
        'traceFormat': _pass,
        'annotation': _init_annotation,
        'annotationXML': _init_annotation_xml,
        'trace': _init_trace,
        'traceGroup': _init_outer_trace_group
    }

    def trace(self, id_string):
        """Get the numpy array corresponding to the given trace."""
        return self._traces[id_string]

    @property
    def traces(self):
        """Get a list of `Trace`s making up this example."""
        return [v for k, v in self._traces.items()]

    @cached_property
    def trace_groups(self):
        """Get the trace_groups for this example."""
        return [TraceGroup(
            self, child) for child in self._outer_trace_group
            if child.tag.split('}')[1] == 'traceGroup']

    @property
    def bounding_box(self):
        """Get the bounding box of the entire example."""
        return meta_bounding_box(
            np.array([t.bounding_box for t in self.traces]))

    def hme_image(self, output_shape, **polyline_kwargs):
        x, y, w, h = self.bounding_box
        offset = np.array([x, y])
        paths = [trace.ndarray - offset for trace in self.traces]
        image = polylines_to_image(
            paths, w, h, output_shape[0], output_shape[1], **polyline_kwargs)
        return image


def resize_image(
        image, target_width, target_height, empty_val=255):
    """Get resized image without affecting aspect ratio."""
    original_height, original_width = image.shape[:2]
    scale_factor = min(
        target_width / original_width, target_height / original_height)
    if len(image.shape) == 3:
        target_shape = (target_height, target_width, image.shape[2])
    else:
        target_shape = (target_height, target_width)
    output = np.ones(target_shape, dtype=np.uint8)*empty_val
    resize_width = int(scale_factor*original_width)
    resize_height = int(scale_factor*original_height)
    x0 = (target_width - resize_width) // 2
    y0 = (target_height - resize_height) // 2
    output[y0: y0 + resize_height, x0: x0 + resize_width] = cv2.resize(
        image, (resize_width, resize_height))
    return output


def polylines_to_image(
        paths, original_width, original_height, target_width, target_height,
        thickness=1, stroke_color=0, background_color=255,
        blur_kwargs={'ksize': (3, 3), 'sigmaX': -1}):
    paths = resize_paths(
        paths, original_width, original_height, target_width, target_height)
    image = np.ones(
        (target_height, target_width), dtype=np.uint8)*background_color
    cv2.polylines(image, paths, False, color=stroke_color, thickness=thickness)
    image = cv2.GaussianBlur(image, **blur_kwargs)
    return image


def resize_paths(
        paths, original_width, original_height, target_width, target_height):
    """Get resized paths without affecting aspect ratio."""
    scale_factor = min(
        target_width / original_width, target_height / original_height)
    x0 = (target_width - scale_factor * original_width) // 2
    y0 = (target_height - scale_factor * original_height) // 2
    offset = np.array([x0, y0])
    return [(p*scale_factor + offset).astype(np.int32) for p in paths]


if __name__ == '__main__':
    import matplotlib.pyplot as plt
    from tex2png import tex_to_image
    path = list(train2011_manager.example_paths)[0]
    example = InkMLExample(path)
    print(example.tex)
    # group = example.trace_groups[0]
    # print([k.bounding_box for k in group.traces])
    # print(group.bounding_box)
    # print(group.tex)
    # print(example.tex)

    # x, y, w, h = example.bounding_box
    # tr = np.array((x, y), dtype=np.int32)
    # paths = [t.ndarray - tr for t in example.traces]
    #
    # target_height = 64
    # target_width = target_height * 4
    #
    # def vis(paths, tex_image):
    #     image = np.ones((h, w), dtype=np.uint8)*255
    #     cv2.polylines(image, paths, False, color=0, thickness=30)
    #     fig, ((ax0, ax1), (ax2, ax3)) = plt.subplots(2, 2)
    #     ax0.imshow(image, cmap='gray')
    #     ax1.imshow(tex_image, cmap='gray')
    #     hand_resized = np.ones(
    #         (target_height, target_width), dtype=np.uint8)*255
    #     resized_paths = resize_paths(
    #         paths, w, h, target_width, target_height)
    #     cv2.polylines(
    #         hand_resized, resized_paths, False, color=0, thickness=1)
    #     hand_resized = cv2.GaussianBlur(hand_resized, (3, 3), -1)
    #     ax2.imshow(hand_resized, cmap='gray')
    #     im_h, im_w = tex_image.shape[:2]
    #     ax3.imshow(resize_image(tex_image, target_width, target_height),
    #                cmap='gray')
    #     plt.show()
    #
    # tex_image = tex_to_image('$%s$ ' % example.tex)
    # vis(paths, tex_image)
    image = example.hme_image((256, 64))
    plt.imshow(image)
    plt.show()
    image = tex_to_image(example.tex)
    plt.imshow(image)
    plt.show()
