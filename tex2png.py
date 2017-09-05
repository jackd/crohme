"""Python wrapper to tex2png bash script in bin/text2png."""
import os
import numpy as np
from subprocess import call
import cv2
from data import _root_dir

_tex2png_path = os.path.join(_root_dir, 'bin', 'tex2png', 'tex2png')
# print(_tex2png_path)
# exit()
# _temp_img_path = os.path.join(_root_dir, 'tmp.png')
_temp_img_path = os.path.join('/tmp', 'tex2png.png')


def render(tex_string, image_path, tex2png_args=[], *dvi2png_args):
    """Save the rendered `tex_string` to `image_path`."""
    args = [_tex2png_path, '-b', 'rgb 1 1 1', '-o', image_path,
            '-c', tex_string]
    args.extend(tex2png_args)
    args.extend(['--', '-T', 'tight', '-q*'])
    args.extend(dvi2png_args)
    return call(args)


def tex_to_image(tex_string, tex2png_args=[], *dvi2png_args):
    """
    Convert the given tex_string to an ndarray.

    Involves creation of a temporary file which is deleted at the end.
    """
    tex_string = tex_string.replace('\lt', '<').replace('\gt', '>')
    c = render(tex_string, _temp_img_path, tex2png_args, *dvi2png_args)
    if c == 0:
        image = cv2.imread(_temp_img_path, cv2.IMREAD_GRAYSCALE)
        return image
    else:
        raise Exception('Failed to generate image from tex %s - '
                        'did you remember surrounding $$?' % tex_string)


if __name__ == '__main__':
    import matplotlib.pyplot as plt
    image = tex_to_image('$ax^2 + bx + c$')
    plt.imshow(image)
    plt.show()
