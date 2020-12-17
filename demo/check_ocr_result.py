# coding=utf-8

"""
Created by Jayvee on 2020-06-15.
https://github.com/JayveeHe
"""
import io
import os

import numpy
from PIL import Image
from masktextspotterv3.textspotter import MaskTextSpotter

from masktextspotterv3.config import cfg

PROJECT_PATH = os.path.dirname(os.path.dirname(__file__))
DATAPATH = '%s/data' % PROJECT_PATH
#
cfg.merge_from_file('%s/seg_rec_poly_fuse_feature.yaml' % DATAPATH)
# print('initing ocr model')
cfg['MODEL']['WEIGHT'] = '%s/MaskTextSpotterV3_trained_model.pth' % DATAPATH
cfg['MODEL']['DEVICE'] = 'cpu'

target_size = 800
mts = MaskTextSpotter(
    cfg,
    min_image_size=target_size,
    confidence_threshold=0.01,
    output_polygon=True,
    spellfix=False
)

test_url = 'https://cdn.shopifycdn.net/s/files/1/0204/0505/9684/products/IMG_20200724_154245_540x.jpg?v=1595578416'
#
import requests
img_obj = Image.open(io.BytesIO(requests.get(test_url, verify=False).content))
img_obj = img_obj.convert('RGB')
#
# img_obj = Image.open('%s/demo_test_image.png' % DATAPATH)
#
# img_obj = img_obj.convert('RGB')

img_size = img_obj.size
# print(datetime.datetime.now(), 'convert rgb')
width = img_size[0]
height = img_size[1]
if max(width, height) > target_size:
    if width > height:
        new_width = target_size
        new_height = int(height * target_size / width)
    else:
        new_width = int(width * target_size / height)
        new_height = target_size
    img_obj = img_obj.resize((new_width, new_height))
    print('ocr resize %s,%s -> %s,%s' % (width, height, new_width, new_height))
else:
    img_obj = img_obj
result_polygons, result_words, result_lines = mts.run_on_pillow_image(img_obj)
open_cv_image = numpy.array(img_obj)
# Convert RGB to BGR
open_cv_image = open_cv_image[:, :, ::-1].copy()
result_image = mts.visualization(open_cv_image, result_polygons, result_words)
result_image = Image.fromarray(result_image[:, :, ::-1])
result_image.show()
# result_image.save('%s/demo_results.jpg' % DATAPATH)
print(result_words)
