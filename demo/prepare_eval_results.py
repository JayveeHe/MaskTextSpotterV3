# coding=utf-8

"""
Created by jiawhe on 2020-11-30.
https://github.paypal.com/jiawhe
"""
import datetime
import os
import sys
import traceback

from PIL import Image
from tqdm import tqdm

from masktextspotterv3.textspotter import MaskTextSpotter
from masktextspotterv3.config import cfg


def predict_img_path(mts_inst, img_path, output_path, target_size=800):
    try:
        print(datetime.datetime.now(), 'predicting %s, output_path= %s' % (img_path, output_path))
        img_obj = Image.open(open(img_path, 'rb'))
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
        result_polygons, result_words, result_lines = mts_inst.run_on_pillow_image(img_obj)
        with open(output_path, 'w') as fout:
            for i in range(len(result_polygons)):
                polygon = [str(a) for a in result_polygons[i]]
                word = result_words[i]
                result_str = ','.join(polygon + [word]) + '\n'
                fout.write(result_str)
    except Exception as e:
        traceback.print_exc()


def prepare_prediction_result(input_dir, output_dir, DATAPATH):
    # DATAPATH = '/Users/jiawhe/playground/MaskTextSpotterV3/data'

    cfg.merge_from_file('%s/seg_rec_poly_fuse_feature.yaml' % DATAPATH)
    cfg['MODEL']['WEIGHT'] = '%s/MaskTextSpotterV3_trained_model.pth' % DATAPATH
    # print('initing ocr model')
    # cfg.merge_from_file('%s/SynthText-Pretrain/seg_rec_poly_fuse_feature.yaml' % DATAPATH)
    # cfg['MODEL']['WEIGHT'] = '%s/SynthText-Pretrain/SynthText-Pretrain_model.pth' % DATAPATH

    cfg['MODEL']['DEVICE'] = 'cpu'

    target_size = 800
    mts = MaskTextSpotter(
        cfg,
        min_image_size=target_size,
        confidence_threshold=0.5,
        output_polygon=False,
        spellfix=False
    )

    #
    input_flist = os.listdir(input_dir)
    if not os.path.exists(output_dir):
        print('mkdir %s' % output_dir)
        os.mkdir(output_dir)
    for fname in tqdm(input_flist):
        # image_name = re.findall('(img_\d+\.jpg).txt', fname)[0]
        input_path = os.path.join(input_dir, fname)
        output_path = os.path.join(output_dir, 'res_%s.txt' % fname)
        predict_img_path(mts, input_path, output_path)


if __name__ == '__main__':
    input_dir = sys.argv[1]
    output_dir = sys.argv[2]
    DATAPATH = sys.argv[3]
    prepare_prediction_result(
        input_dir,
        output_dir, DATAPATH)
