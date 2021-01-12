# coding=utf-8

import datetime
import os
import re
import sys
import traceback

from PIL import Image
from tqdm import tqdm


def predict_img_path(mts_inst, img_path, output_path):
    try:
        # print(datetime.datetime.now(), 'predicting %s, output_path= %s' % (img_path, output_path))
        img_obj = Image.open(open(img_path, 'rb'))
        img_obj = img_obj.convert('RGB')
        #
        # img_obj = Image.open('%s/demo_test_image.png' % DATAPATH)
        #
        # img_obj = img_obj.convert('RGB')

        # img_size = img_obj.size
        # origin_size = img_obj.size
        # # print(datetime.datetime.now(), 'convert rgb')
        # width = img_size[0]
        # height = img_size[1]
        # if max(width, height) > target_size:
        #     if width > height:
        #         new_width = target_size
        #         new_height = int(height * target_size / width)
        #     else:
        #         new_width = int(width * target_size / height)
        #         new_height = target_size
        #     img_obj = img_obj.resize((new_width, new_height))
        #     print('ocr resize %s,%s -> %s,%s' % (width, height, new_width, new_height))
        # else:
        #     img_obj = img_obj
        #     new_width, new_height = width, height
        result_polygons, result_words, result_lines = mts_inst.run_on_pillow_image(img_obj)
        with open(output_path, 'w') as fout:
            # w_scale_ratio = origin_size[0] / new_width
            # h_scale_ratio = origin_size[1] / new_height
            for i in range(len(result_polygons)):
                tmp_polygon = result_polygons[i]
                # for j in range(len(tmp_polygon)):
                #     tmp_polygon[j] = (int(tmp_polygon[j] * w_scale_ratio)) if j % 2 == 0 else (int(
                #         tmp_polygon[j] * h_scale_ratio))
                polygon = [str(a) for a in tmp_polygon]
                word = result_words[i]
                result_str = ','.join(polygon + [word]) + '\r\n'
                fout.write(result_str)
    except Exception as e:
        traceback.print_exc()


def prepare_prediction_result(input_dir, output_dir, mts_version=3, c=0.7, min_side_size=800):
    if mts_version == 3:
        from masktextspotterv3.textspotter import MaskTextSpotter
        from masktextspotterv3.config import cfg
        DATAPATH = 'MaskTextSpotterV3/data'

        cfg.merge_from_file('%s/seg_rec_poly_fuse_feature.yaml' % DATAPATH)
        cfg['MODEL']['WEIGHT'] = '%s/MaskTextSpotterV3_trained_model.pth' % DATAPATH
        # print('initing ocr model')
        # cfg.merge_from_file('%s/SynthText-Pretrain/seg_rec_poly_fuse_feature.yaml' % DATAPATH)
        # cfg['MODEL']['WEIGHT'] = '%s/SynthText-Pretrain/SynthText-Pretrain_model.pth' % DATAPATH

        cfg['MODEL']['DEVICE'] = 'cpu'
        print('mts v3')
        # target_size = 800
        mts = MaskTextSpotter(
            cfg,
            min_image_size=min_side_size,
            confidence_threshold=c,
            output_polygon=False,
            spellfix=True
        )
    elif mts_version == 2:

        # v2

        from maskrcnn_benchmark.textspotter import MaskTextSpotter
        # DATAPATH = '/Users/jiawhe/Jobs/WebInspection-Model-Pipeline/data'

        from maskrcnn_benchmark.config import cfg

        #

        print('mts v2')
        V2_DATAPATH = 'data'
        cfg.merge_from_file('%s/models/OCR/batch.yaml' % V2_DATAPATH)
        # cfg = pickle.load(open('%s/models/OCR/config.pkl' % DATAPATH, 'rb'))
        # print('initing ocr model')
        cfg['MODEL']['WEIGHT'] = '%s/models/OCR/model_pretrain.pth' % V2_DATAPATH
        cfg['MODEL']['DEVICE'] = 'cpu'

        # target_size = 800
        mts = MaskTextSpotter(
            cfg,
            min_image_size=min_side_size,
            confidence_threshold=c,
            output_polygon=False
        )
    else:
        return

        #
    input_flist = os.listdir(input_dir)
    if not os.path.exists(output_dir):
        print('mkdir %s' % output_dir)
        os.mkdir(output_dir)
    for fname in tqdm(input_flist):
        image_seq = re.findall('img_(\d+)\..*', fname)[0]
        input_path = os.path.join(input_dir, fname)
        output_path = os.path.join(output_dir, 'res_img_%s.txt' % image_seq)
        predict_img_path(mts, input_path, output_path)


if __name__ == '__main__':
    input_dir = sys.argv[1]
    output_dir = sys.argv[2]
    model_version = int(sys.argv[3])
    c_threshold = float(sys.argv[4])
    min_side_size = int(sys.argv[5])
    prepare_prediction_result(
        input_dir,
        output_dir, mts_version=model_version, c=c_threshold, min_side_size=min_side_size)
