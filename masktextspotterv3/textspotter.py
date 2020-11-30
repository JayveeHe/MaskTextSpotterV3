# coding=utf-8

"""
Created by Jayvee_He on 2020-10-20.
"""
import copy
import random

import cv2
import pkg_resources
import torch
from shapely.geometry import Polygon, LineString, Point
from torchvision import transforms as T

from masktextspotterv3.modeling.detector import build_detection_model
from masktextspotterv3.utils.checkpoint import DetectronCheckpointer
from masktextspotterv3.structures.image_list import to_image_list
from masktextspotterv3.utils.chars import getstr_grid, get_tight_rect

from PIL import Image
import numpy as np
from symspellpy import SymSpell, Verbosity


class MaskTextSpotter(object):
    def __init__(
            self,
            cfg,
            confidence_threshold=0.7,
            min_image_size=224,
            output_polygon=True,
            spellfix=True
    ):
        self.cfg = cfg.clone()
        self.model = build_detection_model(cfg)
        self.model.eval()
        self.device = torch.device(cfg.MODEL.DEVICE)
        self.model.to(self.device)
        self.min_image_size = min_image_size

        self.spellfix = spellfix

        self.sym_spell = SymSpell(max_dictionary_edit_distance=2, prefix_length=7)
        dictionary_path = pkg_resources.resource_filename("symspellpy", "frequency_dictionary_en_82_765.txt")

        bigram_dictionary_path = pkg_resources.resource_filename("symspellpy",
                                                                 "frequency_bigramdictionary_en_243_342.txt")

        self.sym_spell.load_dictionary(dictionary_path, term_index=0, count_index=1)

        self.sym_spell.load_bigram_dictionary(bigram_dictionary_path, term_index=0, count_index=2)

        checkpointer = DetectronCheckpointer(cfg, self.model)
        if len(cfg.MODEL.WEIGHT):
            import logging
            logging.info('loading MaskTextSpotter from %s' % cfg.MODEL.WEIGHT)
            _ = checkpointer.load(cfg.MODEL.WEIGHT)

        self.transforms = self.build_transform()
        self.cpu_device = torch.device("cpu")
        self.confidence_threshold = confidence_threshold
        self.output_polygon = output_polygon

    def build_transform(self):
        """
        Creates a basic transformation that was used to train the models
        """
        cfg = self.cfg
        # we are loading images with OpenCV, so we don't need to convert them
        # to BGR, they are already! So all we need to do is to normalize
        # by 255 if we want to convert to BGR255 format, or flip the channels
        # if we want it to be in RGB in [0-1] range.
        if cfg.INPUT.TO_BGR255:
            to_bgr_transform = T.Lambda(lambda x: x * 255)
        else:
            to_bgr_transform = T.Lambda(lambda x: x[[2, 1, 0]])

        normalize_transform = T.Normalize(
            mean=cfg.INPUT.PIXEL_MEAN, std=cfg.INPUT.PIXEL_STD
        )

        transform = T.Compose(
            [
                T.ToPILImage(),
                T.Resize(self.min_image_size),
                T.ToTensor(),
                to_bgr_transform,
                normalize_transform,
            ]
        )
        return transform

    def run_on_opencv_image(self, image):
        """
        Arguments:
            image (np.ndarray): an image as returned by OpenCV
        Returns:
            result_polygons (list): detection results
            result_words (list): recognition results
        """
        result_polygons, result_words, result_dict = self.compute_prediction(image)
        return result_polygons, result_words, result_dict

    def run_on_pillow_image(self, image):
        arr = np.array(image, dtype=np.uint8)
        result_polygons, result_words, result_dict = self.run_on_opencv_image(arr)
        return result_polygons, result_words, result_dict

    def compute_prediction(self, original_image):

        def spell_fix(wd):
            if self.spellfix:
                new_word = [s.term for s in
                            self.sym_spell.lookup(wd, Verbosity.CLOSEST, max_edit_distance=2, include_unknown=True)][0]
            else:
                new_word = wd
            return new_word

        def chunks(l, n):
            for i in range(0, len(l), n):
                yield l[i: i + n]

        def mk_direction(char_polygons):

            def centroid(char_polygon):
                centroid = Polygon(list(chunks(char_polygon, 2))).centroid.coords
                return list(centroid)[0]

            first, last = char_polygons[0], char_polygons[-1]
            start, end = centroid(first), centroid(last)
            if start[0] == end[0]:
                end = (end[0] + 1, end[1])
            return start, end

        def line_detection(dicts, char_ratio=1.5):
            # box  [x1, y1, x2, y2]
            sorted_res = sorted(dicts, key=lambda d: d["box"][0])
            lines = dict()

            def point_in_next_word(word):
                width = word["box"][2] - word["box"][0]  # width = x2 - x1
                avg_char_width = width / float(len(word["seq_word"]))
                last_right_border = word["box"][2]
                next_word_pos_x = last_right_border + char_ratio * avg_char_width
                next_word_pos_y = word["box"][1]
                direction = word["direction"]
                point = Point(next_word_pos_x, next_word_pos_y)
                line = LineString(direction)
                x = np.array(point.coords[0])
                u = np.array(line.coords[0])
                v = np.array(line.coords[len(line.coords) - 1])
                n = v - u
                n /= np.linalg.norm(n, 2)
                P = u + n * np.dot(x - u, n)
                return (int(P[0]), int(P[1]))

            def distance_to_mid(word_point, word_box):
                point = Point(word_point["next_point"])
                box = word_box["box"]
                return abs(point.y - (box[1] + box[3]) / 2.0)  # abs( y - (y2+y1)/2 )

            def find_next_word(word, index, sorted_words):
                next_point = Point(word["next_point"])
                next_words = [other for other in sorted_words[index + 1:] if
                              Polygon(chunks(other["polygon"], 2)).contains(next_point)]
                if next_words:
                    return min(next_words, key=lambda x: distance_to_mid(word, x))
                else:
                    return None

            def find_previous_word(prev, word):
                if "previous_word" not in word.keys():
                    return prev
                else:
                    return min(prev, word["previous_word"], key=lambda x: distance_to_mid(x, word))

            for w in sorted_res:
                w["next_point"] = point_in_next_word(w)

            for i, w in enumerate(sorted_res):
                next_word = find_next_word(w, i, sorted_res)
                w["next_word"] = None
                if next_word:
                    better_previous = find_previous_word(w, next_word)
                    if better_previous == w:
                        w["next_word"] = next_word
                        if "previous_word" in next_word.keys():
                            next_word["previous_word"]["next_word"] = None
                        next_word["previous_word"] = w

            for w in sorted_res:
                if "previous_word" not in w.keys():
                    a = w
                    key_y = a["box"][1]
                    while key_y in lines.keys():
                        key_y = key_y + 1
                    lines[key_y] = [a]
                    while a["next_word"]:
                        a = a["next_word"]
                        lines[key_y].append(a)

            sorted_lines = sorted(lines.items(), key=lambda x: x[0])
            return ",".join([" ".join([w["seq_word"] for w in line]) for _, line in sorted_lines]), sorted_lines

        # apply pre-processing to image
        import datetime, time
        start_time = time.time()
        # print('transform', datetime.datetime.now())
        image = self.transforms(original_image)
        # convert to an ImageList, padded so that it is divisible by
        # cfg.DATALOADER.SIZE_DIVISIBILITY
        # print('to image list', datetime.datetime.now())
        image_list = to_image_list(image, self.cfg.DATALOADER.SIZE_DIVISIBILITY)
        image_list = image_list.to(self.device)
        # compute predictions
        with torch.no_grad():
            # print('predict', datetime.datetime.now())
            self.model.eval()
            predictions, _, _ = self.model(image_list)
            if not predictions or len(predictions) < 1:
                # print('no text detected')
                return [], [], {'label': '', 'details': []}
        # print('post process', datetime.datetime.now())
        global_predictions = predictions[0]
        char_predictions = predictions[1]
        char_mask = char_predictions['char_mask']
        char_boxes = char_predictions['boxes']
        words, rec_scores, rec_char_scores, char_polygons = self.process_char_mask(char_mask, char_boxes)
        detailed_seq_scores = char_predictions['detailed_seq_scores']
        seq_words = char_predictions['seq_outputs']
        seq_scores = char_predictions['seq_scores']
        global_predictions = [o.to(self.cpu_device) for o in global_predictions]

        # always single image is passed at a time
        global_prediction = global_predictions[0]

        # reshape prediction (a BoxList) into the original image size
        height, width = original_image.shape[:-1]
        test_image_width, test_image_height = global_prediction.size
        global_prediction = global_prediction.resize((width, height))
        resize_ratio = float(height) / test_image_height
        boxes = global_prediction.bbox.tolist()
        scores = global_prediction.get_field("scores").tolist()
        masks = global_prediction.get_field("mask").cpu().numpy()

        result_polygons = []
        result_words = []
        result_dicts = []

        for k, box in enumerate(boxes):
            score = scores[k]
            if score < self.confidence_threshold:
                continue
            box = list(map(int, box))
            mask = masks[k, 0, :, :]
            polygon = self.mask2polygon(mask, box, original_image.shape, threshold=0.5,
                                        output_polygon=self.output_polygon)

            if polygon is None:
                polygon = [box[0], box[1], box[2], box[1], box[2], box[3], box[0], box[3]]
            result_polygons.append(polygon)
            word = words[k]
            rec_score = rec_scores[k]
            char_score = rec_char_scores[k]
            seq_word = seq_words[k]
            seq_char_scores = seq_scores[k]
            seq_score = sum(seq_char_scores) / float(len(seq_char_scores))
            # spell_fix = lambda word: \
            #     [s.term for s in sym_spell.lookup(word, Verbosity.CLOSEST, max_edit_distance=2, include_unknown=True)][
            #         0]
            detailed_seq_score = detailed_seq_scores[k]
            detailed_seq_score = np.squeeze(np.array(detailed_seq_score), axis=1)
            # if 'total_text' in output_folder or 'cute80' in output_folder:
            #     result_log = [int(x * 1.0) for x in box[:4]] + polygon + [word] + [seq_word] + [score] + [rec_score] + [
            #         seq_score] + [char_score] + [detailed_seq_score] + [len(polygon)]
            # else:
            result_log = [int(x * 1.0) for x in box[:4]] + polygon + [word] + [seq_word] + [score] + [rec_score] + [
                seq_score] + [char_score] + [detailed_seq_score]
            # result_logs.append(result_log)
            if len(seq_word) > 0 and len(char_polygons[k]) > 0:
                d = {
                    "seq_word": seq_word if len(seq_word) < 4 else spell_fix(seq_word),
                    "seq_word_orig": seq_word,
                    "direction": mk_direction([[int(c * resize_ratio) for c in p] for p in char_polygons[k]]),
                    "word": word if len(word) < 4 else spell_fix(word),
                    "word_orig": word,
                    "box": [int(x * 1.0) for x in box[:4]],
                    "polygon": polygon,
                    "prob": score * seq_score
                }
                result_words.append(d['seq_word'])
                result_dicts.append(d)

        # default_logger.debug('done', datetime.datetime.now())
        label, details = line_detection(result_dicts)
        end_time = time.time()
        # default_logger.debug('cost time: %s' % (end_time - start_time))
        line_result = {'label': label, 'details': details}
        # line_result_words = []
        # line_result_polygons = []
        # for ocr_detail in line_result['details']:
        #     pass
        # line_result_words = [a[1][0]['seq_word'] for a in line_result['details']]
        # line_result_polygons = [a[1][0]['polygon'] for a in line_result['details']]
        line_result_words = [a['seq_word'] for a in result_dicts]
        line_result_polygons = [a['polygon'] for a in result_dicts]
        # return result_polygons, result_words, line_result
        return line_result_polygons, line_result_words, line_result

    # def process_char_mask(self, char_masks, boxes, threshold=192):
    #     texts, rec_scores = [], []
    #     for index in range(char_masks.shape[0]):
    #         box = list(boxes[index])
    #         box = list(map(int, box))
    #         text, rec_score, _, _ = getstr_grid(char_masks[index, :, :, :].copy(), box, threshold=threshold)
    #         texts.append(text)
    #         rec_scores.append(rec_score)
    #     return texts, rec_scores

    def process_char_mask(self, char_masks, boxes, threshold=192):
        texts, rec_scores, rec_char_scores, char_polygons = [], [], [], []
        for index in range(char_masks.shape[0]):
            box = list(boxes[index])
            box = list(map(int, box))
            text, rec_score, rec_char_score, char_polygon = getstr_grid(char_masks[index, :, :, :].copy(), box,
                                                                        threshold=threshold)
            texts.append(text)
            rec_scores.append(rec_score)
            rec_char_scores.append(rec_char_score)
            char_polygons.append(char_polygon)
            # segmss.append(segms)
        return texts, rec_scores, rec_char_scores, char_polygons

    def mask2polygon(self, mask, box, im_size, threshold=0.5, output_polygon=True):
        # mask 32*128
        image_width, image_height = im_size[1], im_size[0]
        box_h = box[3] - box[1]
        box_w = box[2] - box[0]
        cls_polys = (mask * 255).astype(np.uint8)
        poly_map = np.array(Image.fromarray(cls_polys).resize((box_w, box_h)))
        poly_map = poly_map.astype(np.float32) / 255
        poly_map = cv2.GaussianBlur(poly_map, (3, 3), sigmaX=3)
        ret, poly_map = cv2.threshold(poly_map, 0.5, 1, cv2.THRESH_BINARY)
        if output_polygon:
            SE1 = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
            poly_map = cv2.erode(poly_map, SE1)
            poly_map = cv2.dilate(poly_map, SE1);
            poly_map = cv2.morphologyEx(poly_map, cv2.MORPH_CLOSE, SE1)
            try:
                _, contours, _ = cv2.findContours((poly_map * 255).astype(np.uint8), cv2.RETR_LIST,
                                                  cv2.CHAIN_APPROX_NONE)
            except:
                contours, _ = cv2.findContours((poly_map * 255).astype(np.uint8), cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
            if len(contours) == 0:
                print(contours)
                print(len(contours))
                return None
            max_area = 0
            max_cnt = contours[0]
            for cnt in contours:
                area = cv2.contourArea(cnt)
                if area > max_area:
                    max_area = area
                    max_cnt = cnt
            perimeter = cv2.arcLength(max_cnt, True)
            epsilon = 0.01 * cv2.arcLength(max_cnt, True)
            approx = cv2.approxPolyDP(max_cnt, epsilon, True)
            pts = approx.reshape((-1, 2))
            pts[:, 0] = pts[:, 0] + box[0]
            pts[:, 1] = pts[:, 1] + box[1]
            polygon = list(pts.reshape((-1,)))
            polygon = list(map(int, polygon))
            if len(polygon) < 6:
                return None
        else:
            SE1 = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
            poly_map = cv2.erode(poly_map, SE1)
            poly_map = cv2.dilate(poly_map, SE1);
            poly_map = cv2.morphologyEx(poly_map, cv2.MORPH_CLOSE, SE1)
            idy, idx = np.where(poly_map == 1)
            xy = np.vstack((idx, idy))
            xy = np.transpose(xy)
            hull = cv2.convexHull(xy, clockwise=True)
            # reverse order of points.
            if hull is None:
                return None
            hull = hull[::-1]
            # find minimum area bounding box.
            rect = cv2.minAreaRect(hull)
            corners = cv2.boxPoints(rect)
            corners = np.array(corners, dtype="int")
            pts = get_tight_rect(corners, box[0], box[1], image_height, image_width, 1)
            polygon = [x * 1.0 for x in pts]
            polygon = list(map(int, polygon))
        return polygon

    def visualization(self, img, polygons, words):
        cur_img = copy.deepcopy(img)
        for polygon, word in zip(polygons, words):
            pts = np.array(polygon, np.int32)
            pts = pts.reshape((-1, 1, 2))
            xmin = min(pts[:, 0, 0])
            ymin = min(pts[:, 0, 1])
            r = random.randint(0, 255)
            g = random.randint(0, 255)
            b = random.randint(0, 255)
            cv2.polylines(cur_img, [pts], True, (b, g, r))
            cv2.putText(cur_img, word, (xmin, ymin), cv2.FONT_HERSHEY_TRIPLEX, 0.5, (b, g, r), 1)
        return cur_img
