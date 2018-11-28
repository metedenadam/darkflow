"""
tfnet secondary (helper) methods
"""
from ..utils.loader import create_loader
from time import time as timer
import tensorflow as tf
import numpy as np
import sys
import cv2
import os
import json
import xml.etree.ElementTree as ET
from glob import glob

old_graph_msg = 'Resolving old graph def {} (no guarantee)'

class Obj:
    def __init__(self, label, bbox):
        self.label = label
        self.bbox = bbox

def bb_intersection_over_union(boxA, boxB):
    # determine the (x, y)-coordinates of the intersection rectangle
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    # compute the area of intersection rectangle
    interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)

    # compute the area of both the prediction and ground-truth
    # rectangles
    boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
    boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)

    # compute the intersection over union by taking the intersection
    # area and dividing it by the sum of prediction + ground-truth
    # areas - the interesection area
    iou = interArea / float(boxAArea + boxBArea - interArea)

    # return the intersection over union value
    return iou

def json_to_object_list(pred_data):
    pred_objs = []
    for obj in pred_data:
        label = obj['label']
        xmin = obj['topleft']['x']
        ymin = obj['topleft']['y']
        xmax = obj['bottomright']['x']
        ymax = obj['bottomright']['y']
        box = [xmin, ymin, xmax, ymax]
        prediction = Obj(label, box)
        pred_objs.append(prediction)
    return pred_objs

def xml_to_object_list(ann_data):
    gt_objs = []
    for child in ann_data:
        if(child.tag == 'object'):
            for obj in child:
                if(obj.tag == 'name'):
                    label = obj.text
                if(obj.tag == 'bndbox'):
                    box = []
                    for bbx in obj:
                        box.append(int(bbx.text))
                    groundtruth = Obj(label, box)
            gt_objs.append(groundtruth)
    return gt_objs

def calc_accuracy(self):
    if self.FLAGS.train:
        annotations = self.FLAGS.val_annotation
        predictions = self.FLAGS.val_dataset + 'out/'
    else:
        annotations = self.FLAGS.imgdir_annotation
        predictions = self.FLAGS.imgdir + 'out/'

    pred_list = glob(predictions + '*')
    obj_count = 0
    true_count = 0
    for pred in pred_list:
        with open(pred) as f:
            pred_data = json.load(f)
            
        ann = annotations + (pred.split('/'))[4][:-5] + '.xml'
        ann_data = ET.parse(ann).getroot()
            
        pred_objs = json_to_object_list(pred_data)
        ann_objs = xml_to_object_list(ann_data)
        
        for pred_obj in pred_objs:
            obj_count += 1
            max_iou = 0
            for ann_obj in ann_objs:
                if pred_obj.label != ann_obj.label:
                    continue
                iou = bb_intersection_over_union(pred_obj.bbox, ann_obj.bbox)
                max_iou = max(max_iou, iou)
            if max_iou > 0.5:
                true_count += 1
    if obj_count == 0: obj_count = 1
    return 100 * true_count/obj_count

def build_train_op(self):
    self.framework.loss(self.out)
    self.say('Building {} train op'.format(self.meta['model']))
    optimizer = self._TRAINER[self.FLAGS.trainer](self.FLAGS.lr)
    gradients = optimizer.compute_gradients(self.framework.loss)
    self.train_op = optimizer.apply_gradients(gradients)

def load_from_ckpt(self):
    if self.FLAGS.load < 0: # load lastest ckpt
        with open(os.path.join(self.FLAGS.backup, 'checkpoint'), 'r') as f:
            last = f.readlines()[-1].strip()
            load_point = last.split(' ')[1]
            load_point = load_point.split('"')[1]
            load_point = load_point.split('-')[-1]
            self.FLAGS.load = int(load_point)
    
    load_point = os.path.join(self.FLAGS.backup, self.meta['name'])
    load_point = '{}-{}'.format(load_point, self.FLAGS.load)
    self.say('Loading from {}'.format(load_point))
    try: self.saver.restore(self.sess, load_point)
    except: load_old_graph(self, load_point)

def say(self, *msgs):
    if not self.FLAGS.verbalise:
        return
    msgs = list(msgs)
    for msg in msgs:
        if msg is None: continue
        print(msg)

def load_old_graph(self, ckpt): 
    ckpt_loader = create_loader(ckpt)
    self.say(old_graph_msg.format(ckpt))
    
    for var in tf.global_variables():
        name = var.name.split(':')[0]
        args = [name, var.get_shape()]
        val = ckpt_loader(args)
        assert val is not None, \
        'Cannot find and load {}'.format(var.name)
        shp = val.shape
        plh = tf.placeholder(tf.float32, shp)
        op = tf.assign(var, plh)
        self.sess.run(op, {plh: val})

def _get_fps(self, frame):
    elapsed = int()
    start = timer()
    preprocessed = self.framework.preprocess(frame)
    feed_dict = {self.inp: [preprocessed]}
    net_out = self.sess.run(self.out, feed_dict)[0]
    processed = self.framework.postprocess(net_out, frame, False)
    return timer() - start

def camera(self):
    file = self.FLAGS.demo
    SaveVideo = self.FLAGS.saveVideo
    
    if file == 'camera':
        file = 0
    else:
        assert os.path.isfile(file), \
        'file {} does not exist'.format(file)
        
    camera = cv2.VideoCapture(file)
    
    if file == 0:
        self.say('Press [ESC] to quit demo')
        
    assert camera.isOpened(), \
    'Cannot capture source'
    
    if file == 0:#camera window
        cv2.namedWindow('', 0)
        _, frame = camera.read()
        height, width, _ = frame.shape
        cv2.resizeWindow('', width, height)
    else:
        _, frame = camera.read()
        height, width, _ = frame.shape

    if SaveVideo:
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        if file == 0:#camera window
          fps = 1 / self._get_fps(frame)
          if fps < 1:
            fps = 1
        else:
            fps = round(camera.get(cv2.CAP_PROP_FPS))
        videoWriter = cv2.VideoWriter(
            'video.avi', fourcc, fps, (width, height))

    # buffers for demo in batch
    buffer_inp = list()
    buffer_pre = list()
    
    elapsed = int()
    start = timer()
    self.say('Press [ESC] to quit demo')
    # Loop through frames
    while camera.isOpened():
        elapsed += 1
        _, frame = camera.read()
        if frame is None:
            print ('\nEnd of Video')
            break
        preprocessed = self.framework.preprocess(frame)
        buffer_inp.append(frame)
        buffer_pre.append(preprocessed)
        
        # Only process and imshow when queue is full
        if elapsed % self.FLAGS.queue == 0:
            feed_dict = {self.inp: buffer_pre}
            net_out = self.sess.run(self.out, feed_dict)
            for img, single_out in zip(buffer_inp, net_out):
                postprocessed = self.framework.postprocess(
                    single_out, img, False)
                if SaveVideo:
                    videoWriter.write(postprocessed)
                if file == 0: #camera window
                    cv2.imshow('', postprocessed)
            # Clear Buffers
            buffer_inp = list()
            buffer_pre = list()

        if elapsed % 5 == 0:
            sys.stdout.write('\r')
            sys.stdout.write('{0:3.3f} FPS'.format(
                elapsed / (timer() - start)))
            sys.stdout.flush()
        if file == 0: #camera window
            choice = cv2.waitKey(1)
            if choice == 27: break

    sys.stdout.write('\n')
    if SaveVideo:
        videoWriter.release()
    camera.release()
    if file == 0: #camera window
        cv2.destroyAllWindows()

def to_darknet(self):
    darknet_ckpt = self.darknet

    with self.graph.as_default() as g:
        for var in tf.global_variables():
            name = var.name.split(':')[0]
            var_name = name.split('-')
            l_idx = int(var_name[0])
            w_sig = var_name[1].split('/')[-1]
            l = darknet_ckpt.layers[l_idx]
            l.w[w_sig] = var.eval(self.sess)

    for layer in darknet_ckpt.layers:
        for ph in layer.h:
            layer.h[ph] = None

    return darknet_ckpt


