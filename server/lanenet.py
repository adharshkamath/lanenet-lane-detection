import argparse
import os.path as ops
import time
import base64

import cv2
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), "../"))

from lanenet_model import lanenet
from lanenet_model import lanenet_postprocess
from local_utils.config_utils import parse_config_utils
from local_utils.log_util import init_logger

CFG = parse_config_utils.lanenet_cfg
LOG = init_logger.get_logger(log_file_name_prefix="lanenet_inference")


class LaneNetServer:
    def __init__(self, weights_path):
        self.weights_path = weights_path
        self.postprocessor = lanenet_postprocess.LaneNetPostProcessor(cfg=CFG)
        self.input_tensor = tf.compat.v1.placeholder(
            dtype=tf.float32, shape=[1, 256, 512, 3], name="input_tensor"
        )
        self.net = lanenet.LaneNet(phase="test", cfg=CFG)
        binary_seg_ret, instance_seg_ret = self.net.inference(
            input_tensor=self.input_tensor, name="LaneNet"
        )
        self.binary_seg_ret = binary_seg_ret
        self.instance_seg_ret = instance_seg_ret

        sess_config = tf.compat.v1.ConfigProto(
            device_count={"CPU": 1, "GPU": 0},
            allow_soft_placement=True,
            log_device_placement=False,
        )
        self.sess = tf.compat.v1.InteractiveSession(config=sess_config)
        with tf.variable_scope(name_or_scope="moving_avg"):
            variable_averages = tf.train.ExponentialMovingAverage(
                CFG.SOLVER.MOVING_AVE_DECAY
            )
            variables_to_restore = variable_averages.variables_to_restore()
 
        self.saver = tf.train.Saver(variables_to_restore)
        with self.sess.as_default():
            self.saver.restore(sess=self.sess, save_path=weights_path)

    def minmax_scale(self, input_arr):
        min_val = np.min(input_arr)
        max_val = np.max(input_arr)

        output_arr = (input_arr - min_val) * 255.0 / (max_val - min_val)

        return output_arr

    def run_inference(self, image, with_lane_fit=True):
        image = cv2.imdecode(np.frombuffer(image.read(), np.uint8), -1)


        LOG.info("Start reading image and preprocessing")
        mask_image = None
        image_vis = None
        embedding_image = None
        binary_seg_image = None
        lane_params = []
        t_start = time.time()
        image_vis = image
        image = cv2.resize(image, (512, 256), interpolation=cv2.INTER_LINEAR)
        image = image / 127.5 - 1.0
        LOG.info(
            "Image load complete, cost time: {:.5f}s".format(time.time() - t_start)
        )

        with self.sess.as_default():
            t_start = time.time()
            binary_seg_image, instance_seg_image = self.sess.run(
                [self.binary_seg_ret, self.instance_seg_ret], feed_dict={self.input_tensor: [image]}
            )
            t_cost = time.time() - t_start
            LOG.info("Single image inference cost time: {:.5f}s".format(t_cost))

            postprocess_result = self.postprocessor.postprocess(
                binary_seg_result=binary_seg_image[0],
                instance_seg_result=instance_seg_image[0],
                source_image=image_vis,
                with_lane_fit=with_lane_fit,
                data_source="tusimple",
            )
            mask_image = postprocess_result["mask_image"]
            if with_lane_fit:
                lane_params = postprocess_result["fit_params"]
                LOG.info("Model successfully fitted {:d} lanes".format(len(lane_params)))
                for i in range(len(lane_params)):
                    LOG.info(
                        "Inferred 2nd order lane {:d} curve param: {}".format(
                            i + 1, lane_params[i]
                        )
                    )
            else:
                lane_params = []

            for i in range(CFG.MODEL.EMBEDDING_FEATS_DIMS):
                instance_seg_image[0][:, :, i] = self.minmax_scale(
                    instance_seg_image[0][:, :, i]
                )
            embedding_image = np.array(instance_seg_image[0], np.uint8)

        result = {
            "mask_image": mask_image[:, :, (2, 1, 0)],
            "src_image": image_vis[:, :, (2, 1, 0)],
            "instance_image": embedding_image[:, :, (2, 1, 0)],
            "binary_image": binary_seg_image[0] * 255,
        }
        # convert all images to base64
        for key in result.keys():
            result[key] = base64.b64encode(cv2.imencode(".png", result[key])[1]).decode("utf-8")

        result["lane_params"] = np.array(lane_params).tolist()

        return result

    def __del__(self):
        self.sess.close()
