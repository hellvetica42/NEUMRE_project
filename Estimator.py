import posenet 

class Estimator():

    def __init__(self, sess):
        self.sess = sess

        print("Loading posenet model")

        self.model_cfg, self.model_outputs = posenet.load_model(101,sess)

        if self.model_cfg is not None:
            print("Model loaded")

        self.output_stride = self.model_cfg['output_stride']
        pass

    def get_keypoints(self, img, scale_factor=0.7125):

        input_image, display_image, output_scale = posenet.process_input(img, scale_factor, self.output_stride)

        heatmaps_result, offsets_result, displacement_fwd_result, displacement_bwd_result = self.sess.run(
            self.model_outputs,
            feed_dict={'image:0': input_image}
        )

        pose_scores, keypoint_scores, keypoint_coords = posenet.decode_multi.decode_multiple_poses(
            heatmaps_result.squeeze(axis=0),
            offsets_result.squeeze(axis=0),
            displacement_fwd_result.squeeze(axis=0),
            displacement_bwd_result.squeeze(axis=0),
            output_stride=self.output_stride,
            max_pose_detections=1,
            min_pose_score=0.1)

        keypoint_coords *= output_scale

        return keypoint_coords

