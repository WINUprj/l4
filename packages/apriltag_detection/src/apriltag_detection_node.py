#!/usr/bin/env python3
import yaml

import rospy
from duckietown.dtros import DTROS, NodeType
from std_msgs.msg import String 
from sensor_msgs.msg import CompressedImage

import numpy as np
import cv2
from dt_apriltags import Detector

def read_yaml_file(path):
    with open(path, 'r') as f:
        try:
            yaml_dict = yaml.safe_load(f)
            return yaml_dict
        except yaml.YAMLError as exc:
            print(f"YAML syntax error. File: {path}. Exc: {exc}")
            rospy.signal_shutdown()
            return


def parse_calib_params(int_path=None, ext_path=None):
    # Load dictionaries from files
    int_dict, ext_dict = None, None
    if int_path:
        int_dict = read_yaml_file(int_path)
    if ext_path:
        ext_dict = read_yaml_file(ext_path)
    
    # Reconstruct the matrices from loaded dictionaries
    camera_mat, distort_coef, proj_mat = None, None, None
    hom_mat = None
    if int_dict:
        camera_mat = np.array(list(map(np.float32, int_dict["camera_matrix"]["data"]))).reshape((3, 3))
        distort_coef = np.array(list(map(np.float32, int_dict["distortion_coefficients"]["data"]))).reshape((1, 5))
        proj_mat = np.array(list(map(np.float32, int_dict["projection_matrix"]["data"]))).reshape((3, 4))
    if ext_dict:
        hom_mat = np.array(list(map(np.float32, ext_dict["homography"]))).reshape((3, 3))

    return (camera_mat, distort_coef, proj_mat, hom_mat)


class AprilTagDetectionNode(DTROS):
    def __init__(self, node_name="apriltag_detection"):
        super(AprilTagDetectionNode, self).__init__(
            node_name=node_name,
            node_type=NodeType.GENERIC
        )

        # Get arguments from a launch file
        self._veh = rospy.get_param("~veh")
        self._int_path = rospy.get_param("~int_file")
        self._ext_path = rospy.get_param("~ext_file")

        # Prepare apriltag detector
        self.apriltag_detector = Detector(families="tag36h11",
                                          nthreads=1)

        # Initialize matrices
        self.camera_mat, self.distort_coef, self.proj_mat, self.hom_mat = \
            parse_calib_params(self._int_path, self._ext_path)
        
        # Subscriber
        self.sub_cam = rospy.Subscriber(
            f"/{self._veh}/{node_name}/image/compressed",
            CompressedImage,
            self.cb_cam
        )

        # Publishers
        self.pub_info = rospy.Publisher(
            f"/{self._veh}/{node_name}/area",
            String,
            queue_size=1
        )

    def undistort(self, img):
        return cv2.undistort(img,
                             self.camera_mat,
                             self.distort_coef,
                             None,
                             self.camera_mat)

    def cb_cam(self, msg):
        # Reconstruct byte sequence to image
        img = np.frombuffer(msg.data, np.uint8) 
        img = cv2.imdecode(img, cv2.IMREAD_COLOR)
        
        # Undistort an image
        img = self.undistort(img)

        # Convert image to grayscale
        gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # Detect an apriltag
        c_params = [self.camera_mat[0,0],
                    self.camera_mat[1,1],
                    self.camera_mat[0,2],
                    self.camera_mat[1,2]]

        tags = self.apriltag_detector.detect(gray_img, True, c_params, 0.065)

        mx_area = 0
        mx_tag_id = 0
        if len(tags) > 0:
            # Take a tag with maximum area
            for tag in tags:
                (ptA, ptB, ptC, _) = tag.corners

                ptA = (int(ptA[0]), int(ptA[1]))
                ptB = (int(ptB[0]), int(ptB[1]))
                ptC = (int(ptC[0]), int(ptC[1]))

                area = int(abs(ptA[0]-ptB[0]) * abs(ptB[1] * ptC[1]))
                if area > mx_area:
                    mx_area = area
                    mx_tag_id = tag.id

        # Publish maximum area and tag ids
        msg = String()
        msg.data = f"{mx_area},{mx_tag_id}"
        self.pub_info.publish(msg)
        
if __name__ == "__main__":
    apriltag_ar = AprilTagDetectionNode()
    rate = rospy.Rate(8)
    while not rospy.is_shutdown():
        rate.sleep()