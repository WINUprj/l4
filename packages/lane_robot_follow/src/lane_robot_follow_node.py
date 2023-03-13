#!/usr/bin/env python3
import rospy

from duckietown.dtros import DTROS, NodeType
from std_msgs.msg import String, Float32
from duckietown_msgs.msg import BoolStamped, VehicleCorners
from sensor_msgs.msg import CompressedImage
from turbojpeg import TurboJPEG
import cv2
import numpy as np
from duckietown_msgs.msg import Twist2DStamped

ROAD_MASK = [(20, 60, 0), (50, 255, 255)]
DEBUG = False
ENGLISH = False

class LaneAndRobotFollowNode(DTROS):
    def __init__(self, node_name):
        super(LaneAndRobotFollowNode, self).__init__(
            node_name=node_name,
            node_type=NodeType.GENERIC
        )
        self.node_name = node_name
        self.veh = rospy.get_param("~veh")
        
        self.cur_dist_to_leader = 1.5

        # Subscribers
        self.sub = rospy.Subscriber(
            "/" + self.veh + "/camera_node/image/compressed",
            CompressedImage,
            self.callback,
            queue_size=1,
            buff_size="20MB"
        )
        self.sub_distance = rospy.Subscriber(
            f"/{self.veh}/duckiebot_distance_node/distance",
            Float32,
            self.cb_distance
        )
        self.sub_centers = rospy.Subscriber(
            f"/{self.veh}/duckiebot_detection_node/centers",
            VehicleCorners,
            self.cb_corners
        )
        self.sub_detect = rospy.Subscriber(
            f"/{self.veh}/duckiebot_detection_node/detection",
            BoolStamped,
            self.cb_detect
        )

        # Publishers
        self.pub = rospy.Publisher(
            "/" + self.veh + "/output/image/mask/compressed",
            CompressedImage,
            queue_size=1
        )
        self.vel_pub = rospy.Publisher(
            "/" + self.veh + "/car_cmd_switch_node/cmd",
            Twist2DStamped,
            queue_size=1
        )

        self.jpeg = TurboJPEG()

        self.loginfo("Initialized")

        # PID Variables
        self.proportional = None
        if ENGLISH:
            self.offset = -220
        else:
            self.offset = 220
        self.velocity = 0.35
        self.twist = Twist2DStamped(v=self.velocity, omega=0)

        # PID related terms
        self.P = 0.037
        self.D = -0.004
        self.last_error = 0
        self.last_time = rospy.get_time()

        # Thresholds
        self.dist_thresh = 0.45
        self.eps = 0.003
        self.area_thresh = 10000        # TODO: Need adjustment on this
        
        # Variables to track on
        self.in_front = False
        self.center = [-1, -1, -1]
        self.right_turn = False
        self.left_turn = False

        # Shutdown hook
        rospy.on_shutdown(self.hook)

    def cb_distance(self, msg):
        self.cur_dist_to_leader = msg.data
        print(self.cur_dist_to_leader)

    def cb_corners(self, msg):
        # print("Length of corners list", len(msg.corners))
        # self.center = [msg.corners.x,
        #                msg.corners.y,
        #                msg.corners.z]
        pass

    def cb_detect(self, msg):
        self.in_front = msg.data

    def callback(self, msg):
        if not self.in_front:
            img = self.jpeg.decode(msg.data)
            crop = img[300:-1, :, :]
            crop_width = crop.shape[1]
            hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
            mask = cv2.inRange(hsv, ROAD_MASK[0], ROAD_MASK[1])
            crop = cv2.bitwise_and(crop, crop, mask=mask)
            contours, hierarchy = cv2.findContours(mask,
                                                cv2.RETR_EXTERNAL,
                                                cv2.CHAIN_APPROX_NONE)

            # Search for lane in front
            max_area = 20
            max_idx = -1
            for i in range(len(contours)):
                area = cv2.contourArea(contours[i])
                if area > max_area:
                    max_idx = i
                    max_area = area

            if max_idx != -1:
                M = cv2.moments(contours[max_idx])
                try:
                    cx = int(M['m10'] / M['m00'])
                    cy = int(M['m01'] / M['m00'])
                    self.proportional = cx - int(crop_width / 2) + self.offset
                    if DEBUG:
                        cv2.drawContours(crop, contours, max_idx, (0, 255, 0), 3)
                        cv2.circle(crop, (cx, cy), 7, (0, 0, 255), -1)
                except:
                    pass
            else:
                self.proportional = None

            if DEBUG:
                rect_img_msg = CompressedImage(format="jpeg", data=self.jpeg.encode(crop))
                self.pub.publish(rect_img_msg)
        
        else:
            # Follow the robot in front
            self.proportional = self.center[0] - int(crop_width / 2)

    def stop(self):
        """Stop the vehicle completely."""
        self.twist.v = 0
        self.twist.omega = 0
    
    def turn(self, right=True):
        """Turn the car at the intersection."""
        if right:
            self.twist.v = self.velocity
            self.twist.omega = -3.3
        else:
            self.twist.v = self.velocity 
            self.twist.omega = 2.2

    def drive(self):
        if self.proportional is None:
            self.twist.omega = 0
        else:
            # P Term
            P = -self.proportional * self.P

            # D Term
            d_error = (self.proportional - self.last_error) / (rospy.get_time() - self.last_time)
            self.last_error = self.proportional
            self.last_time = rospy.get_time()
            D = d_error * self.D

            self.twist.v = self.velocity * min(self.cur_dist_to_leader, 1)
            self.twist.omega = P + D

            if DEBUG:
                self.loginfo(self.proportional, P, D, self.twist.omega, self.twist.v)

        if self.cur_dist_to_leader < self.dist_thresh + self.eps and self.in_front:
            # Stop when distance to the bot in front lower the distance threshold
            self.twist.v = 0
            self.twist.omega = 0

        self.vel_pub.publish(self.twist)

    def hook(self):
        print("SHUTTING DOWN")
        self.twist.v = 0
        self.twist.omega = 0
        self.vel_pub.publish(self.twist)
        for i in range(8):
            self.vel_pub.publish(self.twist)


if __name__ == "__main__":
    node = LaneAndRobotFollowNode("lane_robot_follow_node")
    rate = rospy.Rate(8)  # 8hz
    while not rospy.is_shutdown():
        node.drive()
        rate.sleep()