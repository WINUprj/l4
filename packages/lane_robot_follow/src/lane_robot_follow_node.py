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
# Mask resource: https://cvexplained.wordpress.com/2020/04/28/color-detection-hsv/#:~:text=The%20HSV%20values%20for%20true,10%20and%20160%20to%20180.
STOP_MASK = [[(0, 100, 20), (10, 255, 255)], [(160, 100, 20), (179, 255, 255)]]
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

        ### Subscribers
        # Camera images
        self.sub = rospy.Subscriber(
            "/" + self.veh + "/camera_node/image/compressed",
            CompressedImage,
            self.callback,
            queue_size=1,
            buff_size="20MB"
        )
        # Leader duckiebot distance
        self.sub_distance = rospy.Subscriber(
            f"/{self.veh}/duckiebot_distance_node/distance",
            Float32,
            self.cb_distance
        )
        # Leader duckiebot centers
        self.sub_centers = rospy.Subscriber(
            f"/{self.veh}/duckiebot_detection_node/centers",
            VehicleCorners,
            self.cb_corners
        )
        # Leader duckiebot detection results
        self.sub_detect = rospy.Subscriber(
            f"/{self.veh}/duckiebot_detection_node/detection",
            BoolStamped,
            self.cb_detect
        )
        # Apriltag information
        self.sub_detect = rospy.Subscriber(
            f"/{self.veh}/apriltag_detection/area",
            String,
            self.cb_apriltag_area
        )

        ### Publishers
        # Publish mask image for debug
        self.pub = rospy.Publisher(
            "/" + self.veh + "/output/image/mask/compressed",
            CompressedImage,
            queue_size=1
        )
        # Publish car command to control robot
        self.vel_pub = rospy.Publisher(
            "/" + self.veh + "/car_cmd_switch_node/cmd",
            Twist2DStamped,
            queue_size=1
        )

        self.jpeg = TurboJPEG()

        self.loginfo("Initialized")

        # Image related parameters
        self.width = None
        self.lower_thresh = 400

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
        
        # Variables to track on
        self.in_front = False       # indicates whether leader bot is in front
        self.is_stop = False
        self.center_x = -1 

        # Apriltag-related data
        self.t_intersection_right = [58, 133]
        self.t_intersection_left = [153, 62]
        self.stop_sign_ids = [162, 169]
        self.at_id = -1

        # Timer for stopping at the intersection
        self.t_stop = 4     # stop for this amount of seconds at intersections
        self.t_start = 0

        # Timer for turning/going straight at the intersection
        self.turning = False 
        self.t_turn = 1
        self.t_turn_start = 0

        # Shutdown hook
        rospy.on_shutdown(self.hook)

    def cb_distance(self, msg):
        self.cur_dist_to_leader = msg.data
        print(self.cur_dist_to_leader)

    def cb_corners(self, msg):
        if len(msg.corners) > 0:
            # Get x value where middle of leading bot exists
            self.center_x = int(
                np.mean([msg.corners[i].x for i in range(len(msg.corners))])
            )

    def cb_detect(self, msg):
        self.in_front = msg.data

    def cb_apriltag_area(self, msg):
        self.at_id = msg.data
    
    def get_max_contour(self, contours):
        max_area = 20
        max_idx = -1
        for i in range(len(contours)):
            area = cv2.contourArea(contours[i])
            if area > max_area:
                max_idx = i
                max_area = area

        return max_area, max_idx

    def get_contour_center(contours, idx):
        x, y = -1, -1
        if idx != -1:
            M = cv2.moments(contours[idx])
            try:
                x = int(M['m10'] / M['m00'])
                y = int(M['m01'] / M['m00'])
            except:
                pass
        
        return x, y

    def callback(self, msg):
        img = self.jpeg.decode(msg.data)

        if self.width == None:
            self.width = img.shape[1]
        
        # Check for the red line for all the incoming images
        red_crop = img[300:-1, 100:-100, :]
        red_crop_hsv = cv2.cvtColor(red_crop, cv2.COLOR_BGR2HSV)
        lower_mask = cv2.inRange(red_crop_hsv, STOP_MASK[0][0], STOP_MASK[0][1])
        upper_mask = cv2.inRange(red_crop_hsv, STOP_MASK[1][0], STOP_MASK[1][1])
        full_mask = lower_mask + upper_mask
        
        # Get contour and check if contour is "close enough" to car itself
        red_contours, _ = cv2.findContours(mask,
                                           cv2.RETR_EXTERNAL,
                                           cv2.CHAIN_APPROX_NONE)
        _, mx_idx = self.get_max_contour(red_contours)
        rx, ry = self.get_contour_center(red_contours, mx_idx)
        if not self.is_stop and not self.turning and \
           mx_idx != -1 and rx != -1 and ry != -1 and \
           ry >= self.lower_thresh:
            # Indicate the stop when car is DRIVING (i.e. not at the stop state
            # or passing through the intersections).
            self.is_stop = True
            self.t_start = rospy.get_rostime().secs

        if not self.in_front:
            # Preprocess image and extract contours for yellow line on road
            crop = img[300:-1, :, :]
            # crop_width = crop.shape[1]
            hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
            mask = cv2.inRange(hsv, ROAD_MASK[0], ROAD_MASK[1])
            crop = cv2.bitwise_and(crop, crop, mask=mask)
            contours, hierarchy = cv2.findContours(mask,
                                                cv2.RETR_EXTERNAL,
                                                cv2.CHAIN_APPROX_NONE)

            # Search for lane in front
            _, max_idx = self.get_max_contour(contours)
            cx, cy = self.get_contourr_center(contours, max_idx)
            
            if max_idx == -1:
                self.proportional = None
            elif cx != -1 and cy != -1:
                self.proportional = cx - int(self.width / 2) + self.offset

            if DEBUG:
                rect_img_msg = CompressedImage(format="jpeg", data=self.jpeg.encode(crop))
                self.pub.publish(rect_img_msg)
        
        else:
            # Follow the robot in front
            self.proportional = self.center_x - int(img.shape[1] / 2)

    def stop(self):
        """Stop the vehicle completely."""
        self.twist.v = 0
        self.twist.omega = 0
    
    def straight(self):
        """Move vehicle in forward direction."""
        self.twist.v = self.velocity
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
        ### Handle all the variables/flags which are independent of incoming messages 
        # print("Stopping? : ", self.is_stop, rospy.get_rostime().secs - self.t_start)
        # print("Turning? : ", self.turning, rospy.get_rostime().secs - self.t_turn_start)
        if self.is_stop and rospy.get_rostime().secs - self.t_start >= self.t_stop:
            # Move a bit when waiting duration is over
            self.is_stop = False
            self.turning = True
            self.t_turn_start = rospy.get_rostime().secs
        elif self.turning and rospy.get_rostime().secs - self.t_turn_start >= self.t_turn:
            # Switch from manual control to PID control (TURN to DRIVE)
            self.turning = False

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

            # Assign PID control values as the default values
            self.twist.v = self.velocity * min(self.cur_dist_to_leader, 1)
            self.twist.omega = P + D

            if DEBUG:
                self.loginfo(self.proportional, P, D, self.twist.omega, self.twist.v)
        
        # Handle special cases
        if (self.in_front and self.cur_dist_to_leader < self.dist_thresh + self.eps) or \
           (self.is_stop and rospy.get_rostime().secs - self.t_start < self.t_stop):
            # Track the center of leader robot to decide the direction to turn
            self.stop()
        elif self.turning and rospy.get_rostime().secs - self.t_turn_start < self.t_turn_start:
            if self.in_front or self.center_x == -1:
                # If one detect leader robot in front, or never detected the robot,
                # circle around the Duckietown
                self.straight()
            elif not self.in_front:
                if self.center_x < self.width // 2 or self.at_id in self.t_intersection_left:
                    self.turn(False)
                else:
                    self.turn(True)
            
        # Publish the resultant control values
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