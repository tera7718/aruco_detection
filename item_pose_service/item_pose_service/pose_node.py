#!/usr/bin/env python3
import sys
import datetime
import rclpy
import yaml
import numpy as np
import collections
import cv2
from cv2 import aruco
from cv_bridge import CvBridge
from pathlib import Path

from rclpy.node import Node
from sensor_msgs.msg import Image, CameraInfo
from geometry_msgs.msg import Point32, TransformStamped, PointStamped, Point
from tf2_ros import StaticTransformBroadcaster, Buffer, TransformListener
from tf2_geometry_msgs import do_transform_point
from rclpy.duration import Duration

from item_tools_interfaces.srv import GetItemPoses


class ItemPoseNode(Node):
    def __init__(self, ids):
        super().__init__("item_pose_node")

        self.declare_parameter("image_topic",  "/camera/color/image_raw")
        self.declare_parameter("info_topic",   "/camera/color/camera_info")
        self.declare_parameter("dict_name",    "DICT_4X4_50")
        self.declare_parameter("marker_length",0.05)
        self.declare_parameter("sample_frames",9)

        self.ids = [int(i) for i in ids]

        self.bridge = CvBridge()
        self.K = self.D = None
        N = int(self.get_parameter("sample_frames").value)
        self.frame_buf = collections.deque(maxlen=N)
        self.header_buf= collections.deque(maxlen=N)

        self.detector = aruco.ArucoDetector(
            aruco.getPredefinedDictionary(
                getattr(aruco, self.get_parameter("dict_name").value)),
            aruco.DetectorParameters())
        self.mlen = float(self.get_parameter("marker_length").value)

        self.create_subscription(CameraInfo,
                                 self.get_parameter("info_topic").value,
                                 self.cb_info, 10)
        self.create_subscription(Image,
                                 self.get_parameter("image_topic").value,
                                 self.cb_image, 10)

        self.srv = self.create_service(GetItemPoses,
                            "aruco/get_marker_poses",
                            self.handle_request)

        self.cli = self.create_client(GetItemPoses, "aruco/get_marker_poses")

        self.static_br = StaticTransformBroadcaster(self)

        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        yaml_path = Path(__file__).parent /"frames.yaml"
        with open(yaml_path, "r") as f:
            self.config = yaml.safe_load(f)
        # YAML 読み込み後…
        self.aruco_list = self.config['aruco']['markers']

        # id→dict の辞書を作成
        self.aruco_info = { m['id']: m for m in self.aruco_list }

        self.retry_timer = None
        self.future = None
        self.call_service_once()

        self.get_logger().info(f"Ready. Request IDs={self.ids}, buffer {N} frames")

    def cb_info(self, msg: CameraInfo):
        if self.K is None:
            self.K = np.asarray(msg.k).reshape(3, 3)
            self.D = np.asarray(msg.d, dtype=np.float64)

    def cb_image(self, msg: Image):
        try:
            cv_image = self.bridge.imgmsg_to_cv2(msg)
            if cv_image is None or cv_image.size == 0:
                self.get_logger().warn("Empty image received. Skipping.")
                return
            if msg.encoding == "rgb8":
                cv_image = cv2.cvtColor(cv_image, cv2.COLOR_RGB2BGR)
            self.frame_buf.append(cv_image)
            self.header_buf.append(msg.header)
        except Exception as e:
            import traceback
            self.get_logger().error(f"Image conversion error: {str(e)}")
            self.get_logger().error(traceback.format_exc())

    def handle_request(self, req, res):
        self.get_logger().info("handle_request called")
        if self.K is None or not self.frame_buf:
            self.get_logger().warn("Calibration or frame buffer not ready")
            return res

        acc = {mid: [] for mid in req.ids}
        for frame in list(self.frame_buf):
            corners, ids, _ = self.detector.detectMarkers(frame)
            if ids is None:
                continue
            rvecs, tvecs, _ = aruco.estimatePoseSingleMarkers(
                corners, self.mlen, self.K, self.D)
            for idx, mid in enumerate(ids.flatten()):
                mid = int(mid)
                if mid in acc:
                    acc[mid].append(tvecs[idx].flatten())

        last_header = self.header_buf[-1]
        for mid in req.ids:
            vecs = acc[mid]
            if vecs:
                mean = np.vstack(vecs).mean(axis=0)
                res.positions.append(Point32(x=float(mean[0]), y=float(mean[1]), z=float(mean[2])))
                res.found.append(True)
                tf = TransformStamped()
                tf.header = last_header
                tf.child_frame_id = f"aruco_static_{mid}"
                tf.transform.translation.x, tf.transform.translation.y, tf.transform.translation.z = mean
                tf.transform.rotation.w = 1.0
                self.static_br.sendTransform(tf)
            else:
                res.positions.append(Point32())
                res.found.append(False)
                self.get_logger().warn(f"ID {mid} not found in buffer")
        return res

    def call_service_once(self):
        if self.retry_timer:
            self.retry_timer.cancel()
            self.retry_timer = None
        if self.K is None or not self.frame_buf:
            self.get_logger().info("Waiting for calibration data and images...")
            self.retry_timer = self.create_timer(1.0, self.call_service_once)
            return
        if not self.cli.service_is_ready():
            self.get_logger().warn("Service not available yet")
            self.retry_timer = self.create_timer(1.0, self.call_service_once)
            return
        req = GetItemPoses.Request(ids=self.ids)
        self.future = self.cli.call_async(req)
        self.future.add_done_callback(self.response_callback)

    def response_callback(self, future):
        try:
            res = future.result()
        except Exception as e:
            self.get_logger().error(f"Service call failed: {e}")
            rclpy.shutdown()
            return
        self.process_response(res)

    def process_response(self, res):
        tf_cnt = 0
        for mid, pos, ok in zip(self.ids, res.positions, res.found):
            if not ok:
                self.get_logger().warn(f"ID {mid} not found")
                continue
            marker_pt = PointStamped()
            marker_pt.header.frame_id = "camera_color_optical_frame"
            marker_pt.header.stamp = self.get_clock().now().to_msg()
            marker_pt.point = Point(x=pos.x, y=pos.y, z=pos.z)
            try:
                tf_pt = self.tf_buffer.transform(marker_pt, "camera_link", timeout=Duration(seconds=1.0))
                self.get_logger().info(
                    f"ID {mid}: x={tf_pt.point.x} y={tf_pt.point.y} z={tf_pt.point.z}")

                aruco_base_pos = np.array([tf_pt.point.x, tf_pt.point.y, tf_pt.point.z])
                resolved_positions = {"aruco": aruco_base_pos}

                aruco_tf = TransformStamped()
                aruco_tf.header.frame_id = "camera_link"
                aruco_tf.header.stamp = tf_pt.header.stamp
                aruco_tf.child_frame_id = f"aruco_static_{mid}"
                aruco_tf.transform.translation.x = aruco_base_pos[0]
                aruco_tf.transform.translation.y = aruco_base_pos[1]
                aruco_tf.transform.translation.z = aruco_base_pos[2]
                aruco_tf.transform.rotation.w = 1.0
                self.static_br.sendTransform(aruco_tf)

                aruco_info = self.aruco_info[mid]
                self.get_logger().info(f"ID {mid} info: {aruco_info}")
                tf_cnt += 1


            except Exception as e:
                self.get_logger().error(f"TF transform failed for ID {mid}: {e}")
        self.get_logger().info(f"Sent {tf_cnt} static transforms")

        # ✅ 自動終了
        self.get_logger().info("Shutting down after successful processing.")
        # rclpy.shutdown()



def main(args=None):
    rclpy.init(args=args)
    if len(sys.argv) < 2:
        print("Usage: ros2 run <package> <node> <id1> <id2> ...")
        sys.exit(1)

    node = ItemPoseNode(sys.argv[1:])
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        if rclpy.ok():
            node.destroy_node()
            rclpy.shutdown()


if __name__ == "__main__":
    main()
