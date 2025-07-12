#!/usr/bin/env python3
import sys
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
from geometry_msgs.msg import PointStamped, Point32, TransformStamped
from tf2_ros import StaticTransformBroadcaster, Buffer, TransformListener
from tf2_geometry_msgs import do_transform_point
from rclpy.duration import Duration

from item_tools_interfaces.srv import GetItemPoses


class ItemPoseService(Node):
    def __init__(self):
        super().__init__("item_pose_service")

        # パラメータ宣言
        self.declare_parameter("image_topic",   "/camera/color/image_raw")
        self.declare_parameter("info_topic",    "/camera/color/camera_info")
        self.declare_parameter("dict_name",     "DICT_4X4_50")
        self.declare_parameter("marker_length", 0.05)
        self.declare_parameter("sample_frames", 9)

        # キャリブ＆バッファ
        self.bridge = CvBridge()
        self.K = self.D = None
        N = self.get_parameter("sample_frames").value
        self.frame_buf  = collections.deque(maxlen=N)
        self.header_buf = collections.deque(maxlen=N)

        # ArUco 検出器
        dic = aruco.getPredefinedDictionary(
            getattr(aruco, self.get_parameter("dict_name").value))
        self.detector = aruco.ArucoDetector(dic, aruco.DetectorParameters())
        self.mlen     = float(self.get_parameter("marker_length").value)

        # サブスクライバー登録
        self.create_subscription(CameraInfo,
                                 self.get_parameter("info_topic").value,
                                 self.cb_info, 10)
        self.create_subscription(Image,
                                 self.get_parameter("image_topic").value,
                                 self.cb_image, 10)

        # YAML からマーカー info ロード
        yaml_path = Path(__file__).parent /"item_info.yaml"
        with open(yaml_path, "r") as f:
            cfg = yaml.safe_load(f)
        self.aruco_info = {
            int(m["id"]): {
                "name": m.get("name", ""),
                "barcode": int(m.get("barcode", 0))
            }
            for m in cfg["aruco"]["markers"]
        }

        # TF broadcaster と listener
        self.static_br   = StaticTransformBroadcaster(self)
        self.tf_buffer   = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        # サービス提供
        self.srv = self.create_service(
            GetItemPoses,
            "aruco/get_item_poses",
            self.handle_request)

        self.get_logger().info("ItemPoseService ready.")

    def cb_info(self, msg: CameraInfo):
        if self.K is None:
            self.K = np.asarray(msg.k).reshape(3, 3)
            self.D = np.asarray(msg.d, dtype=np.float64)

    def cb_image(self, msg: Image):
        img = self.bridge.imgmsg_to_cv2(msg, "bgr8")
        self.frame_buf.append(img)
        self.header_buf.append(msg.header)

    def handle_request(self, req, res):
        # キャリブ or フレーム未準備
        if self.K is None or not self.frame_buf:
            self.get_logger().warn("Calibration or frames not ready")
            return res

        # 各 ID の tvec を蓄積
        acc = {mid: [] for mid in req.ids}
        for frame in list(self.frame_buf):
            corners, ids, _ = self.detector.detectMarkers(frame)
            if ids is None:
                continue
            _, tvecs, _ = aruco.estimatePoseSingleMarkers(
                corners, self.mlen, self.K, self.D)
            for idx, mid in enumerate(ids.flatten()):
                m = int(mid)
                if m in acc:
                    acc[m].append(tvecs[idx].flatten())

        # 各 ID について検出されたものだけレスポンスに追加
        for mid in req.ids:
            vecs = acc[mid]
            if not vecs:
                    self.get_logger().info(f"ID {mid} not found")                # placeholder を入れて長さを揃える
                    res.positions.append(Point32())
                    res.item.append("")       # 空文字
                    res.barcode.append("0")
                    res.found.append(False)
                    continue

            # 平均 tvec (camera_color_optical_frame)
            mean_t = np.vstack(vecs).mean(axis=0)
            # PointStamped に詰めて camera_link へ変換
            p_opt = PointStamped()
            p_opt.header.frame_id = "camera_color_optical_frame"
            p_opt.header.stamp    = self.header_buf[-1].stamp
            p_opt.point.x, p_opt.point.y, p_opt.point.z = mean_t
            try:
                tf_opt2link = self.tf_buffer.lookup_transform(
                    "camera_link",
                    "camera_color_optical_frame",
                    rclpy.time.Time(),
                    Duration(seconds=1.0))
                p_link = do_transform_point(p_opt, tf_opt2link)
            except Exception as e:
                self.get_logger().error(f"Transform failed: {e}")
                p_link = p_opt

            # レスポンスに追加（camera_link 座標）
            res.positions.append(Point32(
                x=p_link.point.x,
                y=p_link.point.y,
                z=p_link.point.z
            ))
            res.item.append(str(self.aruco_info[mid]["name"]))
            res.barcode.append(str(self.aruco_info[mid]["barcode"]))
            res.found.append(True)

            # TF 配信（parent=camera_link）
            tf = TransformStamped()
            tf.header.frame_id = "camera_link"
            tf.header.stamp    = p_link.header.stamp
            tf.child_frame_id  = f"aruco_static_{mid}"
            tf.transform.translation.x = p_link.point.x
            tf.transform.translation.y = p_link.point.y
            tf.transform.translation.z = p_link.point.z
            tf.transform.rotation.w   = 1.0
            self.static_br.sendTransform(tf)

        return res


def main(args=None):
    rclpy.init(args=args)
    node = ItemPoseService()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()