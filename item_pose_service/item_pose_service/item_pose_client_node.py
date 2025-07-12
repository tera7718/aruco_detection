#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from item_tools_interfaces.srv import GetItemPoses
from geometry_msgs.msg import Point32

class ItemPoseClient(Node):
    def __init__(self, ids):
        super().__init__('item_pose_client')

        # ── ここで直接マーカーIDを指定 ──
        self.ids = ids # 例えばID=2,5,8を検出したい

        # サービスクライアントの作成
        self.cli = self.create_client(GetItemPoses, 'aruco/get_item_poses')
        self.get_logger().info('Waiting for service /aruco/get_marker_poses...')
        if not self.cli.wait_for_service(timeout_sec=5.0):
            self.get_logger().error('Service not available, quitting.')
            rclpy.shutdown()
            return

        # リクエスト送信
        req = GetItemPoses.Request()
        req.ids = self.ids
        self.get_logger().info(f'Calling service with IDs: {self.ids}')
        self.future = self.cli.call_async(req)
        self.future.add_done_callback(self.callback)

    def callback(self, future):
        try:
            res = future.result()
        except Exception as e:
            self.get_logger().error(f'Service call failed: {e}')
        else:
            # ids と同じ順序で barcode, item, positions, found が返ってくる
            for mid, item, pos, bc, found in zip(
                    self.ids,
                    res.item,
                    res.positions,
                    res.barcode,
                    res.found):
                if found:
                    self.get_logger().info(
                        f'ID {mid}: item="{item}", barcode={bc}, '
                        f'pos=({pos.x:.3f}, {pos.y:.3f}, {pos.z:.3f})'
                    )
                else:
                    self.get_logger().warn(f'ID {mid}: not found')
        rclpy.shutdown()

def main(args=None, ids=[2, 5, 8]):
    rclpy.init(args=args)
    client = ItemPoseClient(ids)
    rclpy.spin(client)

if __name__ == '__main__':
    main()