"""天工机器人 ROS2 输入输出。"""

from __future__ import annotations

import cv2
import numpy as np
from openpi_client import image_tools
from rclpy.node import Node
from rclpy.qos import QoSHistoryPolicy
from rclpy.qos import QoSProfile
from rclpy.qos import QoSReliabilityPolicy

from bodyctrl_msgs.msg import CmdSetMotorPosition
from bodyctrl_msgs.msg import MotorStatusMsg
from bodyctrl_msgs.msg import SetMotorPosition

from tienkung_config import Args
from tienkung_config import MODEL_IMAGE_SIZE
from tienkung_config import RobotLayout

try:
    from foxglove_msgs.msg import CompressedVideo

    COMPRESSED_IMAGE_MSG = CompressedVideo
except ImportError:
    from sensor_msgs.msg import CompressedImage

    COMPRESSED_IMAGE_MSG = CompressedImage


def decode_compressed_image(msg) -> np.ndarray | None:
    """把压缩图像消息解码为 RGB uint8。"""
    if not msg.data:
        return None
    image_bgr = cv2.imdecode(np.frombuffer(msg.data, np.uint8), cv2.IMREAD_COLOR)
    if image_bgr is None:
        return None
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    return image_tools.convert_to_uint8(image_tools.resize_with_pad(image_rgb, MODEL_IMAGE_SIZE, MODEL_IMAGE_SIZE))


class TienkungRosIO:
    """封装天工 ROS 订阅、状态缓存和命令发布。"""

    def __init__(self, node: Node, args: Args, layout: RobotLayout) -> None:
        self._node = node
        self._args = args
        self._layout = layout
        self._latest_motor_positions: dict[int, float] = {}
        self._images: dict[str, np.ndarray | None] = {"base": None, "left": None, "right": None}

        qos_best_effort = QoSProfile(
            reliability=QoSReliabilityPolicy.BEST_EFFORT,
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=10,
        )
        qos_reliable = QoSProfile(
            reliability=QoSReliabilityPolicy.RELIABLE,
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=10,
        )

        node.create_subscription(MotorStatusMsg, args.arm_status_topic, self._motor_status_callback, qos_best_effort)
        node.create_subscription(
            COMPRESSED_IMAGE_MSG,
            args.base_image_topic,
            lambda msg: self._image_callback("base", msg),
            qos_best_effort,
        )
        node.create_subscription(
            COMPRESSED_IMAGE_MSG,
            args.left_wrist_image_topic,
            lambda msg: self._image_callback("left", msg),
            qos_best_effort,
        )
        node.create_subscription(
            COMPRESSED_IMAGE_MSG,
            args.right_wrist_image_topic,
            lambda msg: self._image_callback("right", msg),
            qos_best_effort,
        )
        self._command_pub = node.create_publisher(CmdSetMotorPosition, args.arm_command_topic, qos_reliable)

    def _motor_status_callback(self, msg: MotorStatusMsg) -> None:
        """缓存最新电机位置。"""
        for status in msg.status:
            self._latest_motor_positions[int(status.name)] = float(status.pos)

    def _image_callback(self, name: str, msg) -> None:
        """缓存最新图像。"""
        image = decode_compressed_image(msg)
        if image is not None:
            self._images[name] = image

    def is_ready(self) -> bool:
        """检查三路图像和 26 维状态是否都已准备。"""
        motors_ready = all(joint_id in self._latest_motor_positions for joint_id in self._layout.state_joint_ids)
        images_ready = all(image is not None for image in self._images.values())
        if not motors_ready or not images_ready:
            self._node.get_logger().info(
                f"等待数据... motors={motors_ready}, base={self._images['base'] is not None}, "
                f"left={self._images['left'] is not None}, right={self._images['right'] is not None}",
                throttle_duration_sec=2.0,
            )
        return motors_ready and images_ready

    def get_state(self) -> np.ndarray:
        """按训练约定拼接 26 维状态。"""
        state = np.array(
            [self._latest_motor_positions[joint_id] for joint_id in self._layout.state_joint_ids],
            dtype=np.float32,
        )
        return np.clip(state, self._layout.lower_limits, self._layout.upper_limits)

    def build_observation(self, prompt: str) -> dict:
        """构造 policy server 请求。"""
        return {
            "observation/base_image": self._images["base"],
            "observation/left_image": self._images["left"],
            "observation/right_image": self._images["right"],
            "observation/state": self.get_state(),
            "prompt": prompt,
        }

    def publish_action(self, action: np.ndarray) -> None:
        """发布 26 维绝对位置命令。"""
        action = np.clip(np.asarray(action, dtype=np.float32), self._layout.lower_limits, self._layout.upper_limits)
        cmd_msg = CmdSetMotorPosition()
        for joint_id, target in zip(self._layout.state_joint_ids, action, strict=True):
            motor_cmd = SetMotorPosition()
            motor_cmd.name = int(joint_id)
            motor_cmd.pos = float(target)
            motor_cmd.spd = float(self._args.command_speed)
            motor_cmd.cur = float(self._args.command_current)
            cmd_msg.cmds.append(motor_cmd)
        self._command_pub.publish(cmd_msg)
