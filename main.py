import depthai as dai
import cv2
import numpy as np
import typing
import socket

from depthai import MessageQueue

"""
The part of the pipeline that is the same for the left and right camera
The stereo node is shared, so it must be created before and linked
"""


class MonoPipeline:
    def __init__(
        self, pipeline: dai.Pipeline, stereo_node: dai.node.StereoDepth, is_left: bool
    ) -> None:
        self.pipeline = pipeline
        self.stereo_node = stereo_node
        self.is_left = is_left
        self.socket = (
            dai.CameraBoardSocket.CAM_B if self.is_left else dai.CameraBoardSocket.CAM_C
        )
        self.cam = self.pipeline.create(dai.node.Camera).build(self.socket)

        # Request mono output
        self.mono_out = self.cam.requestOutput((1280, 720), type=dai.ImgFrame.Type.GRAY8)

        if self.is_left:
            self.mono_out.link(self.stereo_node.left)
            self.rect_out = self.stereo_node.rectifiedLeft
        else:
            self.mono_out.link(self.stereo_node.right)
            self.rect_out = self.stereo_node.rectifiedRight

        self.img_q = self.rect_out.createOutputQueue(maxSize=1, blocking=False)
        self.control_in = self.cam.inputControl
        self.ctrl_q = self.control_in.createInputQueue()

    def set_exposure(self, exp_time: int, sens_iso: int) -> None:
        msg = dai.CameraControl()
        msg.setManualExposure(exp_time, sens_iso)
        self.ctrl_q.send(msg)

    def get_projection_matrix(self) -> np.ndarray:
        device = self.pipeline.getDefaultDevice()
        calib_data = device.readCalibration()
        intrinsics = calib_data.getCameraIntrinsics(self.socket, 1280, 720)
        P = np.zeros((3, 4))
        P[:3, :3] = intrinsics
        if not self.is_left:
            return P
        extrinsics = np.array(
            calib_data.getCameraExtrinsics(
                dai.CameraBoardSocket.CAM_B, dai.CameraBoardSocket.CAM_C
            )
        )
        return P @ extrinsics

    def try_get_img(self) -> typing.Tuple[bool, cv2.typing.MatLike | None]:
        if self.img_q.has():
            img = self.img_q.get()
            if isinstance(img, dai.ImgFrame):
                return True, img.getCvFrame()
        return False, None


def try_get_centroid(
    img: cv2.typing.MatLike, threshold01: float, show_preview: bool, preview_name: str
) -> typing.Tuple[bool, float, float]:
    ret, thresh = cv2.threshold(img, 254 * threshold01, 255, 0)
    M = cv2.moments(thresh)
    if M["m00"] == 0.0:
        return False, -1, -1
    cX = M["m10"] / M["m00"]
    cY = M["m01"] / M["m00"]
    if show_preview:
        circle_pos = (int(cX), int(cY))
        cv2.circle(img, circle_pos, 5, (255, 0, 0), -1)
        text_pos = (int(cX), int(cY) - 25)
        cv2.putText(
            img,
            "centroid",
            text_pos,
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 255, 255),
            2,
        )
        cv2.imshow(preview_name, img)
        cv2.imshow(preview_name + " thresh", thresh)
    return True, cX, cY


def DLT(
    proj_l: np.ndarray,
    proj_r: np.ndarray,
    point_l: typing.Tuple[float, float],
    point_r: typing.Tuple[float, float],
) -> np.ndarray:
    # cv.triangulatePoints operates on lists of points
    points_l = np.array(point_l)
    points_r = np.array(point_r)
    points4d: np.ndarray = cv2.triangulatePoints(proj_r, proj_l, points_r, points_l)
    first = points4d[:, 0]
    first = first[:3] / first[3]  # homogenous -> cartesian
    return first

with dai.Pipeline() as pipeline:
    stereo_depth = pipeline.create(dai.node.StereoDepth)
    pipeline_l = MonoPipeline(pipeline, stereo_depth, is_left=True)
    pipeline_r = MonoPipeline(pipeline, stereo_depth, is_left=False)

    IP = "127.0.0.1"
    PORT = 4241
    sock: socket.socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

    pipeline.start()
    # Super low exposure so only LED is visible
    pipeline_r.set_exposure(200, 100)
    pipeline_l.set_exposure(200, 100)
    p_l = pipeline_l.get_projection_matrix()
    p_r = pipeline_r.get_projection_matrix()
    threshold01 = 0.5

    with pipeline:
        while pipeline.isRunning():
            success_l, img_l = pipeline_l.try_get_img()
            success_r, img_r = pipeline_r.try_get_img()
            if success_l and success_r and img_l is not None and img_r is not None:
                success_l, cX_l, cY_l = try_get_centroid(
                    img_l, threshold01=threshold01, show_preview=True, preview_name="left"
                )
                success_r, cX_r, cY_r = try_get_centroid(
                    img_r, threshold01=threshold01, show_preview=True, preview_name="right"
                )
                if success_l and success_r:
                    tracked_pos = DLT(p_r, p_l, (cX_r, cY_r), (cX_l, cY_r))
                    print(tracked_pos)
                    tracked_pos_with_empty_rotation = np.zeros(6)
                    tracked_pos_with_empty_rotation[:3] = tracked_pos
                    tracked_pos_bytes = tracked_pos_with_empty_rotation.tobytes()
                    sock.sendto(tracked_pos_bytes, (IP, PORT))
            key = cv2.waitKey(1)
            if key == ord("+"):
                threshold01 = min(threshold01 + 0.05, 0.99)
                print(threshold01)
            elif key == ord("-"):
                threshold01 = max(threshold01 - 0.05, 0.0)
                print(threshold01)
            elif key == ord("q"):
                break
