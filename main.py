import depthai as dai
import cv2
import numpy as np
import typing
import socket

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
        self.cam = self.pipeline.create(dai.node.MonoCamera)
        self.socket = (
            dai.CameraBoardSocket.CAM_B if self.is_left else dai.CameraBoardSocket.CAM_C
        )
        self.cam.setBoardSocket(self.socket)
        self.img_out = self.pipeline.create(dai.node.XLinkOut)
        self.img_out.setStreamName("img_l" if self.is_left else "img_r")
        if self.is_left:
            self.cam.out.link(self.stereo_node.left)
            self.stereo_node.rectifiedLeft.link(self.img_out.input)
        else:
            self.cam.out.link(self.stereo_node.right)
            self.stereo_node.rectifiedRight.link(self.img_out.input)
        self.control_in = pipeline.create(dai.node.XLinkIn)
        self.control_in.out.link(self.cam.inputControl)
        self.control_in.setStreamName("ctrl_l" if self.is_left else "ctrl_r")

    def set_exposure(self, device: dai.Device, exp_time: int, sens_iso: int) -> None:
        msg = dai.CameraControl()
        msg.setManualExposure(exp_time, sens_iso)
        name = self.control_in.getStreamName()
        ctrl_q = device.getInputQueue(name)
        ctrl_q.send(msg)

    def get_projection_matrix(self, device: dai.Device) -> np.ndarray:
        calibData = device.readCalibration()
        intrinsics = calibData.getCameraIntrinsics(self.socket, 1280, 720)
        P = np.zeros((3, 4))
        P[:3, :3] = intrinsics
        if not self.is_left:
            return P
        extrinsics = np.array(
            calibData.getCameraExtrinsics(
                dai.CameraBoardSocket.CAM_B, dai.CameraBoardSocket.CAM_C
            )
        )
        return P @ extrinsics

    def get_img_out_name(self) -> str:
        return self.img_out.getStreamName()


def convert_to_cv_frame(data: dai.ADatatype) -> cv2.typing.MatLike:
    img_frame = typing.cast(dai.ImgFrame, data)
    obj = img_frame.getCvFrame()
    mat_like = typing.cast(cv2.typing.MatLike, obj)
    return mat_like


def try_get_img(img_q: dai.DataOutputQueue) -> typing.Tuple[bool, cv2.typing.MatLike | None]:
    if img_q.has():
        return True, convert_to_cv_frame(img_q.get())
    return False, None


def try_get_centroid(
    img: cv2.typing.MatLike, threshold01: float, show_preview: bool, preview_name: str
) -> typing.Tuple[bool, int, int]:
    ret, thresh = cv2.threshold(img, 254 * threshold01, 255, 0)
    M = cv2.moments(thresh)
    if M["m00"] == 0.0:
        return False, -1, -1
    cX = int(M["m10"] / M["m00"])
    cY = int(M["m01"] / M["m00"])
    if show_preview:
        cv2.circle(img, (cX, cY), 5, (255, 0, 0), -1)
        cv2.putText(
            img,
            "centroid",
            (cX - 25, cY - 25),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 255, 255),
            2,
        )
        cv2.imshow(preview_name, img)
        cv2.imshow(preview_name + " thresh", thresh)
    return True, cX, cY


def DLT(proj_l, proj_r, point_l, point_r) -> np.ndarray:
    # cv.triangulatePoints operates on lists of points
    points_l = np.zeros((2, 1))
    points_l[:, 0] = point_l
    points_r = np.zeros((2, 1))
    points_r[:, 0] = point_r
    points4d: np.ndarray = cv2.triangulatePoints(proj_r, proj_l, points_r, points_l)
    first = points4d[:, 0]
    first = first[:3] / first[3]  # homogenous -> cartesian
    return first


pipeline = dai.Pipeline()
stereo_depth = pipeline.create(dai.node.StereoDepth)
pipeline_l = MonoPipeline(pipeline, stereo_depth, is_left=True)
pipeline_r = MonoPipeline(pipeline, stereo_depth, is_left=False)

IP = "127.0.0.1"
PORT = 4241
sock: socket.socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

with dai.Device(pipeline) as device:
    device = typing.cast(dai.Device, device)
    # Super low exposure so only LED is visible
    pipeline_r.set_exposure(device, 200, 100)
    pipeline_l.set_exposure(device, 200, 100)
    p_l = pipeline_l.get_projection_matrix(device)
    p_r = pipeline_r.get_projection_matrix(device)
    img_q_l = device.getOutputQueue(pipeline_l.get_img_out_name(), 1, False)
    img_q_r = device.getOutputQueue(pipeline_r.get_img_out_name(), 1, False)
    threshold01 = 0.5

    while True:
        success_l, img_l = try_get_img(img_q_l)
        success_r, img_r = try_get_img(img_q_r)
        if success_l and success_r and img_l is not None and img_r is not None:
            success_l, cX_l, cY_l = try_get_centroid(
                img_l, threshold01=threshold01, show_preview=True, preview_name="left"
            )
            success_r, cX_r, cY_r = try_get_centroid(
                img_r, threshold01=threshold01, show_preview=True, preview_name="right"
            )
        if success_l and success_r:
            tracked_pos = DLT(p_r, p_l, [cX_r, cY_r], [cX_l, cY_r])
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
