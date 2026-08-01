import typing
from dataclasses import dataclass
from types import TracebackType
from enum import StrEnum

import cv2
import depthai as dai
import numpy as np

from config import Config


class CameraConfigKeys(StrEnum):
    resolution_x = "resolution_x"
    resolution_y = "resolution_y"
    fps = "fps"
    exposure = "exposure"
    iso = "iso"


DEFAULT_CONFIG = {
    CameraConfigKeys.resolution_x: 1280,
    CameraConfigKeys.resolution_y: 720,
    CameraConfigKeys.fps: 75,
    CameraConfigKeys.exposure: 150,
    CameraConfigKeys.iso: 200
}


@dataclass(frozen=True)
class CameraSocketParams:
    projection: np.ndarray
    intrinsics: np.ndarray
    distortion: np.ndarray
    rotation: np.ndarray


@dataclass(frozen=True)
class StereoFrame:
    left_frame: np.ndarray
    right_frame: np.ndarray
    frame_time_ms: float
    left_time_of_capture: float
    right_time_of_capture: float
    time_of_arrival: float


class StereoCamera:
    def __init__(
            self, config: Config
    ):
        self.cam_params_r = None
        self.cam_params_l = None
        self.x_out_q = None
        self.cam_r_ctrl_q = None
        self.cam_l_ctrl_q = None
        self.sync = None
        self.pipeline = None
        self.config = config
        res_x = config.get("resolution_x")
        res_y = config.get("resolution_y")
        assert isinstance(res_x, int) and isinstance(res_y, int)
        self.resolution = (res_x, res_y)
        self.fps = config.get("fps")
        self.prev_frame_arrival_time = -1.0

    def __enter__(self) -> "StereoCamera":
        self.start()
        return self

    def start(self):
        self.config.add_callback(self.on_config_val_changed)
        self.pipeline = dai.Pipeline()
        self.pipeline.setXLinkChunkSize(0)
        self.sync = self.pipeline.create(dai.node.Sync)
        assert isinstance(self.sync, dai.node.Sync)
        self.cam_l_ctrl_q = self.add_cam(dai.CameraBoardSocket.CAM_B, "left")
        self.cam_r_ctrl_q = self.add_cam(dai.CameraBoardSocket.CAM_C, "right")
        self.x_out_q = self.sync.out.createOutputQueue()

        self.pipeline.start()
        self.set_exposure(int(self.config.get(CameraConfigKeys.exposure)),
                          int(self.config.get(CameraConfigKeys.iso)))
        calibration = self.pipeline.getCalibrationData()
        self.cam_params_l, self.cam_params_r = self.compute_stereo_rectification(calibration)

    def __exit__(self, exc_type: type[BaseException] | None, exc_val: BaseException | None,
                 exc_tb: TracebackType | None):
        self.stop()

    def stop(self):
        self.config.remove_callback(self.on_config_val_changed)
        self.pipeline.stop()

    def on_config_val_changed(self, changed: typing.List[str]):
        if any(key in [CameraConfigKeys.exposure, CameraConfigKeys.iso] for key in changed):
            self.set_exposure(int(self.config.get(CameraConfigKeys.exposure)),
                              int(self.config.get(CameraConfigKeys.iso)))

    def add_cam(self, socket: dai.CameraBoardSocket, name: str) -> dai.InputQueue:
        cam = self.pipeline.create(dai.node.Camera).build(socket)
        cam_out = cam.requestOutput(self.resolution, fps=self.fps)
        cam_out.link(self.sync.inputs[name])
        self.sync.inputs[name].setBlocking(False)
        self.sync.inputs[name].setMaxSize(1)
        return cam.inputControl.createInputQueue(1, False)

    def compute_stereo_rectification(self, calibration: dai.CalibrationHandler) -> typing.Tuple[
        CameraSocketParams, CameraSocketParams]:
        intrinsics_l = np.array(
            calibration.getCameraIntrinsics(
                dai.CameraBoardSocket.CAM_B, self.resolution[0], self.resolution[1]
            ),
        )
        intrinsics_r = np.array(
            calibration.getCameraIntrinsics(
                dai.CameraBoardSocket.CAM_C, self.resolution[0], self.resolution[1]
            ),
        )

        distortion_l = np.array(
            calibration.getDistortionCoefficients(dai.CameraBoardSocket.CAM_B),
        )
        distortion_r = np.array(
            calibration.getDistortionCoefficients(dai.CameraBoardSocket.CAM_C),
        )

        l_to_r_transformation = np.array(
            calibration.getCameraExtrinsics(
                dai.CameraBoardSocket.CAM_B, dai.CameraBoardSocket.CAM_C
            )
        )
        l_to_r_rotation = l_to_r_transformation[:3, :3]
        l_to_r_translation = l_to_r_transformation[:3, 3:4]

        rotation_l, rotation_r, projection_l, projection_r, _, _, _ = cv2.stereoRectify(
            intrinsics_l, distortion_l.flatten(),
            intrinsics_r, distortion_r.flatten(),
            imageSize=self.resolution,
            R=l_to_r_rotation,
            T=l_to_r_translation,
            flags=cv2.CALIB_ZERO_DISPARITY,
            alpha=0
        )

        return (CameraSocketParams(projection_l, intrinsics_l, distortion_l, rotation_l),
                CameraSocketParams(projection_r, intrinsics_r, distortion_r, rotation_r))

    def get_stereo_frame(self) -> StereoFrame:
        message_group = self.x_out_q.get()
        arrival_time = dai.Clock.now().total_seconds()
        assert isinstance(message_group, dai.MessageGroup)
        raw_frame_l = message_group["left"]
        raw_frame_r = message_group["right"]
        assert isinstance(raw_frame_l, dai.ImgFrame) and isinstance(raw_frame_r, dai.ImgFrame)
        frame_time_ms = (arrival_time - self.prev_frame_arrival_time) * 1000
        self.prev_frame_arrival_time = arrival_time
        return StereoFrame(
            left_frame=raw_frame_l.getCvFrame(),
            right_frame=raw_frame_r.getCvFrame(),
            frame_time_ms=frame_time_ms,
            left_time_of_capture=raw_frame_l.getTimestamp().total_seconds(),
            right_time_of_capture=raw_frame_r.getTimestamp().total_seconds(),
            time_of_arrival=arrival_time
        )

    def triangulate(
            self,
            point_l: typing.Tuple[float, float],
            point_r: typing.Tuple[float, float],
    ) -> typing.Tuple[float, float, float]:
        # cv.triangulatePoints operates on 2xN arrays of points
        points_l = np.array(point_l).reshape(2, 1)
        points_r = np.array(point_r).reshape(2, 1)
        points4d: np.ndarray = cv2.triangulatePoints(self.cam_params_l.projection, self.cam_params_r.projection,
                                                     points_l, points_r)
        first = points4d[:, 0]
        first = first[:3] / first[3]  # homogenous -> cartesian
        return first

    def set_exposure(self, exp_time: int, sens_iso: int) -> None:
        msg = dai.CameraControl()
        exp_time = min(max(10, exp_time), 1000)
        msg.setManualExposure(exp_time, sens_iso)
        self.cam_l_ctrl_q.send(msg)
        self.cam_r_ctrl_q.send(msg)

    def get_time(self) -> float:
        return dai.Clock.now().total_seconds()

    def rectify_points(self, points: typing.List[typing.Tuple[float, float]], is_left: bool) -> typing.List[
        typing.Tuple[float, float]]:
        # cv2.undistortPoints expects a 1xNx2 or Nx1x2 array
        pts = np.array(points, dtype=np.float32).reshape(-1, 1, 2)
        params = self.cam_params_l if is_left else self.cam_params_r
        undistorted = cv2.undistortPoints(pts, params.intrinsics, params.distortion,
                                          R=params.rotation, P=params.projection)
        return [(float(pt[0][0]), float(pt[0][1])) for pt in undistorted]
