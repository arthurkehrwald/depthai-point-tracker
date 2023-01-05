import depthai as dai
import cv2
import numpy as np
import typing
import socket

"""
The part of the pipeline that is the same for the left and right camera
The stereo node is shared, so it must be created before and linked
"""
class MonoPipeline():
    def __init__(self, pipeline: dai.Pipeline, stereo_node: dai.node.StereoDepth, is_left: bool) -> None:
        self.pipeline = pipeline
        self.stereo_node = stereo_node
        self.is_left = is_left
        self.cam = self.pipeline.create(dai.node.MonoCamera)
        self.socket = dai.CameraBoardSocket.LEFT if self.is_left else dai.CameraBoardSocket.RIGHT
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
    
    def set_exposure(self, device: dai.Device, exp_time: int, sens_iso: int) ->None:
        msg = dai.CameraControl()
        msg.setManualExposure(exp_time, sens_iso)
        name = self.control_in.getStreamName()
        ctrl_q = device.getInputQueue(name)
        ctrl_q.send(msg)

    def try_get_centroid(self, device: dai.Device, show_preview: bool) -> typing.Tuple[bool, int, int]:
        q = device.getOutputQueue(self.img_out.getStreamName())
        output = q.tryGet()
        if output is None:
            return False, -1, -1
        img = output.getCvFrame()
        res = img.shape
        ret, thresh = cv2.threshold(img, 127, 255, 0)
        M = cv2.moments(thresh)
        if M["m00"] == 0.0:
            return False, -1, -1        
        cX = int(M["m10"] / M["m00"])
        cY = int(M["m01"] / M["m00"])
        if show_preview:
            cv2.circle(img, (cX, cY), 5, (255, 0, 0), -1)
            cv2.putText(img, "centroid", (cX - 25, cY - 25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)        
            cv2.imshow("left" if self.is_left else "right", img)
        return True, cX, cY

    def get_projection_matrix(self, device: dai.Device) -> np.ndarray:
        calibData = device.readCalibration()
        socket = dai.CameraBoardSocket.LEFT if self.is_left else dai.CameraBoardSocket.RIGHT
        intrinsics = calibData.getCameraIntrinsics(socket, 1280, 720)
        P = np.zeros((3,4))
        P[:3, :3] = intrinsics
        if not self.is_left:
            return P
        extrinsics = np.array(calibData.getCameraExtrinsics(dai.CameraBoardSocket.LEFT, dai.CameraBoardSocket.RIGHT))
        return P @ extrinsics

def DLT(proj_l, proj_r, point_l, point_r) -> np.array:
    # cv.triangulatePoints operates on lists of points
    points_l = np.zeros((2, 1))
    points_l[:, 0] = point_l
    points_r = np.zeros((2, 1))
    points_r[:, 0] = point_r
    points4d : np.ndarray = cv2.triangulatePoints(proj_r, proj_l, points_r, points_l)
    first = points4d[:, 0]
    first = first[:3] / first[3]
    return first

pipeline = dai.Pipeline()
stereo_depth = pipeline.create(dai.node.StereoDepth)
pipeline_l = MonoPipeline(pipeline, stereo_depth, is_left=True)
pipeline_r = MonoPipeline(pipeline, stereo_depth, is_left=False)

IP = "127.0.0.1"
PORT = 4241
sock: socket.socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

with dai.Device(pipeline) as device:
    device: dai.Device
    # Super low exposure so only LED is visible
    pipeline_r.set_exposure(device, 500, 300)
    pipeline_l.set_exposure(device, 500, 300)
    p_l = pipeline_l.get_projection_matrix(device)
    p_r = pipeline_r.get_projection_matrix(device)

    while True:
        success_l, cX_l, cY_l = pipeline_l.try_get_centroid(device, False)
        success_r, cX_r, cY_r = pipeline_r.try_get_centroid(device, False)
        if success_l and success_r:
            tracked_pos : np.array = DLT(p_r, p_l, [cX_r, cY_r], [cX_l, cY_r])
            tracked_pos_with_empty_rotation : np.array = np.zeros(6)
            tracked_pos_with_empty_rotation[:3] = tracked_pos
            tracked_pos_bytes = tracked_pos_with_empty_rotation.tobytes()
            sock.sendto(tracked_pos_bytes, (IP, PORT))
        if cv2.waitKey(1) == ord('q'):
            break