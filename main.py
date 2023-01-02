import depthai as dai
import cv2
import numpy as np
import typing

"""
The part of the pipeline that is the same for the left and right camera
"""
class MonoPipeline():
    def __init__(self, pipeline: dai.Pipeline, is_left: bool) -> None:
        self.pipeline = pipeline
        self.is_left = is_left
        self.cam = self.pipeline.create(dai.node.MonoCamera)
        self.socket = dai.CameraBoardSocket.LEFT if self.is_left else dai.CameraBoardSocket.RIGHT
        self.cam.setBoardSocket(self.socket)
        self.img_out = self.pipeline.create(dai.node.XLinkOut)
        self.img_out.setStreamName("img_l" if self.is_left else "img_r")
        self.cam.out.link(self.img_out.input)
        self.control_in = pipeline.create(dai.node.XLinkIn)
        self.control_in.out.link(self.cam.inputControl)
        self.control_in.setStreamName("ctrl_l" if self.is_left else "ctrl_r")
        print
    
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
            cv2.imshow("left" if self.is_left else "right", img)
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
    points_l = np.zeros((2, 1))
    points_l[:, 0] = point_l
    points_r = np.zeros((2, 1))
    points_r[:, 0] = point_r
    points4d : np.ndarray = cv2.triangulatePoints(proj_r, proj_l, points_r, points_l)
    first = points4d[:, 0]
    first = first[:3] / first[3]
    print(first)


pipeline = dai.Pipeline()
pipeline_l = MonoPipeline(pipeline, is_left=True)
pipeline_r = MonoPipeline(pipeline, is_left=False)

with dai.Device(pipeline) as device:
    device: dai.Device
    # Super low exposure so only LED is visible
    pipeline_r.set_exposure(device, 1, 100)
    pipeline_l.set_exposure(device, 1, 100)
    p_l = pipeline_l.get_projection_matrix(device)
    p_r = pipeline_r.get_projection_matrix(device)

    while True:
        success_l, cX_l, cY_l = pipeline_l.try_get_centroid(device, True)
        success_r, cX_r, cY_r = pipeline_r.try_get_centroid(device, True)
        if success_l and success_r:
            DLT(p_r, p_l, [cX_r, cY_r], [cX_l, cY_r])
        if cv2.waitKey(1) == ord('q'):
            break