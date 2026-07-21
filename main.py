import sys
import json
import socket
import numpy as np
import cv2
import depthai as dai
import typing
import time
from pathlib import Path

from PySide6 import QtCore, QtWidgets, QtGui
import pyqtgraph as pg
import pyqtgraph.opengl as gl

CONFIG_FILE = "config.json"

def load_config():
    if Path(CONFIG_FILE).exists():
        try:
            with open(CONFIG_FILE, "r") as f:
                return json.load(f)
        except Exception:
            pass
    return {"exposure": 200, "iso": 100, "threshold": 0.9}

def save_config(config):
    with open(CONFIG_FILE, "w") as f:
        json.dump(config, f)

class MonoPipeline:
    def __init__(
            self, pipeline: dai.Pipeline, is_left: bool
    ) -> None:
        self.pipeline = pipeline
        self.is_left = is_left
        self.socket = (
            dai.CameraBoardSocket.CAM_B if self.is_left else dai.CameraBoardSocket.CAM_C
        )
        self.cam = self.pipeline.create(dai.node.Camera).build(self.socket)

        # Request mono output
        self.mono_out = self.cam.requestOutput(self.resolution)
        self.img_q = self.mono_out.createOutputQueue(maxSize=1, blocking=False)
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
        img: cv2.typing.MatLike, threshold01: float
) -> typing.Tuple[bool, float, float]:
    ret, thresh = cv2.threshold(img, 254 * threshold01, 255, 0)
    M = cv2.moments(thresh)
    if M["m00"] == 0.0:
        return False, -1, -1
    cX = M["m10"] / M["m00"]
    cY = M["m01"] / M["m00"]
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

class Worker(QtCore.QThread):
    frame_ready = QtCore.Signal(np.ndarray)
    centroid_ready = QtCore.Signal(float, float, bool)
    position_ready = QtCore.Signal(np.ndarray)

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.running = True
        self.view_left = True
        self.show_thresholded = False
        self.exposure = int(config.get('exposure', 200))
        self.iso = int(config.get('iso', 100))
        self.threshold = float(config.get('threshold', 0.9))
        self.settings_changed = True

    def run(self):
        with dai.Pipeline() as pipeline:
            pipeline_l = MonoPipeline(pipeline, is_left=True)
            pipeline_r = MonoPipeline(pipeline, is_left=False)

            IP = "127.0.0.1"
            PORT = 4241
            sock: socket.socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

            pipeline.start()
            
            p_l = pipeline_l.get_projection_matrix()
            p_r = pipeline_r.get_projection_matrix()
            while self.running and pipeline.isRunning():
                if self.settings_changed:
                    pipeline_l.set_exposure(self.exposure, self.iso)
                    pipeline_r.set_exposure(self.exposure, self.iso)
                    self.settings_changed = False

                success_l, img_l = pipeline_l.try_get_img()
                success_r, img_r = pipeline_r.try_get_img()

                if success_l and success_r and img_l is not None and img_r is not None:
                    s_l, cX_l, cY_l = try_get_centroid(img_l, self.threshold)
                    s_r, cX_r, cY_r = try_get_centroid(img_r, self.threshold)

                    img_to_show = img_l if self.view_left else img_r
                    found_to_show = s_l if self.view_left else s_r
                    cX_to_show = cX_l if self.view_left else cX_r
                    cY_to_show = cY_l if self.view_left else cY_r

                    if self.show_thresholded:
                        _, img_to_show = cv2.threshold(img_to_show, int(254 * self.threshold), 255, 0)

                    self.frame_ready.emit(img_to_show.copy())
                    img_height = np.shape(img_to_show)[0]
                    cY_to_show -= img_height
                    cY_to_show *= -1
                    self.centroid_ready.emit(cX_to_show, cY_to_show, found_to_show)

                    if s_l and s_r:
                        tracked_pos = DLT(p_r, p_l, (cX_r, cY_r), (cX_l, cY_r))
                        self.position_ready.emit(tracked_pos)

                        tracked_pos_with_empty_rotation = np.zeros(6)
                        tracked_pos_with_empty_rotation[:3] = tracked_pos
                        sock.sendto(tracked_pos_with_empty_rotation.tobytes(), (IP, PORT))

                time.sleep(0.001)

    def stop(self):
        self.running = False
        self.wait()

class MainWindow(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("DepthAI Point Tracker")
        self.resize(1200, 900)
        
        self.config = load_config()
        
        central_widget = QtWidgets.QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QtWidgets.QHBoxLayout(central_widget)
        
        # Left Panel: Controls
        controls_panel = QtWidgets.QWidget()
        controls_panel.setFixedWidth(300)
        controls_layout = QtWidgets.QVBoxLayout(controls_panel)
        main_layout.addWidget(controls_panel)
        
        # Camera Settings
        controls_layout.addWidget(QtWidgets.QLabel("<b>Camera Settings</b>"))
        
        # Exposure
        controls_layout.addWidget(QtWidgets.QLabel("Exposure (\u03bcs):"))
        self.exp_slider = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self.exp_slider.setRange(1, 33000)
        self.exp_slider.setValue(self.config['exposure'])
        self.exp_spin = QtWidgets.QSpinBox()
        self.exp_spin.setRange(1, 33000)
        self.exp_spin.setValue(self.config['exposure'])
        exp_h = QtWidgets.QHBoxLayout()
        exp_h.addWidget(self.exp_slider)
        exp_h.addWidget(self.exp_spin)
        controls_layout.addLayout(exp_h)
        
        # ISO
        controls_layout.addWidget(QtWidgets.QLabel("ISO:"))
        self.iso_slider = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self.iso_slider.setRange(100, 1600)
        self.iso_slider.setValue(self.config['iso'])
        self.iso_spin = QtWidgets.QSpinBox()
        self.iso_spin.setRange(100, 1600)
        self.iso_spin.setValue(self.config['iso'])
        iso_h = QtWidgets.QHBoxLayout()
        iso_h.addWidget(self.iso_slider)
        iso_h.addWidget(self.iso_spin)
        controls_layout.addLayout(iso_h)
        
        # Threshold
        controls_layout.addWidget(QtWidgets.QLabel("Threshold (0.0 - 1.0):"))
        self.thresh_slider = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self.thresh_slider.setRange(0, 100)
        self.thresh_slider.setValue(int(self.config['threshold'] * 100))
        self.thresh_spin = QtWidgets.QDoubleSpinBox()
        self.thresh_spin.setRange(0.0, 1.0)
        self.thresh_spin.setSingleStep(0.01)
        self.thresh_spin.setValue(self.config['threshold'])
        thresh_h = QtWidgets.QHBoxLayout()
        thresh_h.addWidget(self.thresh_slider)
        thresh_h.addWidget(self.thresh_spin)
        controls_layout.addLayout(thresh_h)
        
        # View Options
        controls_layout.addSpacing(20)
        controls_layout.addWidget(QtWidgets.QLabel("<b>View Options</b>"))
        
        # Camera selection group
        self.camera_group = QtWidgets.QButtonGroup(self)
        self.left_cam_radio = QtWidgets.QRadioButton("Left Camera")
        self.right_cam_radio = QtWidgets.QRadioButton("Right Camera")
        self.camera_group.addButton(self.left_cam_radio)
        self.camera_group.addButton(self.right_cam_radio)
        self.left_cam_radio.setChecked(True)
        controls_layout.addWidget(self.left_cam_radio)
        controls_layout.addWidget(self.right_cam_radio)
        
        controls_layout.addSpacing(10)
        
        # View mode group
        self.view_mode_group = QtWidgets.QButtonGroup(self)
        self.unchanged_radio = QtWidgets.QRadioButton("Unchanged")
        self.thresholded_radio = QtWidgets.QRadioButton("Thresholded")
        self.view_mode_group.addButton(self.unchanged_radio)
        self.view_mode_group.addButton(self.thresholded_radio)
        self.unchanged_radio.setChecked(True)
        controls_layout.addWidget(self.unchanged_radio)
        controls_layout.addWidget(self.thresholded_radio)
        
        # Info
        controls_layout.addStretch()
        controls_layout.addWidget(QtWidgets.QLabel("<b>Tracking Info</b>"))
        self.centroid_label = QtWidgets.QLabel("Centroid: N/A")
        controls_layout.addWidget(self.centroid_label)
        self.pos_label = QtWidgets.QLabel("XYZ: N/A")
        controls_layout.addWidget(self.pos_label)
        
        # Right Panel: Visuals
        visuals_layout = QtWidgets.QVBoxLayout()
        main_layout.addLayout(visuals_layout)
        top_visuals = QtWidgets.QHBoxLayout()

        # Top Visuals: Image Preview and 3D
        visuals_layout.addLayout(top_visuals, stretch=1)

        # Image Preview
        self.image_view = pg.GraphicsLayoutWidget()
        self.image_view.setBackground('gray')
        self.image_view.setMinimumSize(600, 400)
        self.vb = self.image_view.addViewBox()
        self.vb.setAspectLocked(True)
        self.img_item = pg.ImageItem()
        self.vb.addItem(self.img_item)
        self.crosshair_v = pg.InfiniteLine(angle=90, movable=False, pen='r')
        self.crosshair_h = pg.InfiniteLine(angle=0, movable=False, pen='r')
        self.vb.addItem(self.crosshair_v)
        self.vb.addItem(self.crosshair_h)
        self.crosshair_v.hide()
        self.crosshair_h.hide()
        top_visuals.addWidget(self.image_view, stretch=5)

        # 3D View
        self.gl_view = gl.GLViewWidget()
        self.gl_view.setMinimumSize(400, 300)
        grid = gl.GLGridItem(size=QtGui.QVector3D(100,100,1))
        grid.setSpacing(10, 10, 10,)
        self.gl_view.addItem(grid)
        self.pos_marker = gl.GLScatterPlotItem(pos=np.array([[0,0,0]]), color=(1,0,0,1), size=10)
        self.gl_view.addItem(self.pos_marker)
        top_visuals.addWidget(self.gl_view, stretch=3)

        # Plot
        self.plot_widget = pg.PlotWidget(title="XYZ over Time")
        self.plot_widget.setMinimumSize(400, 300)
        self.plot_widget.addLegend()
        self.curve_x = self.plot_widget.plot(pen='r', name='X')
        self.curve_y = self.plot_widget.plot(pen='g', name='Y')
        self.curve_z = self.plot_widget.plot(pen='b', name='Z')
        self.data_x = []
        self.data_y = []
        self.data_z = []
        self.max_points = 100
        visuals_layout.addWidget(self.plot_widget)
        
        # Connect signals
        self.exp_slider.valueChanged.connect(self.exp_spin.setValue)
        self.exp_spin.valueChanged.connect(self.update_exposure)
        self.iso_slider.valueChanged.connect(self.iso_spin.setValue)
        self.iso_spin.valueChanged.connect(self.update_iso)
        self.thresh_slider.valueChanged.connect(lambda v: self.thresh_spin.setValue(v / 100.0))
        self.thresh_spin.valueChanged.connect(self.update_threshold)
        
        self.left_cam_radio.toggled.connect(self.update_view_selection)
        self.right_cam_radio.toggled.connect(self.update_view_selection)
        self.unchanged_radio.toggled.connect(self.update_view_selection)
        self.thresholded_radio.toggled.connect(self.update_view_selection)
        
        # Worker thread
        self.worker = Worker(self.config)
        self.worker.frame_ready.connect(self.on_frame)
        self.worker.centroid_ready.connect(self.on_centroid)
        self.worker.position_ready.connect(self.on_position)
        self.worker.start()

    def update_exposure(self, val):
        self.worker.exposure = val
        self.worker.settings_changed = True
        self.exp_slider.blockSignals(True)
        self.exp_slider.setValue(val)
        self.exp_slider.blockSignals(False)

    def update_iso(self, val):
        self.worker.iso = val
        self.worker.settings_changed = True
        self.iso_slider.blockSignals(True)
        self.iso_slider.setValue(val)
        self.iso_slider.blockSignals(False)

    def update_threshold(self, val):
        self.worker.threshold = val
        self.worker.settings_changed = True
        self.thresh_slider.blockSignals(True)
        self.thresh_slider.setValue(int(val * 100))
        self.thresh_slider.blockSignals(False)

    def update_view_selection(self):
        self.worker.view_left = self.left_cam_radio.isChecked()
        self.worker.show_thresholded = self.thresholded_radio.isChecked()

    @QtCore.Slot(np.ndarray)
    def on_frame(self, frame: cv2.typing.MatLike):
        rotated_frame = cv2.rotate(frame, cv2.ROTATE_180)
        mirrored = cv2.flip(rotated_frame, 1)
        self.img_item.setImage(mirrored.T)

    @QtCore.Slot(float, float, bool)
    def on_centroid(self, x, y, found):
        if found:
            self.crosshair_v.setPos(x)
            self.crosshair_h.setPos(y)
            self.crosshair_v.show()
            self.crosshair_h.show()
            self.centroid_label.setText(f"Centroid: {x:.1f}, {y:.1f}")
        else:
            self.crosshair_v.hide()
            self.crosshair_h.hide()
            self.centroid_label.setText("Centroid: N/A")

    @QtCore.Slot(np.ndarray)
    def on_position(self, pos):
        self.pos_label.setText(f"XYZ: {pos[0]:.2f}, {pos[1]:.2f}, {pos[2]:.2f}")
        self.pos_marker.setData(pos=np.array([[pos[0], pos[2], pos[1]]]))
        self.data_x.append(pos[0])
        self.data_y.append(pos[1])
        self.data_z.append(pos[2])
        if len(self.data_x) > self.max_points:
            self.data_x.pop(0)
            self.data_y.pop(0)
            self.data_z.pop(0)
        self.curve_x.setData(self.data_x)
        self.curve_y.setData(self.data_y)
        self.curve_z.setData(self.data_z)

    def closeEvent(self, event):
        self.worker.stop()
        save_config({
            'exposure': self.worker.exposure,
            'iso': self.worker.iso,
            'threshold': self.worker.threshold
        })
        event.accept()

if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())
