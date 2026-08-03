import time

from PySide6 import QtCore, QtWidgets, QtGui
import pyqtgraph as pg
pg.setConfigOption('imageAxisOrder', 'row-major')
import pyqtgraph.opengl as gl
import numpy as np

from blob_detector import BlobDetectorConfigKeys
import blob_detector
from camera import CameraConfigKeys
import camera
from config import Config
from main import TrackerConfigKeys, Worker
from recorder import Recorder

DEFAULT_CONFIG = {
    TrackerConfigKeys.stereo_conf_threshold: 0.5
}


class MainWindow(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("DepthAI Point Tracker")
        self.resize(1200, 900)

        self.config = Config(defaults={**camera.DEFAULT_CONFIG, **blob_detector.DEFAULT_CONFIG, **DEFAULT_CONFIG})

        monospace_font = QtGui.QFont("Courier New")
        monospace_font.setStyleHint(QtGui.QFont.StyleHint.Monospace)

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

        # Resolution
        controls_layout.addWidget(QtWidgets.QLabel("Resolution:"))
        self.res_combo = QtWidgets.QComboBox()
        self.res_options = [(1280, 800), (1280, 720), (640, 400)]
        for w, h in self.res_options:
            self.res_combo.addItem(f"{w}x{h}")
        self.cam_resolution = int(self.config.get(CameraConfigKeys.resolution_x)), int(
            self.config.get(CameraConfigKeys.resolution_y))
        if self.cam_resolution in self.res_options:
            self.res_combo.setCurrentIndex(self.res_options.index(self.cam_resolution))
        controls_layout.addWidget(self.res_combo)

        # Framerate
        controls_layout.addWidget(QtWidgets.QLabel("Framerate (FPS):"))
        self.fps_spin = QtWidgets.QSpinBox()
        self.fps_spin.setRange(30, 100)
        self.cam_fps = int(self.config.get(CameraConfigKeys.fps))
        self.fps_spin.setValue(self.cam_fps)
        self.target_fps = self.cam_fps
        controls_layout.addWidget(self.fps_spin)

        # Restart warning
        restart_warning = QtWidgets.QLabel("Resolution and FPS changes require restart")
        restart_warning.setWordWrap(True)
        restart_warning.setStyleSheet("color: orange; font-style: italic;")
        controls_layout.addWidget(restart_warning)

        controls_layout.addSpacing(10)

        # Exposure
        controls_layout.addWidget(QtWidgets.QLabel("Exposure (\u03bcs):"))
        self.exp_slider = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self.exp_slider.setRange(10, 1000)
        exposure = int(self.config.get(CameraConfigKeys.exposure))
        self.exp_slider.setValue(exposure)
        self.exp_spin = QtWidgets.QSpinBox()
        self.exp_spin.setRange(1, 33000)
        self.exp_spin.setValue(exposure)
        exp_h = QtWidgets.QHBoxLayout()
        exp_h.addWidget(self.exp_slider)
        exp_h.addWidget(self.exp_spin)
        controls_layout.addLayout(exp_h)

        # ISO
        controls_layout.addWidget(QtWidgets.QLabel("ISO:"))
        self.iso_slider = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self.iso_slider.setRange(100, 1600)
        iso = int(self.config.get(CameraConfigKeys.iso))
        self.iso_slider.setValue(iso)
        self.iso_spin = QtWidgets.QSpinBox()
        self.iso_spin.setRange(100, 1600)
        self.iso_spin.setValue(iso)
        iso_h = QtWidgets.QHBoxLayout()
        iso_h.addWidget(self.iso_slider)
        iso_h.addWidget(self.iso_spin)
        controls_layout.addLayout(iso_h)

        # Blob Detector Settings
        controls_layout.addSpacing(20)
        controls_layout.addWidget(QtWidgets.QLabel("<b>Blob Detector Settings</b>"))

        # Blob Min/Max Threshold
        controls_layout.addWidget(QtWidgets.QLabel("Blob Threshold (Min/Max):"))
        self.blob_min_thresh_spin = QtWidgets.QSpinBox()
        self.blob_min_thresh_spin.setRange(0, 255)
        blob_min_thresh = int(self.config.get(BlobDetectorConfigKeys.min_threshold))
        self.blob_min_thresh_spin.setValue(blob_min_thresh)
        self.blob_max_thresh_spin = QtWidgets.QSpinBox()
        self.blob_max_thresh_spin.setRange(0, 255)
        blob_max_thresh = int(self.config.get(BlobDetectorConfigKeys.max_threshold))
        self.blob_max_thresh_spin.setValue(blob_max_thresh)
        blob_thresh_h = QtWidgets.QHBoxLayout()
        blob_thresh_h.addWidget(self.blob_min_thresh_spin)
        blob_thresh_h.addWidget(self.blob_max_thresh_spin)
        controls_layout.addLayout(blob_thresh_h)

        # Blob Threshold Step
        controls_layout.addWidget(QtWidgets.QLabel("Blob Threshold Step:"))
        self.blob_thresh_step_spin = QtWidgets.QSpinBox()
        self.blob_thresh_step_spin.setRange(1, 100)
        blob_thresh_step = int(self.config.get(BlobDetectorConfigKeys.threshold_step))
        self.blob_thresh_step_spin.setValue(blob_thresh_step)
        controls_layout.addWidget(self.blob_thresh_step_spin)

        # Blob Min/Max Area
        controls_layout.addWidget(QtWidgets.QLabel("Blob Area (Min/Max):"))
        self.blob_min_area_spin = QtWidgets.QSpinBox()
        self.blob_min_area_spin.setRange(1, 10000)
        blob_min_area = int(self.config.get(BlobDetectorConfigKeys.min_area))
        self.blob_min_area_spin.setValue(blob_min_area)
        self.blob_max_area_spin = QtWidgets.QSpinBox()
        self.blob_max_area_spin.setRange(1, 10000)
        blob_max_area = int(self.config.get(BlobDetectorConfigKeys.max_area))
        self.blob_min_area_spin.setValue(blob_min_area)
        self.blob_max_area_spin.setValue(blob_max_area)
        blob_area_h = QtWidgets.QHBoxLayout()
        blob_area_h.addWidget(self.blob_min_area_spin)
        blob_area_h.addWidget(self.blob_max_area_spin)
        controls_layout.addLayout(blob_area_h)

        # Blob Circularity
        controls_layout.addWidget(QtWidgets.QLabel("Min Circularity:"))
        self.blob_circ_slider = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self.blob_circ_slider.setRange(0, 100)
        blob_min_circularity = float(self.config.get(BlobDetectorConfigKeys.min_circularity))
        self.blob_circ_slider.setValue(int(blob_min_circularity * 100))
        self.blob_circ_spin = QtWidgets.QDoubleSpinBox()
        self.blob_circ_spin.setRange(0.0, 1.0)
        self.blob_circ_spin.setSingleStep(0.05)
        self.blob_circ_spin.setValue(blob_min_circularity)
        blob_circ_h = QtWidgets.QHBoxLayout()
        blob_circ_h.addWidget(self.blob_circ_slider)
        blob_circ_h.addWidget(self.blob_circ_spin)
        controls_layout.addLayout(blob_circ_h)

        # Blob Convexity
        controls_layout.addWidget(QtWidgets.QLabel("Min Convexity:"))
        self.blob_conv_slider = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self.blob_conv_slider.setRange(0, 100)
        blob_min_convexity = float(self.config.get(BlobDetectorConfigKeys.min_convexity))
        self.blob_conv_slider.setValue(int(blob_min_convexity * 100))
        self.blob_conv_spin = QtWidgets.QDoubleSpinBox()
        self.blob_conv_spin.setRange(0.0, 1.0)
        self.blob_conv_spin.setSingleStep(0.05)
        self.blob_conv_spin.setValue(blob_min_convexity)
        blob_conv_h = QtWidgets.QHBoxLayout()
        blob_conv_h.addWidget(self.blob_conv_slider)
        blob_conv_h.addWidget(self.blob_conv_spin)
        controls_layout.addLayout(blob_conv_h)

        # Blob Inertia
        controls_layout.addWidget(QtWidgets.QLabel("Min Inertia Ratio:"))
        self.blob_inert_slider = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self.blob_inert_slider.setRange(0, 100)
        blob_min_inertia = float(self.config.get(BlobDetectorConfigKeys.min_inertia))
        self.blob_inert_slider.setValue(int(blob_min_inertia * 100))
        self.blob_inert_spin = QtWidgets.QDoubleSpinBox()
        self.blob_inert_spin.setRange(0.0, 1.0)
        self.blob_inert_spin.setSingleStep(0.05)
        self.blob_inert_spin.setValue(blob_min_inertia)
        blob_inert_h = QtWidgets.QHBoxLayout()
        blob_inert_h.addWidget(self.blob_inert_slider)
        blob_inert_h.addWidget(self.blob_inert_spin)
        controls_layout.addLayout(blob_inert_h)

        # Stereo Match Confidence Threshold
        controls_layout.addSpacing(20)
        controls_layout.addWidget(QtWidgets.QLabel("<b>Stereo Matching</b>"))
        controls_layout.addWidget(QtWidgets.QLabel("Confidence Threshold:"))
        self.stereo_conf_slider = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self.stereo_conf_slider.setRange(0, 100)
        stereo_conf_thresh = float(self.config.get(TrackerConfigKeys.stereo_conf_threshold))
        self.stereo_conf_slider.setValue(int(stereo_conf_thresh * 100))
        self.stereo_conf_spin = QtWidgets.QDoubleSpinBox()
        self.stereo_conf_spin.setRange(0.0, 1.0)
        self.stereo_conf_spin.setSingleStep(0.05)
        self.stereo_conf_spin.setValue(stereo_conf_thresh)
        stereo_conf_h = QtWidgets.QHBoxLayout()
        stereo_conf_h.addWidget(self.stereo_conf_slider)
        stereo_conf_h.addWidget(self.stereo_conf_spin)
        controls_layout.addLayout(stereo_conf_h)

        controls_layout.addSpacing(20)
        self.record_button = QtWidgets.QPushButton("Start Recording")
        self.record_button.setCheckable(True)
        self.record_button.clicked.connect(self.toggle_recording)
        controls_layout.addWidget(self.record_button)

        # Info
        controls_layout.addStretch()
        controls_layout.addWidget(QtWidgets.QLabel("<b>Tracking Info</b>"))
        self.left_centroid_label = QtWidgets.QLabel("Centroid L: N/A")
        self.left_centroid_label.setFont(monospace_font)
        controls_layout.addWidget(self.left_centroid_label)
        self.right_centroid_label = QtWidgets.QLabel("Centroid R: N/A")
        self.right_centroid_label.setFont(monospace_font)
        controls_layout.addWidget(self.right_centroid_label)
        self.pos_label = QtWidgets.QLabel("XYZ: N/A")
        self.pos_label.setFont(monospace_font)
        controls_layout.addWidget(self.pos_label)
        self.conf_label = QtWidgets.QLabel("Confidence: N/A")
        self.conf_label.setFont(monospace_font)
        controls_layout.addWidget(self.conf_label)

        controls_layout.addSpacing(10)
        controls_layout.addWidget(QtWidgets.QLabel("<b>Timing Info</b>"))
        self.frame_time_label = QtWidgets.QLabel("Frame time: N/A")
        self.frame_time_label.setFont(monospace_font)
        controls_layout.addWidget(self.frame_time_label)
        self.capture_latency_label = QtWidgets.QLabel("Capture latency: N/A")
        self.capture_latency_label.setFont(monospace_font)
        controls_layout.addWidget(self.capture_latency_label)
        self.processing_latency_label = QtWidgets.QLabel("Processing latency: N/A")
        self.processing_latency_label.setFont(monospace_font)
        controls_layout.addWidget(self.processing_latency_label)
        self.total_latency_label = QtWidgets.QLabel("Total latency: N/A")
        self.total_latency_label.setFont(monospace_font)
        controls_layout.addWidget(self.total_latency_label)
        self.lr_diff_label = QtWidgets.QLabel("L-R frame diff: N/A")
        self.lr_diff_label.setFont(monospace_font)
        controls_layout.addWidget(self.lr_diff_label)

        # Right Panel: Visuals
        visuals_layout = QtWidgets.QVBoxLayout()
        main_layout.addLayout(visuals_layout)
        top_visuals = QtWidgets.QHBoxLayout()

        # Top Visuals: Image Preview and 3D
        visuals_layout.addLayout(top_visuals, stretch=1)

        # Image Previews (Vertical)
        camera_layout = QtWidgets.QVBoxLayout()
        top_visuals.addLayout(camera_layout, stretch=5)
        self.last_img_update_time = 0
        img_transform= QtGui.QTransform()
        img_transform.scale(1, -1)

        self.image_view_l = pg.GraphicsLayoutWidget()
        self.image_view_l.setBackground('gray')
        self.image_view_l.setMinimumSize(300, 200)
        self.image_view_l.addLabel("<span style='color: #00FF00; font-weight: bold;'>Left Camera</span>", row=0, col=0)
        self.vb_l = self.image_view_l.addViewBox(row=1, col=0)
        self.vb_l.setAspectLocked(True)
        self.img_item_l = pg.ImageItem(levels=(0, 255))
        self.img_item_l.setTransform(img_transform)
        self.vb_l.addItem(self.img_item_l)

        # Overlay items
        self.overlay_item_l = pg.ImageItem()
        self.overlay_item_l.setTransform(img_transform)
        # Scale by 2 to match full resolution
        self.overlay_item_l.setScale(2.0)
        # Create a green color map: from transparent green to opaque green
        # Actually, if the input is monochrome (0-255), we want to map it to green with that intensity
        # ImageItem.setColorMap takes a pg.ColorMap object
        green_cmap = pg.ColorMap(pos=[0, 1.0], color=[(0, 255, 0, 0), (0, 255, 0, 255)])
        self.overlay_item_l.setColorMap(green_cmap)
        self.vb_l.addItem(self.overlay_item_l)

        self.image_view_r = pg.GraphicsLayoutWidget()
        self.image_view_r.setBackground('gray')
        self.image_view_r.setMinimumSize(300, 200)
        self.image_view_r.addLabel("<span style='color: #00FF00; font-weight: bold;'>Right Camera</span>", row=0, col=0)
        self.vb_r = self.image_view_r.addViewBox(row=1, col=0)
        self.vb_r.setAspectLocked(True)
        self.img_item_r = pg.ImageItem(levels=(0, 255))
        self.img_item_r.setTransform(img_transform)
        self.vb_r.addItem(self.img_item_r)

        self.overlay_item_r = pg.ImageItem()
        self.overlay_item_r.setTransform(img_transform)
        self.overlay_item_r.setScale(2.0)
        self.overlay_item_r.setColorMap(green_cmap)
        self.vb_r.addItem(self.overlay_item_r)

        self.pred_marker_r = pg.ScatterPlotItem(size=10, pen=pg.mkPen(None), brush=pg.mkBrush('g'))
        self.vb_r.addItem(self.pred_marker_r)

        # Pre-fill both ImageItems with a blank frame of identical size so the
        # left and right previews have the same shape/aspect ratio right from
        # startup (before any real frame has been received from the worker),
        # and lock their views together so they always stay in sync.
        blank_frame = np.zeros((self.cam_resolution[1], self.cam_resolution[0]), dtype=np.uint8)
        # Disable pyqtgraph's automatic level (brightness/contrast) scaling so the
        # preview reflects the camera's actual exposure instead of being
        # auto-stretched to the min/max of each frame, which otherwise makes
        # dark frames appear artificially lightened.
        self.img_item_l.setImage(blank_frame, autoLevels=False, levels=(0, 255))
        self.img_item_r.setImage(blank_frame, autoLevels=False, levels=(0, 255))
        blank_overlay = np.zeros((self.cam_resolution[1] // 2, self.cam_resolution[0] // 2), dtype=np.uint8)
        self.overlay_item_l.setImage(blank_overlay, autoLevels=False)
        self.overlay_item_r.setImage(blank_overlay, autoLevels=False)
        self.vb_r.setXLink(self.vb_l)
        self.vb_r.setYLink(self.vb_l)

        camera_layout.addWidget(self.image_view_l)
        camera_layout.addWidget(self.image_view_r)

        # 3D View
        self.gl_view = gl.GLViewWidget()
        self.gl_view.setMinimumSize(400, 300)
        self.gl_view.setCameraPosition(distance=200)
        grid = gl.GLGridItem(size=QtGui.QVector3D(100, 100, 1))
        grid.setSpacing(10, 10, 10, )
        self.gl_view.addItem(grid)
        self.pos_marker = gl.GLScatterPlotItem(pos=np.array([[0, 0, 0]]), color=(1, 0, 0, 1), size=10)
        self.gl_view.addItem(self.pos_marker)
        top_visuals.addWidget(self.gl_view, stretch=6)

        # Plot
        self.plot_widget = pg.PlotWidget(title="XYZ over Time")
        self.plot_widget.setMinimumSize(400, 300)
        self.plot_widget.addLegend()
        self.pos_buffer_size = 100
        # Uncommenting may improve performance
        # self.plot_widget.disableAutoRange()
        # self.plot_widget.setRange(xRange=(0, self.pos_buffer_size), yRange=(-100, 100))
        # Only draw points that are currently visible on the screen
        self.plot_widget.setClipToView(True)
        # Downsample the data dynamically if there are more points than screen pixels
        self.plot_widget.setDownsampling(mode='peak')
        self.pos_data = np.zeros((3, self.pos_buffer_size))
        self.pos_ptr = 0
        self.curve_x = self.plot_widget.plot(pen='r', name='X')
        self.curve_y = self.plot_widget.plot(pen='g', name='Y')
        self.curve_z = self.plot_widget.plot(pen='b', name='Z')
        visuals_layout.addWidget(self.plot_widget)

        # Connect signals
        self.exp_slider.valueChanged.connect(self.exp_spin.setValue)
        self.exp_spin.valueChanged.connect(self.update_exposure)
        self.iso_slider.valueChanged.connect(self.iso_spin.setValue)
        self.iso_spin.valueChanged.connect(self.update_iso)

        self.blob_min_thresh_spin.valueChanged.connect(
            lambda v: self.update_blob_param(BlobDetectorConfigKeys.min_threshold, v))
        self.blob_max_thresh_spin.valueChanged.connect(
            lambda v: self.update_blob_param(BlobDetectorConfigKeys.max_threshold, v))
        self.blob_thresh_step_spin.valueChanged.connect(
            lambda v: self.update_blob_param(BlobDetectorConfigKeys.threshold_step, v))
        self.blob_min_area_spin.valueChanged.connect(
            lambda v: self.update_blob_param(BlobDetectorConfigKeys.min_area, v))
        self.blob_max_area_spin.valueChanged.connect(
            lambda v: self.update_blob_param(BlobDetectorConfigKeys.max_area, v))
        self.blob_circ_slider.valueChanged.connect(lambda v: self.blob_circ_spin.setValue(v / 100.0))
        self.blob_circ_spin.valueChanged.connect(
            lambda v: self.update_blob_param(BlobDetectorConfigKeys.min_circularity, v))
        self.blob_conv_slider.valueChanged.connect(lambda v: self.blob_conv_spin.setValue(v / 100.0))
        self.blob_conv_spin.valueChanged.connect(
            lambda v: self.update_blob_param(BlobDetectorConfigKeys.min_convexity, v))
        self.blob_inert_slider.valueChanged.connect(lambda v: self.blob_inert_spin.setValue(v / 100.0))
        self.blob_inert_spin.valueChanged.connect(
            lambda v: self.update_blob_param(BlobDetectorConfigKeys.min_inertia, v))
        self.stereo_conf_slider.valueChanged.connect(lambda v: self.stereo_conf_spin.setValue(v / 100.0))
        self.stereo_conf_spin.valueChanged.connect(self.update_stereo_conf)

        # Worker thread
        self.worker = Worker(self.config)
        self.worker.frame_ready.connect(self.on_frame)
        self.worker.centroid_ready.connect(self.on_centroid)
        self.worker.candidates_ready.connect(self.on_candidates)
        self.worker.position_ready.connect(self.on_position)
        self.last_pos_plot_time = 0
        self.worker.stats_ready.connect(self.on_stats)
        self.worker.start()

    def update_exposure(self, val):
        self.config.set(CameraConfigKeys.exposure, val)
        self.exp_slider.blockSignals(True)
        self.exp_slider.setValue(val)
        self.exp_slider.blockSignals(False)

    def update_iso(self, val):
        self.config.set(CameraConfigKeys.iso, val)
        self.iso_slider.blockSignals(True)
        self.iso_slider.setValue(val)
        self.iso_slider.blockSignals(False)

    def update_blob_param(self, name: str, val: int | float):
        self.config.set(name, val)
        if name == BlobDetectorConfigKeys.min_circularity:
            self.blob_circ_slider.blockSignals(True)
            self.blob_circ_slider.setValue(int(val * 100))
            self.blob_circ_slider.blockSignals(False)
        elif name == BlobDetectorConfigKeys.min_convexity:
            self.blob_conv_slider.blockSignals(True)
            self.blob_conv_slider.setValue(int(val * 100))
            self.blob_conv_slider.blockSignals(False)
        elif name == BlobDetectorConfigKeys.min_inertia:
            self.blob_inert_slider.blockSignals(True)
            self.blob_inert_slider.setValue(int(val * 100))
            self.blob_inert_slider.blockSignals(False)

    def update_stereo_conf(self, val: float):
        self.config.set(TrackerConfigKeys.stereo_conf_threshold, val)
        self.stereo_conf_slider.blockSignals(True)
        self.stereo_conf_slider.setValue(int(val * 100))
        self.stereo_conf_slider.blockSignals(False)

    def toggle_recording(self, checked: bool):
        if checked:
            if not self.worker.recorder:
                self.worker.recorder = Recorder()
            self.worker.recorder.start()
            self.worker.recording_active = True
            self.record_button.setText("Stop Recording")
        else:
            self.worker.recording_active = False
            if self.worker.recorder:
                self.worker.recorder.stop()
            self.record_button.setText("Start Recording")

    def _get_color(self, val: float, green_val: float, red_val: float) -> str:
        """Returns a hex color string interpolating between green and red."""
        if green_val == red_val:
            return "#00FF00"
        # Normalized position: 0 at green_val, 1 at red_val
        t = (val - green_val) / (red_val - green_val)
        t = max(0.0, min(1.0, t))

        # Green: #00FF00 (0, 255, 0), Red: #FF0000 (255, 0, 0)
        r = int(255 * t)
        g = int(255 * (1.0 - t))
        b = 0
        return f"#{r:02x}{g:02x}{b:02x}"

    @QtCore.Slot(np.ndarray, np.ndarray, np.ndarray, np.ndarray)
    def on_frame(self, frame_l: np.ndarray, frame_r: np.ndarray, overlay_l: np.ndarray, overlay_r: np.ndarray):
        time_since_last_update = time.monotonic() - self.last_img_update_time
        if time_since_last_update < 1.0 / 30:
            return
        self.last_img_update_time = time.monotonic()
        for frame, overlay, img_item, overlay_item in [
            (frame_l, overlay_l, self.img_item_l, self.overlay_item_l),
            (frame_r, overlay_r, self.img_item_r, self.overlay_item_r)
        ]:
            img_item.setImage(frame, autoLevels=False)
            overlay_item.setImage(overlay, autoLevels=False)

    @QtCore.Slot(list, list)
    def on_candidates(self, candidates_l: list, candidates_r: list):
        pass

    @QtCore.Slot(bool, float, float, float, float)
    def on_centroid(self, is_detected: bool, pos_2d_raw_l_x: float, pos_2d_raw_l_y: float, pos_2d_raw_r_x: float,
                    pos_2d_raw_r_y: float):
        # Update Left
        if is_detected:
            frame_h = self.cam_resolution[1]
            # OpenCV is Y down, UI is Y up
            y_l = frame_h - pos_2d_raw_l_y
            y_r = frame_h - pos_2d_raw_r_y
            text_l = f"{pos_2d_raw_l_x:5.1f}, {y_l:5.1f}"
            text_r = f"{pos_2d_raw_r_x:5.1f}, {y_r:5.1f}"
        else:
            text_l = "N/A"
            text_r = "N/A"

        self.left_centroid_label.setText(f"Centroid L: {text_l}")
        self.right_centroid_label.setText(f"Centroid R: {text_r}")

    @QtCore.Slot(bool, float, float, float, float)
    def on_position(self, found: bool, pos_x: float, pos_y: float, pos_z: float, confidence: float):
        text = f"XYZ: {pos_x:7.2f}, {pos_y:7.2f}, {pos_z:7.2f}" if found else "XYZ: N/A"
        self.pos_label.setText(text)
        conf_text = f"Confidence: {confidence:3.2f}"
        self.conf_label.setText(conf_text)
        color = self._get_color(confidence, 1.0, 0.0)
        self.conf_label.setStyleSheet(f"background-color: {color};")
        if found:
            # Overwrite the oldest data
            self.pos_data[0, self.pos_ptr] = pos_x
            self.pos_data[1, self.pos_ptr] = pos_y
            self.pos_data[2, self.pos_ptr] = pos_z
            self.pos_ptr = (self.pos_ptr + 1) % self.pos_buffer_size
            # Flip z and y to transform to the space of the UI view
            self.pos_marker.setData(pos=np.array([[pos_x, pos_z, pos_y]]))
        time_since_last_plot = time.monotonic() - self.last_pos_plot_time
        if time_since_last_plot > 1.0 / 30: # Plot at 30 FPS
            self.last_pos_plot_time = time.monotonic()
            # Roll the array so the newest data is on the right
            # For small buffer sizes (<10,000), np.roll is extremely fast
            self.curve_x.setData(np.roll(self.pos_data[0], -self.pos_ptr))
            self.curve_y.setData(np.roll(self.pos_data[1], -self.pos_ptr))
            self.curve_z.setData(np.roll(self.pos_data[2], -self.pos_ptr))

    @QtCore.Slot(float, float, float, float, float)
    def on_stats(self, frame_time: float, time_of_capture_l: float, time_of_capture_r: float, time_of_arrival: float,
                 time_of_3d_detection: float):
        fps = float(1000.0 / frame_time)
        self.frame_time_label.setText(f"Frame time: {frame_time:5.1f}ms ({fps:5.1f} FPS)")
        target_period = 1000.0 / self.target_fps
        ft_color = self._get_color(frame_time, target_period, 1.2 * target_period)
        self.frame_time_label.setStyleSheet(f"background-color: {ft_color};")

        l_arrival = (time_of_arrival - time_of_capture_l) * 1e3
        self.capture_latency_label.setText(f"Capture latency: {l_arrival:5.1f}ms")
        cl_color = self._get_color(l_arrival, 15, 50)
        self.capture_latency_label.setStyleSheet(f"background-color: {cl_color};")

        if time_of_3d_detection >= 0:
            l_processing = (time_of_3d_detection - time_of_arrival) * 1e3
            self.processing_latency_label.setText(f"Processing latency: {l_processing:5.1f}ms")
            pl_color = self._get_color(l_processing, 1, 5)
            self.processing_latency_label.setStyleSheet(f"background-color: {pl_color};")

            l_total = l_arrival + l_processing
            self.total_latency_label.setText(f"Total latency: {l_total:5.1f}ms")
            tl_color = self._get_color(l_total, 16, 55)
            self.total_latency_label.setStyleSheet(f"background-color: {tl_color};")
        else:
            self.processing_latency_label.setText("Processing latency: N/A")
            self.processing_latency_label.setStyleSheet("")
            self.total_latency_label.setText("Total latency: N/A")
            self.total_latency_label.setStyleSheet("")

        diff_ts = (time_of_capture_r - time_of_capture_l) * 1e6
        self.lr_diff_label.setText(f"L-R frame diff: {diff_ts:5.1f}µs")
        lr_color = self._get_color(diff_ts, 20, 100)
        self.lr_diff_label.setStyleSheet(f"background-color: {lr_color};")

    def closeEvent(self, event):
        self.config.save_file()
        self.worker.stop()
        event.accept()
