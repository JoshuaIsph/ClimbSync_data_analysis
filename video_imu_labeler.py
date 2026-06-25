#!/usr/bin/env python3
"""
Video + IMU Synchronized Labeling Tool
Synchronize iPhone video with smartwatch IMU data and label events.
"""

import sys
import csv
from pathlib import Path
import numpy as np
import pandas as pd
import cv2
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QLabel, QPushButton, QSpinBox, QSlider, QFileDialog, QListWidget, QListWidgetItem,
    QInputDialog, QMessageBox, QTableWidget, QTableWidgetItem
)
from PyQt5.QtCore import Qt, QTimer, QRect
from PyQt5.QtGui import QImage, QPixmap, QColor
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg
from matplotlib.figure import Figure


class VideoIMULabeler(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Video + IMU Sync & Labeler (Multi-IMU)")
        self.setGeometry(100, 100, 1600, 1200)
        
        # Data
        self.video_path = None
        self.cap = None
        self.video_fps = 30
        self.current_frame_idx = 0
        self.labels = []  # Shared labels across all IMUs
        
        # Multi-IMU data (5 devices)
        self.num_imus = 5
        self.imu_data = [None] * self.num_imus  # List of dataframes
        self.sync_offset_ms = [0] * self.num_imus  # Per-IMU offset
        self.start_frame = [None] * self.num_imus
        self.start_offset_ms = [0] * self.num_imus
        self.end_frame = [None] * self.num_imus
        self.end_offset_ms = [0] * self.num_imus
        self.drift_rate_ppm = [0] * self.num_imus
        
        # UI Timer
        self.play_timer = QTimer()
        self.play_timer.timeout.connect(self.play_frame)
        self.is_playing = False
        
        self.init_ui()
    
    def init_ui(self):
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QHBoxLayout(central_widget)
        
        # Left: Video + Timeline
        left_layout = QVBoxLayout()
        
        # Video display
        self.video_label = QLabel("Load video first")
        self.video_label.setMinimumSize(480, 360)
        self.video_label.setStyleSheet("border: 1px solid gray; background-color: black;")
        left_layout.addWidget(QLabel("VIDEO"))
        left_layout.addWidget(self.video_label)
        
        # Timeline controls (spinbox + slider)
        timeline_spinbox_layout = QHBoxLayout()
        timeline_spinbox_layout.addWidget(QLabel("Frame:"))
        self.frame_spinbox = QSpinBox()
        self.frame_spinbox.valueChanged.connect(self.on_frame_spinbox_change)
        timeline_spinbox_layout.addWidget(self.frame_spinbox)
        self.time_label = QLabel("0.00 s")
        timeline_spinbox_layout.addWidget(self.time_label)
        left_layout.addLayout(timeline_spinbox_layout)
        
        # Video timeline slider
        self.frame_slider = QSlider(Qt.Horizontal)
        self.frame_slider.setRange(0, 0)
        self.frame_slider.sliderMoved.connect(self.on_frame_slider_change)
        left_layout.addWidget(self.frame_slider)
        
        # Play controls
        control_layout = QHBoxLayout()
        load_video_btn = QPushButton("Load Video")
        load_video_btn.clicked.connect(self.load_video)
        control_layout.addWidget(load_video_btn)
        
        self.play_btn = QPushButton("Play")
        self.play_btn.clicked.connect(self.toggle_play)
        control_layout.addWidget(self.play_btn)
        
        left_layout.addLayout(control_layout)
        
        # Right: IMU Plots + Sync Controls + Labels
        right_layout = QVBoxLayout()
        
        # Multi-IMU Plot (stacked)
        self.imu_figure = Figure(figsize=(8, 10), dpi=100)
        self.imu_canvas = FigureCanvasQTAgg(self.imu_figure)
        self.imu_canvas.mpl_connect('button_press_event', self.on_imu_plot_click)
        right_layout.addWidget(QLabel("IMU SIGNALS x5 (Click to label)"))
        right_layout.addWidget(self.imu_canvas, 1)
        
        # Load IMU buttons
        imu_buttons_layout = QHBoxLayout()
        self.imu_load_btns = []
        for i in range(self.num_imus):
            btn = QPushButton(f"Load IMU {i+1}")
            btn.clicked.connect(lambda checked, idx=i: self.load_imu(idx))
            imu_buttons_layout.addWidget(btn)
            self.imu_load_btns.append(btn)
        right_layout.addLayout(imu_buttons_layout)
        
        # IMU selector (for sync controls)
        imu_selector_layout = QHBoxLayout()
        imu_selector_layout.addWidget(QLabel("Sync IMU:"))
        self.imu_selector = QSpinBox()
        self.imu_selector.setRange(1, self.num_imus)
        self.imu_selector.setValue(1)
        self.imu_selector.valueChanged.connect(self.on_imu_selector_change)
        imu_selector_layout.addWidget(self.imu_selector)
        imu_selector_layout.addStretch()
        right_layout.addLayout(imu_selector_layout)
        
        # Sync offset control (slider + spinbox)
        sync_label_layout = QHBoxLayout()
        sync_label_layout.addWidget(QLabel("Sync Offset (ms):"))
        self.sync_spinbox = QSpinBox()
        self.sync_spinbox.setRange(-100000, 100000)
        self.sync_spinbox.setSingleStep(100)
        self.sync_spinbox.valueChanged.connect(self.on_sync_spinbox_change)
        sync_label_layout.addWidget(self.sync_spinbox)
        sync_label_layout.addStretch()
        right_layout.addLayout(sync_label_layout)
        
        # Slider
        self.sync_slider = QSlider(Qt.Horizontal)
        self.sync_slider.setRange(-100000, 100000)
        self.sync_slider.setValue(0)
        self.sync_slider.setTickPosition(QSlider.TicksBelow)
        self.sync_slider.setTickInterval(10000)
        self.sync_slider.sliderMoved.connect(self.on_sync_slider_change)
        right_layout.addWidget(self.sync_slider)
        
        # Calibration buttons
        calib_layout = QHBoxLayout()
        set_start_btn = QPushButton("Set Start (Clap)")
        set_start_btn.clicked.connect(self.set_start_calibration)
        calib_layout.addWidget(set_start_btn)
        
        set_end_btn = QPushButton("Set End (Clap)")
        set_end_btn.clicked.connect(self.set_end_calibration)
        calib_layout.addWidget(set_end_btn)
        
        reset_calib_btn = QPushButton("Reset Calibration")
        reset_calib_btn.clicked.connect(self.reset_calibration)
        calib_layout.addWidget(reset_calib_btn)
        right_layout.addLayout(calib_layout)
        
        # Drift rate control
        drift_layout = QHBoxLayout()
        drift_layout.addWidget(QLabel("Drift Rate (ppm):"))
        self.drift_label = QLabel("0.0")
        drift_layout.addWidget(self.drift_label)
        self.drift_slider = QSlider(Qt.Horizontal)
        self.drift_slider.setRange(-100, 100)
        self.drift_slider.setValue(0)
        self.drift_slider.setTickPosition(QSlider.TicksBelow)
        self.drift_slider.setTickInterval(10)
        self.drift_slider.sliderMoved.connect(self.on_drift_slider_change)
        drift_layout.addWidget(self.drift_slider)
        right_layout.addLayout(drift_layout)
        
        # Labeling
        right_layout.addWidget(QLabel("LABELS"))
        self.labels_list = QListWidget()
        right_layout.addWidget(self.labels_list)
        
        label_btn_layout = QHBoxLayout()
        add_label_btn = QPushButton("Add Label at Current Time")
        add_label_btn.clicked.connect(self.add_label)
        label_btn_layout.addWidget(add_label_btn)
        
        remove_label_btn = QPushButton("Remove Selected")
        remove_label_btn.clicked.connect(self.remove_label)
        label_btn_layout.addWidget(remove_label_btn)
        
        export_btn = QPushButton("Export CSV")
        export_btn.clicked.connect(self.export_labels)
        label_btn_layout.addWidget(export_btn)
        
        right_layout.addLayout(label_btn_layout)
        
        # Main split
        main_layout.addLayout(left_layout, 40)
        main_layout.addLayout(right_layout, 60)
    
    def load_video(self):
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Select Video", "", "Video Files (*.mp4 *.mov *.avi);;All Files (*)"
        )
        if file_path:
            self.video_path = file_path
            self.cap = cv2.VideoCapture(file_path)
            self.video_fps = self.cap.get(cv2.CAP_PROP_FPS) or 30
            frame_count = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
            self.frame_spinbox.setRange(0, frame_count - 1)
            self.frame_slider.setRange(0, frame_count - 1)
            self.current_frame_idx = 0
            self.show_frame(0)
            QMessageBox.information(self, "Success", f"Loaded video: {Path(file_path).name}\n{self.video_fps} fps")
    
    def load_imu(self, imu_idx):
        file_path, _ = QFileDialog.getOpenFileName(
            self, f"Select IMU {imu_idx+1} CSV", "", "CSV Files (*.csv);;All Files (*)"
        )
        if file_path:
            # Skip comment lines (starting with #)
            self.imu_data[imu_idx] = pd.read_csv(file_path, comment='#')
            self.plot_imu()
            QMessageBox.information(self, "Success", f"Loaded IMU {imu_idx+1}: {len(self.imu_data[imu_idx])} rows")
    
    def on_imu_selector_change(self, imu_num):
        imu_idx = imu_num - 1
        self.sync_spinbox.blockSignals(True)
        self.sync_slider.blockSignals(True)
        self.sync_spinbox.setValue(self.sync_offset_ms[imu_idx])
        self.sync_slider.setValue(self.sync_offset_ms[imu_idx])
        self.drift_label.setText(f"{self.drift_rate_ppm[imu_idx]:.1f}")
        self.sync_spinbox.blockSignals(False)
        self.sync_slider.blockSignals(False)
        self.plot_imu()
    
    def show_frame(self, frame_idx):
        if not self.cap:
            return
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = self.cap.read()
        if ret:
            self.current_frame_idx = frame_idx
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            h, w, ch = frame.shape
            frame_resized = cv2.resize(frame, (480, int(480 * h / w)))
            h, w = frame_resized.shape[:2]
            
            # Draw sync time on video (for current selected IMU)
            video_time_s = frame_idx / self.video_fps
            imu_idx = self.imu_selector.value() - 1
            corrected_offset = self.get_corrected_offset_ms(frame_idx, imu_idx)
            imu_time_s = (video_time_s * 1000 + corrected_offset) / 1000
            cv2.putText(
                frame_resized,
                f"Video: {video_time_s:.2f}s | IMU{imu_idx+1}: {imu_time_s:.2f}s",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 0),
                2
            )
            
            qt_frame = QImage(frame_resized.data, w, h, 3 * w, QImage.Format_RGB888)
            self.video_label.setPixmap(QPixmap.fromImage(qt_frame))
            
            self.frame_spinbox.blockSignals(True)
            self.frame_spinbox.setValue(frame_idx)
            self.frame_spinbox.blockSignals(False)
            
            self.frame_slider.blockSignals(True)
            self.frame_slider.setValue(frame_idx)
            self.frame_slider.blockSignals(False)
            
            self.time_label.setText(f"{video_time_s:.2f} s → IMU{imu_idx+1}: {imu_time_s:.2f}s")
            self.plot_imu()
    
    def on_frame_spinbox_change(self, frame_idx):
        self.show_frame(frame_idx)
    
    def on_frame_slider_change(self, frame_idx):
        self.show_frame(frame_idx)
    
    def toggle_play(self):
        if not self.cap:
            QMessageBox.warning(self, "Error", "Load a video first!")
            return
        self.is_playing = not self.is_playing
        self.play_btn.setText("Pause" if self.is_playing else "Play")
        if self.is_playing:
            self.play_timer.start(int(1000 / self.video_fps))
        else:
            self.play_timer.stop()
    
    def play_frame(self):
        frame_count = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        next_frame = self.current_frame_idx + 1
        if next_frame < frame_count:
            self.show_frame(next_frame)
        else:
            self.toggle_play()
    
    def on_sync_spinbox_change(self, value):
        imu_idx = self.imu_selector.value() - 1
        self.sync_offset_ms[imu_idx] = value
        # If we have a start calibration, preview end offset
        if self.start_frame[imu_idx] is not None:
            self.end_offset_ms[imu_idx] = value
            self.recalculate_drift(imu_idx)
        self.sync_slider.blockSignals(True)
        self.sync_slider.setValue(value)
        self.sync_slider.blockSignals(False)
        self.show_frame(self.current_frame_idx)

    def on_sync_slider_change(self, value):
        imu_idx = self.imu_selector.value() - 1
        self.sync_offset_ms[imu_idx] = value
        # If we have a start calibration, preview end offset and recalculate drift
        if self.start_frame[imu_idx] is not None:
            self.end_offset_ms[imu_idx] = value
            self.recalculate_drift(imu_idx)
        self.sync_spinbox.blockSignals(True)
        self.sync_spinbox.setValue(value)
        self.sync_spinbox.blockSignals(False)
        self.show_frame(self.current_frame_idx)
    
    def set_start_calibration(self):
        imu_idx = self.imu_selector.value() - 1
        self.start_frame[imu_idx] = self.current_frame_idx
        self.start_offset_ms[imu_idx] = self.sync_offset_ms[imu_idx]
        self.recalculate_drift(imu_idx)
        QMessageBox.information(self, "Start Calibration", f"IMU{imu_idx+1}: Start point set at frame {self.start_frame[imu_idx]} with offset {self.start_offset_ms[imu_idx]}ms")
    
    def set_end_calibration(self):
        imu_idx = self.imu_selector.value() - 1
        self.end_frame[imu_idx] = self.current_frame_idx
        self.end_offset_ms[imu_idx] = self.sync_offset_ms[imu_idx]
        self.recalculate_drift(imu_idx)
        QMessageBox.information(self, "End Calibration", f"IMU{imu_idx+1}: End point set at frame {self.end_frame[imu_idx]} with offset {self.end_offset_ms[imu_idx]}ms")
    
    def recalculate_drift(self, imu_idx):
        # Calculate drift if we have start frame and either confirmed end frame or preview end offset
        end_frame_to_use = self.end_frame[imu_idx] if self.end_frame[imu_idx] is not None else self.current_frame_idx
        
        if self.start_frame[imu_idx] is not None and self.start_frame[imu_idx] != end_frame_to_use:
            frame_diff = end_frame_to_use - self.start_frame[imu_idx]
            offset_diff_ms = self.end_offset_ms[imu_idx] - self.start_offset_ms[imu_idx]
            self.drift_rate_ppm[imu_idx] = (offset_diff_ms / frame_diff) * (self.video_fps * 1e6 / 1000)
            self.drift_label.setText(f"{self.drift_rate_ppm[imu_idx]:.1f}")
            self.plot_imu()
    
    def reset_calibration(self):
        imu_idx = self.imu_selector.value() - 1
        self.start_frame[imu_idx] = None
        self.start_offset_ms[imu_idx] = 0
        self.end_frame[imu_idx] = None
        self.end_offset_ms[imu_idx] = 0
        self.drift_rate_ppm[imu_idx] = 0
        self.sync_offset_ms[imu_idx] = 0
        self.sync_spinbox.blockSignals(True)
        self.sync_slider.blockSignals(True)
        self.sync_spinbox.setValue(0)
        self.sync_slider.setValue(0)
        self.drift_label.setText("0.0")
        self.sync_spinbox.blockSignals(False)
        self.sync_slider.blockSignals(False)
        self.plot_imu()
    
    def on_drift_slider_change(self, value):
        imu_idx = self.imu_selector.value() - 1
        self.drift_rate_ppm[imu_idx] = value
        self.drift_label.setText(f"{self.drift_rate_ppm[imu_idx]:.1f}")
        self.plot_imu()
    
    def get_corrected_offset_ms(self, frame_idx, imu_idx):
        if self.start_frame[imu_idx] is not None:
            frame_offset = frame_idx - self.start_frame[imu_idx]
            drift_correction = (frame_offset / self.video_fps) * (self.drift_rate_ppm[imu_idx] / 1e6) * 1000
            return self.start_offset_ms[imu_idx] + drift_correction
        return self.sync_offset_ms[imu_idx]
    
    def plot_imu(self):
        if self.cap is None:
            return
        
        self.imu_figure.clear()
        
        # Create 5 stacked subplots (one per IMU)
        axes = []
        for i in range(self.num_imus):
            ax = self.imu_figure.add_subplot(self.num_imus, 1, i + 1)
            axes.append(ax)
        
        video_time_ms = self.current_frame_idx / self.video_fps * 1000
        
        # Plot each IMU
        for imu_idx in range(self.num_imus):
            ax = axes[imu_idx]
            
            if self.imu_data[imu_idx] is None:
                ax.text(0.5, 0.5, f'IMU {imu_idx+1}: No data', ha='center', va='center', transform=ax.transAxes)
                ax.set_ylabel(f"IMU{imu_idx+1}")
                continue
            
            # Extract time and accel magnitude
            time_ms = pd.to_numeric(self.imu_data[imu_idx]['Time(ms)'], errors='coerce')
            accel_x = pd.to_numeric(self.imu_data[imu_idx]['Accel.X(m/s²)'], errors='coerce')
            accel_y = pd.to_numeric(self.imu_data[imu_idx]['Accel.Y(m/s²)'], errors='coerce')
            accel_z = pd.to_numeric(self.imu_data[imu_idx]['Accel.Z(m/s²)'], errors='coerce')
            accel_mag = np.sqrt(accel_x**2 + accel_y**2 + accel_z**2)
            
            if len(time_ms) > 0:
                time_ms_normalized = time_ms - time_ms.iloc[0]
                
                # Plot accel magnitude
                color = ['blue', 'green', 'red', 'purple', 'orange'][imu_idx]
                ax.plot(time_ms_normalized, accel_mag, color=color, linewidth=0.8)
                
                # Current cursor with drift correction
                corrected_offset = self.get_corrected_offset_ms(self.current_frame_idx, imu_idx)
                imu_cursor_ms = video_time_ms + corrected_offset
                ax.axvline(imu_cursor_ms, color='red', linestyle='--', linewidth=2, alpha=0.7)
                
                # Calibration points
                if self.start_frame[imu_idx] is not None:
                    start_imu_ms = (self.start_frame[imu_idx] / self.video_fps * 1000) + self.start_offset_ms[imu_idx]
                    ax.axvline(start_imu_ms, color='green', linestyle=':', linewidth=1.5, alpha=0.5)
                if self.end_frame[imu_idx] is not None:
                    end_imu_ms = (self.end_frame[imu_idx] / self.video_fps * 1000) + self.end_offset_ms[imu_idx]
                    ax.axvline(end_imu_ms, color='orange', linestyle=':', linewidth=1.5, alpha=0.5)
                
                # Shared labels
                for label_time_ms, label_text in self.labels:
                    ax.axvline(label_time_ms, color='purple', linestyle=':', linewidth=1, alpha=0.4)
                
                ax.set_ylabel(f"IMU{imu_idx+1} (m/s²)")
                ax.grid(True, alpha=0.3)
                
                # Highlight if this is the selected IMU
                if imu_idx == self.imu_selector.value() - 1:
                    ax.set_facecolor('#ffffcc')
        
        axes[-1].set_xlabel("Time (ms)")
        self.imu_figure.tight_layout()
        self.imu_canvas.draw()
    
    def on_imu_plot_click(self, event):
        if event.inaxes is None:
            return
        
        # Clicked time in IMU coordinates
        clicked_time_ms = event.xdata
        if clicked_time_ms is None:
            return
        
        # Determine which IMU was clicked by checking which axis
        imu_idx = self.imu_selector.value() - 1
        
        # Convert IMU time back to video time
        video_time_ms = clicked_time_ms - self.sync_offset_ms[imu_idx]
        video_time_s = video_time_ms / 1000
        frame_idx = int(video_time_s * self.video_fps)
        frame_count = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        if 0 <= frame_idx < frame_count:
            self.show_frame(frame_idx)
    
    def add_label(self):
        if not self.cap or self.imu_data is None:
            QMessageBox.warning(self, "Error", "Load video and IMU first!")
            return
        
        label_text, ok = QInputDialog.getText(self, "Add Label", "Label name:")
        if ok and label_text:
            video_time_ms = self.current_frame_idx / self.video_fps * 1000
            self.labels.append((video_time_ms + self.sync_offset_ms, label_text))
            self.update_labels_list()
            self.plot_imu()
    
    def update_labels_list(self):
        self.labels_list.clear()
        for time_ms, label_text in sorted(self.labels):
            self.labels_list.addItem(f"{time_ms:.1f}ms - {label_text}")
    
    def remove_label(self):
        row = self.labels_list.currentRow()
        if row >= 0:
            self.labels.pop(row)
            self.update_labels_list()
            self.plot_imu()
    
    def export_labels(self):
        if not self.labels:
            QMessageBox.warning(self, "Error", "No labels to export!")
            return
        
        file_path, _ = QFileDialog.getSaveFileName(
            self, "Save Labels", "labels.csv", "CSV Files (*.csv)"
        )
        if file_path:
            with open(file_path, 'w', newline='') as f:
                writer = csv.writer(f)
                # Header: timestamp + label + 5 IMU columns
                header = ['timestamp_ms', 'label']
                for i in range(self.num_imus):
                    header.append(f'imu{i+1}_offset_ms')
                writer.writerow(header)
                
                # Export each label with all 5 IMU offsets
                for label_time_ms, label_text in sorted(self.labels):
                    row = [f"{label_time_ms:.1f}", label_text]
                    
                    # Calculate corrected offsets for each IMU
                    for imu_idx in range(self.num_imus):
                        if self.imu_data[imu_idx] is not None:
                            if self.start_frame[imu_idx] is not None:
                                # Convert back to frame index
                                video_time_ms = label_time_ms - self.start_offset_ms[imu_idx]
                                frame_idx = int((video_time_ms / 1000) * self.video_fps)
                                corrected_offset = self.get_corrected_offset_ms(frame_idx, imu_idx)
                            else:
                                corrected_offset = self.sync_offset_ms[imu_idx]
                            row.append(f"{corrected_offset:.1f}")
                        else:
                            row.append("N/A")
                    
                    writer.writerow(row)
            
            QMessageBox.information(self, "Success", f"Exported {len(self.labels)} labels to {Path(file_path).name}")
            msg = "Drift correction applied:\n"
            for i in range(self.num_imus):
                msg += f"IMU{i+1}: {self.drift_rate_ppm[i]:.1f} ppm\n"
            QMessageBox.information(self, "Info", msg)


def main():
    app = QApplication(sys.argv)
    window = VideoIMULabeler()
    window.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
