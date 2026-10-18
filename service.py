import argparse
import os
import queue
from core.config import ROOT, load_config
import logging
from core.mainthread import MainThread, VideoStreamingThread, PeopleMonitoringThread, HARThread
import cv2
import numpy as np

import sys
from PyQt6.QtWidgets import QWidget, QApplication, QLabel, QVBoxLayout, QPushButton, QMainWindow, QStatusBar, \
    QScrollArea, QListView, QMessageBox
from PyQt6.QtGui import QPixmap, QColor, QImage, QCloseEvent, QAction, QStandardItemModel, QStandardItem, QActionGroup
from PyQt6.QtCore import Qt, pyqtSlot, pyqtSignal, QTimer

logging.basicConfig(format='[%(asctime)s][%(levelname)s][%(name)s] %(message)s', datefmt='%H:%M:%S')
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def parse_arguments():
    parser = argparse.ArgumentParser(description='Skeleton-based activity recognition with Kinect v2.')
    parser.add_argument('--config', default=str(ROOT / 'config.example.json'))
    parser.add_argument('--source', choices=['kinect', 'recording'], default='kinect')
    parser.add_argument('--recording', help='Scene directory containing Body/ and Color/.')
    parser.add_argument('--identify', action='store_true', help='Enable identification using private local assets.')
    args = parser.parse_args()
    if args.source == 'recording' and not (args.recording or os.environ.get('emulator_base_path')):
        parser.error('--source recording requires --recording or emulator_base_path')
    return args


class App(QWidget):
    set_people_list = pyqtSignal(list)

    def __init__(self, args):
        super().__init__()
        self.conf = args.config
        load_config(self.conf)
        self.identify = args.identify
        self._closing = False
        self.size = (1280, 800)
        self.__people_list = None
        self.image_label = QLabel(self)
        self.image_label.resize(*self.size)
        self.text_label = QLabel('', )
        self.text_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.text_label.setStyleSheet("background-color:white; color:red; font-size:60px; font-weight:bold;"
                                      "border: 1px")
        self.timer = QTimer(self)
        self.timer.setSingleShot(True)
        self.timer.timeout.connect(self.hide_label)

        self.pause_button = QPushButton()
        self.pause_button.setText('Pause / resume')
        self.pause_button.setFixedWidth(200)
        self.pause_button.clicked.connect(self.pause)

        self.main_layout = QVBoxLayout()
        self.main_layout.addWidget(self.image_label)
        self.main_layout.addWidget(self.text_label)
        self.main_layout.addWidget(self.pause_button)

        self.setLayout(self.main_layout, )
        grey_image = QPixmap(*self.size)
        grey_image.fill(QColor('grey'))
        self.image_label.setPixmap(grey_image)

        self.main_thread = MainThread(self.conf, args.source, args.recording)
        self.video_streaming_thread = VideoStreamingThread(self.conf)
        self.people_monitoring_thread = PeopleMonitoringThread(self.conf, 6)
        self.human_action_recognition_thread = HARThread(self.conf, 6)

        self.video_streaming_thread.video_frame_sending.connect(self.update_image)
        self.people_monitoring_thread.send_calculated_indexes.connect(self.set_people_indexes)
        # Identity diagnostics are available from the worker; keep the main UI focused.
        self.people_monitoring_thread.send_people_information.connect(self.update_people_identification_list)
        self.main_thread.video_frame_generate.connect(self.video_streaming_append)
        self.main_thread.video_frame_generate.connect(self.human_action_recognition)
        if self.identify:
            self.main_thread.video_frame_generate.connect(self.monitoring_append)
        self.human_action_recognition_thread.send_action_info.connect(self.set_people_actions)
        for worker in self.workers():
            worker.failed.connect(self.on_worker_error)

    def workers(self):
        workers = [self.main_thread, self.video_streaming_thread, self.human_action_recognition_thread]
        if self.identify:
            workers.append(self.people_monitoring_thread)
        return workers

    @pyqtSlot(str)
    def on_worker_error(self, message):
        if not self._closing:
            self.shutdown()
            QMessageBox.critical(self, 'Capture stopped', message)

    def shutdown(self):
        self._closing = True
        self.timer.stop()
        for worker in self.workers():
            worker.requestInterruption()
        for worker in self.workers():
            worker.wait()

    @staticmethod
    def enqueue(frames, frame):
        try:
            frames.put_nowait(frame)
        except queue.Full:
            pass


    def hide_label(self):
        self.text_label.setText('')

    def show_label(self):
        self.text_label.setText('IDs have been updated')
        self.timer.start(1300)

    def start(self):
        # Validate assets before capture starts, so failures do not strand workers.
        self.human_action_recognition_thread.database_connect()
        if self.identify:
            self.people_monitoring_thread.database_connect()
        for worker in self.workers():
            worker.start()

    @pyqtSlot(list)
    def set_monitoring_people_list(self, people):
        self.__people_list = people
        self.people_monitoring_thread.set_monitoring_people(people)

    def pause(self):
        self.main_thread.pause()

    @pyqtSlot(object)
    def human_action_recognition(self, image):
        if not self._closing and not self.human_action_recognition_thread.is_busy():
            self.enqueue(self.human_action_recognition_thread.q_skeletons, image)

    @pyqtSlot(object)
    def monitoring_append(self, image):
        if not self._closing and not self.people_monitoring_thread.is_busy():
            self.enqueue(self.people_monitoring_thread.queue, image)

    @pyqtSlot(object)
    def video_streaming_append(self, image):
        if not self._closing:
            self.enqueue(self.video_streaming_thread.queue, image)

    @pyqtSlot(list)
    def set_people_actions(self, classes):
        self.video_streaming_thread.set_classes(classes)

    @pyqtSlot(list)
    def set_people_indexes(self, people_ids):
        self.show_label()
        self.video_streaming_thread.set_indexes([f'{x}' for x in people_ids])

    def closeEvent(self, event: QCloseEvent) -> None:
        self.shutdown()
        event.accept()

    def convert_cv_qt(self, cv_img):
        rgb_image = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb_image.shape
        bytes_per_line = ch * w
        converted_image = QImage(rgb_image, w, h, bytes_per_line, QImage.Format.Format_RGB888)
        scaled_converted_image = converted_image.scaled(*self.size, Qt.AspectRatioMode.KeepAspectRatio)
        return QPixmap.fromImage(scaled_converted_image)

    @pyqtSlot(np.ndarray)
    def update_image(self, cv_image):
        qt_image = self.convert_cv_qt(cv_image)
        self.image_label.setPixmap(qt_image)

    @pyqtSlot(list)
    def update_people_identification_list(self, people):
        self.set_people_list.emit(people)


class PeopleListWindow(QWidget):
    change_monitoring_people_list = pyqtSignal(list)

    def __init__(self):
        super().__init__()
        self.setWindowTitle("Choose a people to recognize")
        self.resize(300, 400)
        self.string_list = []

        layout = QVBoxLayout()
        list_widget = QWidget()
        list_layout = QVBoxLayout()
        list_widget.setLayout(list_layout)

        self.list_view = QListView()
        self.list_view.setSelectionMode(QListView.SelectionMode.ExtendedSelection)
        self.list_view.setModel(self.create_model())

        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)
        scroll_area.setWidget(self.list_view)

        list_layout.addWidget(scroll_area)

        layout.addWidget(list_widget)

        button_accept = QPushButton("Accept")
        button_accept.clicked.connect(self.on_accept_button_clicked)
        layout.addWidget(button_accept)

        self.setLayout(layout)

    def create_model(self):
        model = QStandardItemModel()
        for string in self.string_list:
            item = QStandardItem(string)
            item.setCheckable(False)
            model.appendRow(item)
        return model

    def set_people_list(self, str_list):
        if str_list:
            self.string_list = str_list
            self.list_view.setModel(self.create_model())

    def on_accept_button_clicked(self):
        selected_indexes = self.list_view.selectedIndexes()
        selected_strings = [self.string_list[index.row()] for index in selected_indexes]
        logger.info(f'Selected people: {selected_strings}')
        if len(selected_strings) > 0:
            QMessageBox.information(self, 'Выбранные строки', "\n".join(selected_strings))
            self.change_monitoring_people_list.emit(selected_strings)
            self.close()
        else:
            QMessageBox.warning(self, "Предупреждение", "Не выбрано ни одной строки")


class MainWindow(QMainWindow):
    def __init__(self, args):
        super(MainWindow, self).__init__()
        self.setGeometry(100, 100, 0, 0)
        self.people_window = PeopleListWindow()
        self.setWindowTitle("Skeleton Activity Recognition")
        self.main_widget = App(args)
        self.setCentralWidget(self.main_widget)
        self.setStatusBar(QStatusBar(self))

        menu = self.menuBar()
        menu.setNativeMenuBar(True)

        setting_menu = menu.addMenu("&Settings")
        change_video_mode_action = setting_menu.addMenu('&Video mode')
        change_video_mode_action.setStatusTip("Change video mode (Real Video, Skeletal Video)")
        choose_people_to_id = QAction('Choose people', self)
        choose_people_to_id.setStatusTip('Choose more than one people to identify')
        choose_people_to_id.triggered.connect(self.on_choose_people)

        choose_people_to_id.setEnabled(args.identify)
        setting_menu.addAction(choose_people_to_id)

        video_mode_group = QActionGroup(self, )
        self.video_mode_skeletal = QAction('Skeletal video', self)
        self.video_mode_real = QAction('Real video', self)
        self.video_mode_skeletal.setStatusTip("Skeletal video")
        self.video_mode_real.setStatusTip("Real video")
        self.video_mode_skeletal.setCheckable(True)
        self.video_mode_real.setCheckable(True)
        self.video_mode_skeletal.setChecked(True)

        self.video_mode_skeletal.triggered.connect(self.set_video_mode_skeletal)
        self.video_mode_real.triggered.connect(self.set_video_mode_real)

        self.main_widget.set_people_list.connect(self.update_people_list)
        video_mode_group.addAction(self.video_mode_real)
        video_mode_group.addAction(self.video_mode_skeletal)

        self.people_window.change_monitoring_people_list.connect(self.main_widget.set_monitoring_people_list)

        change_video_mode_action.addAction(self.video_mode_real)
        change_video_mode_action.addAction(self.video_mode_skeletal)
        self.main_widget.start()

    def set_video_mode_real(self):
        self.main_widget.video_streaming_thread.set_real_video_mode()

    def set_video_mode_skeletal(self):
        self.main_widget.video_streaming_thread.set_skeletal_video_mode()

    @pyqtSlot(list)
    def update_people_list(self, people):
        self.people_window.set_people_list(people)

    def on_choose_people(self):
        self.people_window.show()

    def closeEvent(self, event: QCloseEvent) -> None:
        self.main_widget.shutdown()
        self.people_window.close()
        event.accept()



if __name__ == '__main__':
    args = parse_arguments()
    app = QApplication(sys.argv)
    a = MainWindow(args)
    a.show()
    sys.exit(app.exec())
