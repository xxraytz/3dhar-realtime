import argparse
import json
import logging
from core.mainthread import MainThread, VideoStreamingThread, PeopleMonitoringThread
import cv2
import numpy as np

import sys
from PyQt6.QtWidgets import QWidget, QApplication, QLabel, QVBoxLayout, QPushButton, QToolBar, QMainWindow, QStatusBar, \
    QScrollArea, QListView, QMessageBox, QGridLayout
from PyQt6.QtGui import QPixmap, QColor, QImage, QCloseEvent, QAction, QStandardItemModel, QStandardItem, QActionGroup, \
    QResizeEvent
from PyQt6.QtCore import Qt, pyqtSlot, pyqtSignal, QTimer

logging.basicConfig(format='[%(asctime)s][%(levelname)s][%(name)s] %(message)s', datefmt='%H:%M:%S')
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def get_config(args):
    with open(args.config, 'r') as stream:
        config = json.load(stream)
        EXCLUDE_SKELETON_DISTANCES = config["EXCLUDE_SKELETON_DISTANCES"]
        SKELETON_COLORS = config["SKELETON_COLORS"]
        USABLE_SK_JOINTS = config["SKELETON_USABLE_JOINTS_NUMBER"]
        USABLE_SK_JOINTS = dict(
            sorted(
                USABLE_SK_JOINTS.items(),
                key=lambda x: x[1]
            )
        )
        SKELETON_LINES = config["SKELETON_LINES"]
        SKELETON_POINTS_COUNT = config["SKELETON_POINTS_COUNT"]
        SKELETON_HEIGHTS_CORRECTION = config["SKELETON_HEIGHTS_CORRECTION"]
    config = args.config
    return config


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='config.json', help='JSON config')
    parser.add_argument('-v', default=1, help='Kinect data visualisation')
    return parser.parse_args()


class App(QWidget):
    set_people_list = pyqtSignal(list)

    def __init__(self, config):
        super().__init__()
        self.conf = config
        self.size = (1280, 800)
        self.__people_list = None
        self.image_label = QLabel(self)
        self.image_label.resize(*self.size)
        self.text_label = QLabel('', )
        self.text_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.text_label.setStyleSheet("background-color:white; color:red; font-size:60px; font-weight:bold;"
                                      "border: 1px")
        self.timer = QTimer(self)
        self.timer.setSingleShot(False)
        self.timer.timeout.connect(self.hide_label)

        self.pause_button = QPushButton()
        self.pause_button.setText('pause')
        self.pause_button.setFixedWidth(200)
        self.pause_button.clicked.connect(self.pause)

        self.grid_results = QWidget()

        # self.grid_results_layout.addWidget(self.__styled_label_for_grid('Id1'), 0, 1)
        # self.grid_results_layout.addWidget(self.__styled_label_for_grid('Id2'), 0, 2)
        # self.grid_results_layout.addWidget(self.__styled_label_for_grid('OS'), 1, 0)
        # self.grid_results_layout.addWidget(self.__styled_label_for_grid('Egor'), 2, 0)
        #
        self.main_layout = QVBoxLayout()
        self.main_layout.addWidget(self.image_label)
        self.main_layout.addWidget(self.text_label)
        self.main_layout.addWidget(self.pause_button)
        self.main_layout.addWidget(self.grid_results)

        self.setLayout(self.main_layout, )
        grey_image = QPixmap(*self.size)
        grey_image.fill(QColor('grey'))
        self.image_label.setPixmap(grey_image)

        self.main_thread = MainThread(self.conf)
        self.video_streaming_thread = VideoStreamingThread(self.conf)
        self.people_monitoring_thread = PeopleMonitoringThread(self.conf, 6)

        self.video_streaming_thread.video_frame_sending.connect(self.update_image)
        self.people_monitoring_thread.send_calculated_indexes.connect(self.set_people_indexes)
        self.people_monitoring_thread.send_debug_info.connect(self.debug_get_grid)
        self.people_monitoring_thread.send_people_information.connect(self.update_people_identification_list)
        self.main_thread.video_frame_generate.connect(self.video_streaming_append)
        self.main_thread.video_frame_generate.connect(self.monitoring_append)
        # self.main_thread.video_frame_generate.connect(self.people_monitoring_thread.queue.put)

    @staticmethod
    def __styled_label_for_grid(text):
        label = QLabel(text)
        label.setStyleSheet(" color:black; font-size:20px")
        return label

    def hide_label(self):
        self.text_label.setText('')

    def show_label(self):
        self.text_label.setText('IDs have been updated')
        self.timer.start(1300)

    def start(self):
        self.main_thread.start()
        self.video_streaming_thread.start()
        self.people_monitoring_thread.database_connect()
        self.people_monitoring_thread.start()

    @pyqtSlot(list)
    def set_monitoring_people_list(self, people):
        self.__people_list = people
        self.people_monitoring_thread.set_monitoring_people(people)

    def pause(self):
        self.main_thread.pause()

    @pyqtSlot(object)
    def monitoring_append(self, image):
        if not self.people_monitoring_thread.is_busy():
            self.people_monitoring_thread.queue.put(image)

    @pyqtSlot(object)
    def video_streaming_append(self, image):
        if self.video_streaming_thread.queue.empty():
            self.video_streaming_thread.queue.put(image)

    @pyqtSlot(dict, dict)
    def debug_get_grid(self, info: dict, info_one_class: dict):
        if self.__people_list:
            people_number = len(self.__people_list)
            grid_info = np.asarray(self.__people_list).reshape(-1, 1)
            indexes = list(info.keys())
            header = ["", *indexes]
            for i in indexes:
                try:
                    clf_res = info[i]
                    parts_number = len(clf_res)
                    values = np.asarray([res[name] for name in self.__people_list for res in clf_res])
                    values = values.reshape([people_number, parts_number])
                    values = ["\n".join(value.astype(str)) for value in values]
                    grid_info = np.hstack((grid_info, np.asarray(values).reshape(-1, 1)))
                except IndexError:
                    logger.info('No more people')
            grid_info = np.vstack((header, grid_info))
            grid_info = np.vstack((grid_info, np.asarray(['one-class'] + ['{:f}'.format(np.max(np.round(info_one_class[id][0][0], 7))) for id in indexes])))
            self.debug_set_grid_to_interface(grid_info)

    def debug_set_grid_to_interface(self, grid):
        grid_results_layout = QGridLayout()
        for i in range(grid.shape[0]):
            for j in range(grid.shape[1]):
                grid_results_layout.addWidget(self.__styled_label_for_grid(f'{grid[i, j]}'), i, j)
        self.main_layout.removeWidget(self.grid_results)
        self.grid_results = QWidget()
        self.grid_results.setLayout(grid_results_layout)
        self.grid_results.adjustSize()
        self.main_layout.addWidget(self.grid_results)
        # self.grid_results.update()
        self.main_layout.update()
        self.update()


    @pyqtSlot(list)
    def set_people_indexes(self, people_ids):
        logger.info(people_ids)
        self.show_label()
        self.video_streaming_thread.set_indexes([f'{x}' for x in people_ids])
        self.people_monitoring_thread.stop()

    def closeEvent(self, event: QCloseEvent) -> None:
        self.main_thread.stop()
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
        self.setWindowTitle("OpenCV video demo")
        self.main_widget = App(get_config(args))
        self.setCentralWidget(self.main_widget)
        toolbar = QToolBar("Mine Main Menu")
        self.addToolBar(toolbar)

        button_action = QAction('MenuButton', self)
        button_action.setStatusTip('This is my button')
        button_action.triggered.connect(self.on_click_event)
        toolbar.addAction(button_action)

        self.setStatusBar(QStatusBar(self))

        menu = self.menuBar()
        menu.setNativeMenuBar(True)
        file_menu = menu.addMenu("&File")
        file_menu.addAction(button_action)

        setting_menu = menu.addMenu("&Settings")
        change_video_mode_action = setting_menu.addMenu('&Video mode')
        change_video_mode_action.setStatusTip("Change video mode (Real Video, Skeletal Video)")
        choose_people_to_id = QAction('Choose people', self)
        choose_people_to_id.setStatusTip('Choose more than one people to identify')
        choose_people_to_id.triggered.connect(self.on_choose_people)

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

    def resizeEvent(self, a0: QResizeEvent) -> None:
        ...

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
        self.main_widget.close()

    def on_click_event(self, s):
        logger.info(f'Click some of button {s}')
        self.main_widget.update()
        self.update()



if __name__ == '__main__':
    args = parse_arguments()
    app = QApplication(sys.argv)
    a = MainWindow(args)
    a.show()
    sys.exit(app.exec())
