import argparse
import json
import logging
import os
from datetime import datetime
from core.videothread import VideoThread
import cv2
import numpy as np

import sys
from PyQt6.QtWidgets import QWidget, QApplication, QLabel, QVBoxLayout, QPushButton, QToolBar, QMainWindow, QStatusBar, \
    QScrollArea, QListView, QMessageBox
from PyQt6.QtGui import QPixmap, QColor, QImage, QCloseEvent, QAction, QStandardItemModel, QStandardItem
from PyQt6.QtCore import Qt, QThread, pyqtSlot, pyqtSignal

logging.basicConfig(format='[%(asctime)s][%(levelname)s][%(name)s] %(message)s', datefmt='%H:%M:%S')
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class App(QWidget):
    def __init__(self):
        super().__init__()
        self.size = (640, 480)
        self.image_label = QLabel(self)
        self.image_label.resize(*self.size)
        self.text_label = QLabel('OpenCV demo')
        self.pause_button = QPushButton()
        self.pause_button.setText('pause')
        self.pause_button.setFixedWidth(300)
        self.pause_button.clicked.connect(self.pause)

        vbox = QVBoxLayout()
        vbox.addWidget(self.image_label)
        vbox.addWidget(self.text_label)
        vbox.addWidget(self.pause_button)
        self.setLayout(vbox, )
        grey_image = QPixmap(*self.size)
        grey_image.fill(QColor('grey'))
        self.image_label.setPixmap(grey_image)

        self.thread = VideoThread()
        self.thread.change_pixmap_signal.connect(self.update_image)  # type: ignore
        self.thread.start()

    def pause(self):
        self.thread.pause()

    def closeEvent(self, event: QCloseEvent) -> None:
        self.thread.stop()
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


class PeopleListWindow(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Choose a people to recognize")
        self.resize(300, 400)
        self.string_list = ['String 1', 'String 2', 'String 3', 'String 4', ]

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
            item.setCheckable(True)
            model.appendRow(item)
        return model

    def on_accept_button_clicked(self):
        selected_indexes = self.list_view.selectedIndexes()
        selected_strings = [self.string_list[index.row()] for index in selected_indexes]
        print(selected_strings)

        if len(selected_strings) > 0:
            QMessageBox.information(self, 'Выбранные строки', "\n".join(selected_strings))
        else:
            QMessageBox.warning(self, "Предупреждение", "Не выбрано ни одной строки")


class MainWindow(QMainWindow):
    def __init__(self):
        super(MainWindow, self).__init__()
        self.people_window = PeopleListWindow()
        self.setWindowTitle("OpenCV video demo")
        self.main_widget = App()
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
        change_video_mode_action = QAction('Video mode', self)
        change_video_mode_action.setStatusTip("Change video mode (Real Video, Skeletal Video)")
        choose_people_to_id = QAction('Choose people', self)
        choose_people_to_id.setStatusTip('Choose more than one people to identify')
        choose_people_to_id.triggered.connect(self.on_choose_people)

        setting_menu.addAction(change_video_mode_action)
        setting_menu.addAction(choose_people_to_id)

    def on_choose_people(self):

        self.people_window.show()

    def closeEvent(self, event: QCloseEvent) -> None:
        self.main_widget.close()

    @staticmethod
    def on_click_event(s):
        print("click", s)


if __name__ == '__main__':
    app = QApplication(sys.argv)
    a = MainWindow()
    a.show()
    sys.exit(app.exec())
