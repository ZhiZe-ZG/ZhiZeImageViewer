from PyQt5.QtWidgets import QFileDialog
import numpy as np
import cv2 as cv
from PyQt5 import QtGui
from PyQt5.QtWidgets import QApplication, QLabel, QVBoxLayout, \
    QHBoxLayout, QWidget, QPushButton
from numpy.lib.type_check import imag
from Algorithms import RGB2BGR, RGB2GRAY


class ImageTemp:
    '''
    image temp
    '''
    def __init__(self):
        '''
        creat ImageTemp object
        '''
        self._image_data = np.zeros([512, 512, 3])
        self._image_data = self._image_data.astype(np.uint8)

    def set_image_data(self, image_data: np.ndarray):
        '''
        update image_data
        '''
        self._image_data = image_data
        self._image_data = self._image_data.astype(np.uint8)

    def get_image_data(self) -> np.ndarray:
        '''
        return a copy of image_data
        '''
        return self._image_data.copy()

    def get_QImage(self):
        '''
        return QImage of data
        '''
        image_bytes = self._image_data.data.tobytes()
        qimage = QtGui.QImage(
            image_bytes,
            self._image_data.shape[1],
            self._image_data.shape[0],
            self._image_data.shape[1] * 3,
            # keep image not skew
            QtGui.QImage.Format_RGB888)
        return qimage


class MainWindow:
    def __init__(self):
        '''
        init all the widgets
        '''
        self.image_temp = ImageTemp()
        self.start_path = 'C:/'
        self.button_openfile = QPushButton('Open File')
        self.button_openfile.clicked.connect(self.load_image)
        self.button_saveas = QPushButton('Save As')
        self.button_saveas.clicked.connect(self.save_image)
        self.button_rgb2bgr = QPushButton('RGB To BGR')
        self.button_rgb2bgr.clicked.connect(self.rgb2bgr)
        self.button_rgb2gray = QPushButton('RGB To Gray')
        self.button_rgb2gray.clicked.connect(self.rgb2gray)
        self.image_label = QLabel('this is an image')
        self.tool_box = self.get_tool_box()
        self.window = self.get_main_window()

    def get_tool_box(self):
        '''
        layout of tool box
        '''
        tool_box = QWidget()
        layout = QVBoxLayout()
        layout.addWidget(self.button_openfile)
        layout.addWidget(self.button_saveas)
        layout.addWidget(self.button_rgb2bgr)
        layout.addWidget(self.button_rgb2gray)
        tool_box.setLayout(layout)
        return tool_box

    def get_main_window(self):
        '''
        window and layout of main window
        '''
        window = QWidget()
        layout = QHBoxLayout()
        layout.addWidget(self.image_label)
        self.update_image_label()
        layout.addWidget(self.tool_box)
        window.setLayout(layout)
        return window

    def update_image_label(self):
        '''
        update image show
        '''
        self.image_label.setPixmap(QtGui.QPixmap(self.image_temp.get_QImage()))

    def load_image(self):
        '''
        load a image
        '''
        data_path, _ = QFileDialog.getOpenFileName(None, "Select File",
                                                   self.start_path,
                                                   "*.png;*.jpg;*.jpeg")
        image = cv.imread(data_path, cv.IMREAD_COLOR)
        image = cv.cvtColor(image, cv.COLOR_BGR2RGB)
        self.image_temp.set_image_data(image)
        self.update_image_label()

    def save_image(self):
        '''
        save image temp as a image
        '''
        save_path, _ = QFileDialog.getSaveFileName(None, "Save As",
                                                   self.start_path,
                                                   "*.png;*.jpg;*.jpeg")
        image = self.image_temp.get_image_data()
        image = cv.cvtColor(image, cv.COLOR_RGB2BGR)
        cv.imwrite(save_path, image)

    def rgb2bgr(self):
        '''
        rgb image show as bgr image
        '''
        image = self.image_temp.get_image_data()
        image = RGB2BGR(image)
        self.image_temp.set_image_data(image)
        self.update_image_label()

    def rgb2gray(self):
        '''
        rgb image to gray image
        '''
        image = self.image_temp.get_image_data()
        image = RGB2GRAY(image)
        image = np.array([image, image, image])
        image = np.transpose(image, (1, 2, 0))
        print(image.shape)
        self.image_temp.set_image_data(image)
        print('data',self.image_temp.get_image_data().shape)
        self.update_image_label()

# create app
app = QApplication([])
main_window = MainWindow()
main_window.window.show()
# run app
app.exec()
