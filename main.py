from GUI import Ui_MainWindow
import sys
from PyQt5 import QtCore, QtGui, QtWidgets
import cv2
import os
from PyQt5.QtGui import *
from PyQt5.QtWidgets import *
from PyQt5.QtWidgets import QMessageBox
import FaceDetection
import FaceRecognition




class ApplicationWindow(QtWidgets.QMainWindow):
    def __init__(self):
        super(ApplicationWindow, self).__init__()
        self.ui = Ui_MainWindow()
        self.ui.setupUi(self)

        self.ui.actionImage_Recognition.triggered.connect(self.Face_Recognition)
        self.ui.actionImage_Detection.triggered.connect(self.Face_Detection)

    def Face_Detection(self):
        self.ui.label_4.setText('Processed image')
        filename, _ = QFileDialog.getOpenFileNames(self, "Select file(s)", " ", "Images (*.png *.xpm *.jpg )")
        if len(filename) == 1 and filename != []:
            img = cv2.imread(f'{filename[0]}')
            self.Display_Original_img(img)
            Face_img = FaceDetection.face_detection(img)
            self.display_processed_img(Face_img)

    def Face_Recognition(self):
        self.ui.label_4.setText('Recognized image')
        self.ui.processed_image.clear()
        filename, _ = QFileDialog.getOpenFileNames(self, "Select file(s)", " ", "Images (*.png *.xpm *.jpg *.pgm)")
        if len(filename) == 1 and filename != []:
            test_img = cv2.imread(f'{filename[0]}', 0)
            self.Display_Original_img(test_img)

            dataset_path = os.getcwd() + '/FaceDataset/'
            num_of_imgs, names, all_img = FaceRecognition.Dataset_Info_Extractor(dataset_path)
            eigenfaces, mean_vector, zero_mean_img = FaceRecognition.Processing(num_of_imgs, all_img)
            Face_found = FaceRecognition.Face_recognetion(test_img, eigenfaces, mean_vector, num_of_imgs, names,
                                                               zero_mean_img)
            if Face_found:
                img_found = cv2.imread(f'{filename[0]}', 0)
                self.display_processed_img(img_found)

            else:
                self.PopUp()


    def PopUp(self):
        msg = QMessageBox()
        msg.setWindowTitle("Face Recognition")
        msg.setText("This Face is not in the Data set")
        msg.setIcon(QMessageBox.Information)
        msg.setStandardButtons(QMessageBox.Ok)
        msg.setDefaultButton(QMessageBox.Ok)
        msg.exec_()


    def display_processed_img(self, img):
        cv2.imwrite("result.jpg", img)
        result = QPixmap("result.jpg").scaled(500, 500)
        self.ui.processed_image.setPixmap(QPixmap(result))



    def Display_Original_img(self, img):

        cv2.imwrite("Original_gray_img.jpg", img)
        gry_result = QPixmap("Original_gray_img.jpg").scaled(500, 500)

        self.ui.original_image.setPixmap(QPixmap(gry_result))


def main():
    app = QtWidgets.QApplication(sys.argv)
    application = ApplicationWindow()
    application.show()
    app.exec_()

if __name__ == "__main__":
    main()

