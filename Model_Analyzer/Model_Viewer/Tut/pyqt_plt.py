#!/usr/bin/env python
import sys
 
from PyQt5.QtWidgets import QApplication, QMainWindow, QMenu, QVBoxLayout, QHBoxLayout, QSizePolicy, QMessageBox, QWidget, QPushButton
from PyQt5.QtGui import QIcon
 
import matplotlib
matplotlib.use("Qt5Agg")
 
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
 
import random
 
class App(QMainWindow):
 
    def __init__(self):
        super().__init__()
        self.left = 10
        self.top = 10
        self.title = 'PyQt5 matplotlib example'
        self.width = 640
        self.height = 400
        
        self.initUI()
 
    def initUI(self):
        self.setWindowTitle(self.title)
        self.setGeometry(self.left, self.top, self.width, self.height)
 
        m = PlotCanvas(self, width=5, height=4)
 
        button = QPushButton('PyQt5 button', self)
        button.setToolTip('This s an example button')
        
        layout = QHBoxLayout()
        layout.addWidget(m)
        layout.addWidget(button)
        w = QWidget()
        w.setLayout(layout)
        self.setCentralWidget(w)
        
        self.show()
 
 
class PlotCanvas(matplotlib.backends.backend_qt5agg.FigureCanvasQTAgg):
 
    def __init__(self, parent=None, width=5, height=4, dpi=100):
        fig = Figure(figsize=(width, height), dpi=dpi)
        self.axes = fig.add_subplot(111)
 
        super(PlotCanvas, self).__init__(fig)
        self.setParent(parent)
 
        super(PlotCanvas, self).setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        super(PlotCanvas, self).updateGeometry()
        self.plot()
 
    def plot(self):
        data = [random.random() for i in range(25)]
        ax = self.figure.add_subplot(111)
        ax.plot(data, 'r-')
        ax.set_title('PyQt Matplotlib Example')
        self.draw()
 
if __name__ == '__main__':
    app = QApplication(sys.argv)
    ex = App()
    sys.exit(app.exec_())