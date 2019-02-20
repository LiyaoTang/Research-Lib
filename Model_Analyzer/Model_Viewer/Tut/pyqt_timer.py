from PyQt5.QtCore import *
from PyQt5.QtWidgets import *
import sys

class TestWindow(QDialog):
    def __init__(self):
        
        super().__init__()

        self.sec = 0

        btn1 = QPushButton("Start", self)
        btn2 = QPushButton("Stop", self)
        self.sec_label = QLabel(self)

        layout = QGridLayout(self)
        layout.addWidget(btn1,0,0)
        layout.addWidget(btn2,0,1)
        layout.addWidget(self.sec_label,1,0,1,2)

        timer = QTimer()
        timer.timeout.connect(self.update) # 计时器挂接到槽：update
        btn1.clicked.connect(lambda :timer.start(1000))
        btn2.clicked.connect(lambda :timer.stop())
        

    def update(self):
        self.sec += 1
        self.sec_label.setText(str(self.sec))


app=QApplication(sys.argv)
form=TestWindow()
form.show()
app.exec_()