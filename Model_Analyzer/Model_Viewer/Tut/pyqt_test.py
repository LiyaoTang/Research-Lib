import sys

import PyQt5 as qt
from PyQt5.QtWidgets import *
from PyQt5.QtCore import *
from PyQt5.QtGui import *

class Color(QLabel):

    def __init__(self, color, *args, **kwargs):
        super(Color, self).__init__(*args, **kwargs)
        self.setAutoFillBackground(True)
        
        palette = self.palette()
        palette.setColor(QPalette.Window, QColor(color))
        self.setPalette(palette)


class CustomDialog(QDialog):

    def __init__(self, *args, **kwargs):
        super(CustomDialog, self).__init__(*args, **kwargs)
        
        self.setWindowTitle("HELLO!")
        
        QBtn = QDialogButtonBox.Ok | QDialogButtonBox.Cancel # bitmap for defined button
        
        # create selecetd buttons (with the bitmap & respect to host's button layout standard)
        self.buttonBox = QDialogButtonBox(QBtn)
        self.buttonBox.accepted.connect(self.accept)
        self.buttonBox.rejected.connect(self.reject)

        self.layout = QVBoxLayout()
        self.layout.addWidget(self.buttonBox)  # add to layout
        self.setLayout(self.layout)

# class CustomQGL(qt.QtWidgets.QGLWidget):

#     def __init__(self, *args, **kwargs):
#         super(CustomQGL, self).__init__(*args, **kwargs)
        
#         self.setWindowTitle("HELLO!")
        
#         QBtn = QDialogButtonBox.Ok | QDialogButtonBox.Cancel # bitmap for defined button
        
#         # create selecetd buttons (with the bitmap & respect to host's button layout standard)
#         self.buttonBox = QDialogButtonBox(QBtn)
#         self.buttonBox.accepted.connect(self.accept)
#         self.buttonBox.rejected.connect(self.reject)

#         self.layout = QVBoxLayout()
#         self.layout.addWidget(self.buttonBox)  # add to layout
#         self.setLayout(self.layout)



class MainWindow(QMainWindow):
    def __init__(self, *args, **kwargs):
        super(MainWindow, self).__init__(*args, **kwargs)

        # connect signal to multiple callbacks
        self.windowTitleChanged.connect(self.onWindowTitleChange)
        self.windowTitleChanged.connect(lambda x: self.my_custom_fn())
        self.windowTitleChanged.connect(lambda x: self.my_custom_fn(x))
        self.windowTitleChanged.connect(lambda x: self.my_custom_fn(x, 'extra'))
        
        self.setWindowTitle("My Awesome App")  # trigger all the above signals

        # configure icon setting
        self.setToolButtonStyle(Qt.ToolButtonTextUnderIcon)

        # initialize QAction - functionality oriented (a functionality allowing UI components to register an entry to)
        button_action = QAction(QIcon("stickman.png"), "Your button", self)  # self (MainWindow) as its UI parent
        button_action.setStatusTip("This is your button")
        button_action.setCheckable(True)
        button_action.setShortcuts(QKeySequence("Ctrl+a")) # set short cut from qt namespace
        button_action.triggered.connect(self.onMyToolBarButtonClick)

        # action2
        button_action2 = QAction(QIcon("stickman.png"), "Your button", self)  # self (MainWindow) as its UI parent
        button_action2.setStatusTip("This is your button 2")
        button_action2.setCheckable(True)
        button_action2.triggered.connect(self.onMyToolBarButtonClick)

        # action3 - not registered
        button_action3 = QAction(QIcon("stickman.png"), "Your button", self)  # self (MainWindow) as its UI parent
        button_action3.setStatusTip("This is your button 3")
        button_action3.setCheckable(True)
        button_action3.setShortcuts(QKeySequence("Ctrl+p")) # set short cut from qt namespace
        button_action3.triggered.connect(self.onShortCut)

        # configure toolbar
        toolbar = QToolBar("My main toolbar")
        toolbar.setIconSize(QSize(16, 16))  # configure local icon size
        # adding stuff to tool bar
        toolbar.addAction(button_action)  # register an entry to button_action
        toolbar.addSeparator()
        toolbar.addAction(button_action2)
        toolbar.addWidget(QLabel("Hello"))
        toolbar.addWidget(QCheckBox())
        self.addToolBar(toolbar)

        # status bar (with default setting)
        self.setStatusBar(QStatusBar(self)) # default at the bottom with left aligned

        # configure menu bar
        menu = self.menuBar()
        file_menu = menu.addMenu("&File") # & creates a shortcut Alt+f
        file_menu.addAction(button_action)  # register entry to same button action again (toggling the same underlain object)
        file_menu.addSeparator()
        file_menu.addMenu("Submenu")  # sub menu
        file_menu.addAction(button_action2)

        # nested layout
        layout1 = QHBoxLayout() # horizontal
        layout2 = QVBoxLayout() # vertical
        layout3 = QVBoxLayout() # vertical

        layout2.addWidget(Color('red'))
        layout2.addWidget(Color('yellow'))
        layout2.addWidget(Color('purple'))

        layout3.addWidget(Color('red'))
        layout3.addWidget(Color('purple'))

        layout1.addLayout( layout2 )
        layout1.addWidget(Color('green'))        
        layout1.addLayout( layout3 )
        layout1.setContentsMargins(0, 0, 0, 0) # spacing
        layout1.setSpacing(20)

        # Grid layout
        layout4 = QGridLayout()
        layout4.addWidget(Color('red'), 0, 0)
        layout4.addWidget(Color('green'), 1, 0)
        layout4.addWidget(Color('blue'), 1, 1)
        layout4.addWidget(Color('purple'), 2, 2)

        line = QWidget()
        line.setFixedHeight(2)
        line.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        line.setStyleSheet("background-color: #c0c0c0;")
        
        line2 = QFrame()
        line2.setFixedHeight(3)
        line2.setFrameShape(QFrame.HLine)

        # stacked layout
        layout5 = QStackedLayout()
        layout5.addWidget(Color('red'))
        layout5.addWidget(Color('green'))
        layout5.addWidget(Color('blue'))
        layout5.addWidget(Color('yellow'))        
        layout5.setCurrentIndex(3)

        # tab layout with button
        pagelayout = QVBoxLayout() # page as a container
        button_layout = QHBoxLayout() # button
        content_layout = QStackedLayout() # content

        for n, color in enumerate(['red','green','blue','yellow']):
            btn = QPushButton( str(color) )
            btn.pressed.connect( lambda n=n: content_layout.setCurrentIndex(n) ) # button to change displayed page (stacked layout)
            button_layout.addWidget(btn) # add button to button layout (horizontal)
            content_layout.addWidget(Color(color))  # add corresponding content to content layout (stacked)
        # add to the page
        pagelayout.addLayout(button_layout)
        pagelayout.addLayout(content_layout)

        # tab widget
        tabs = QTabWidget()
        tabs.setDocumentMode(True)
        tabs.setTabPosition(QTabWidget.North)
        tabs.setMovable(True)

        for color in ['red','green','blue','yellow']:
            tabs.addTab( Color(color), color)

        # overall layout
        layout = QVBoxLayout() # new layout
        # widgets = [QCheckBox, QComboBox, QDateEdit, QDateTimeEdit, QDial, QDoubleSpinBox, QFontComboBox, QLCDNumber,
        #            QLabel, QLineEdit, QProgressBar, QPushButton, QRadioButton, QSlider, QSpinBox, QTimeEdit]        

        layout.addWidget(tabs) # add a tab widget

        layout.addLayout(pagelayout) # add a tab layout (self-implemented)
        layout.addWidget(line2)

        layout.addLayout(layout1) # add a nested layout
        layout.addWidget(line)

        layout.addLayout(layout4) # add a grid layout
        layout.addWidget(line2)

        layout.addLayout(layout5) # add a stacked layout
        layout.addWidget(line)

        # color
        w = Color('red', 'asdasdasd')
        layout.addWidget(w)

        # label
        w = QLabel('Hello')
        font = w.font()
        font.setPointSize(30)
        w.setFont(font)
        w.setAlignment(Qt.AlignHCenter|Qt.AlignVCenter)
        layout.addWidget(w)
        
        # image
        w = QLabel('asd')
        w.setPixmap(QPixmap('stickman.png'))
        w.setScaledContents(True) # scale to all available space (ratio discarded)
        layout.addWidget(w)

        # checkbox
        w = QCheckBox()
        w.setCheckState(Qt.Checked)
        w.stateChanged.connect(self.show_state)
        layout.addWidget(w)

        # combo box
        w = QComboBox()
        w.addItems(["One", "Tasdasdwo", "Three"])
        w.currentIndexChanged.connect(self.index_changed)
        w.currentIndexChanged[str].connect(self.text_changed)  # The same signal can also send a text string
        w.setEditable(True)
        layout.addWidget(w)

        # list box
        w = QListWidget()
        w.addItems(["One", "Two", "Three"])
        w.currentItemChanged.connect(self.index_changed)
        w.currentTextChanged.connect(self.text_changed)  # The same signal can also send a text string
        layout.addWidget(w)

        # editable line
        w = QLineEdit()
        w.setMaxLength(10)
        w.setPlaceholderText("Enter your text")
        w.setClearButtonEnabled(True)
        # widget.setReadOnly(True) # can be readonly
        # w.setInputMask('000.000.000.000;_')
        w.returnPressed.connect(self.return_pressed)
        w.selectionChanged.connect(self.selection_changed)
        w.textChanged.connect(self.text_changed)
        w.textEdited.connect(self.text_edited)
        layout.addWidget(w)

        # spin box
        w = QSpinBox()
        w.setMinimum(-5)
        w.setMaximum(5)
        w.setPrefix("$")
        w.setSuffix("c")
        w.setSingleStep(2)
        print(w.valueChanged)
        w.valueChanged.connect(self.value_changed)
        w.valueChanged[str].connect(self.value_changed_str)
        layout.addWidget(w)

        # slider
        widget = QSlider(Qt.Horizontal) # or QDoubleSpinBox()
        widget.setMinimum(-10)
        widget.setMaximum(3)
        # Or: widget.setRange(-10,3)
        widget.setSingleStep(3) # Or e.g. 0.5 for QDoubleSpinBox
        widget.valueChanged.connect(self.value_changed)
        widget.sliderMoved.connect(self.slider_position)        
        widget.sliderPressed.connect(self.slider_pressed)
        widget.sliderReleased.connect(self.slider_released)
        layout.addWidget(widget)

        # dial
        widget = QDial()
        widget.setRange(-10,100)
        widget.setSingleStep(0.5)
        widget.valueChanged.connect(self.value_changed)
        widget.sliderMoved.connect(self.slider_position)        
        widget.sliderPressed.connect(self.slider_pressed)
        widget.sliderReleased.connect(self.slider_released)
        layout.addWidget(widget)

        # container widget to hold layout (with buttons inside)
        widget = QWidget()
        widget.setLayout(layout)
        self.setCentralWidget(widget) # take up all the space in the window by default.

    def onMyToolBarButtonClick(self, s):
        print("onMyToolBarButtonClick ", s)

        dlg = CustomDialog(self) # a dialog
        rst = dlg.exec_()  # a blocking dialog
        if rst:
            print('success')
        else:
            print('cancel')

    def slider_position(self, p):
        print("position", p)
        
    def slider_pressed(self):
        print("Pressed!")
        
    def slider_released(self):
        print("Released")

    def value_changed(self, i):
        print('val ch', i)
    def value_changed_str(self, s):
        print(s)

    def return_pressed(self):
        print("Return pressed!")
        print(self.centralWidget())
        self.centralWidget().setText("BOOM!")

    def selection_changed(self):
        print("Selection changed")
        print(self.centralWidget().selectedText)

    def text_edited(self, s):
        print("Text edited:",s)

    def index_changed(self, i):
        print('idx ch', i)

    def text_changed(self, i):
        print('text ch', i)

    def show_state(self, s):
        print(s, s == Qt.Checked)

    def onShortCut(self, s):
        print('onShortCut', s)

    # automatic registered to corresponding event
    def contextMenuEvent(self, e):
        print("Context menu event!")
        super(MainWindow, self).contextMenuEvent(e)  # propogate to parent class
        e.accept()  # hide event from parent UI component
        e.ignore()  # propogate to parent UI component

    # SLOT: This accepts a string, e.g. the window title, and prints it
    def onWindowTitleChange(self, s):
        print('onWindowTitleChange', s)
        
    # SLOT: This has default parameters and can be called without a value
    def my_custom_fn(self, a="default_a", b='default_b'):
        print('my_custom_fn', a, b)

# one (and only one) QApplication instance per application.
app = QApplication(sys.argv) # [] for no args

# at least one (… but can have more); app exits when last main window is closed
window = MainWindow()
window.show() # hidden by default.

# Start the event loop.
app.exec_()
# Your application won't reach here until you exit and the event 
# loop has stopped