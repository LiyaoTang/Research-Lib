#!/usr/bin/env python
# coding: utf-8
'''
module: construct QT viewer for model analysis
script: display the viewer UI
'''


import sys
root_dir = '../../'
sys.path.append(root_dir)
sys.path.append(root_dir + 'Data/corner/scripts')

import os
import matplotlib
matplotlib.use('Qt5Agg')

import numpy as np
import PyQt5.Qt as Qt
import PyQt5.QtGui as qtui
import PyQt5.QtCore as qtc
import PyQt5.QtOpenGL as qgl
import PyQt5.QtWidgets as qtwg
import matplotlib.pyplot as plt
import Model_Analyzer.Loader as loader

from matplotlib.figure import Figure
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas

class Matlibplot_Widget(qtwg.QWidget):
    '''
    qt widget to display a matplotlib image - can be interactive
    '''
    def __init__(self, scatter=True, interactive=True, cmap='rainbow'):
        super(Matlibplot_Widget, self).__init__()
        # plot setting
        self.scatter = scatter
        self.cmap = plt.get_cmap(cmap)
        self.press = None
        self.min_y = -20
        self.max_y = 20
        self.min_x = -100
        self.max_x = 60
        self.visible_step = min([self.max_x - self.min_x, self.max_y - self.min_y]) / 50

        # plot object
        self.figure = Figure(tight_layout=True)
        self.canvas = FigureCanvas(self.figure)
        self.canvas.setSizePolicy(qtwg.QSizePolicy.Expanding, qtwg.QSizePolicy.Expanding)
        
        # button
        button = qtwg.QPushButton('clear annotation', self)
        button.pressed.connect(self._reset_ann)

        # container layout
        layout = qtwg.QVBoxLayout(self)
        layout.addWidget(self.canvas)
        layout.addWidget(button)
        self.setLayout(layout)

        if interactive:
            self.canvas.mpl_connect('button_press_event', self._on_press)
            self.canvas.mpl_connect('button_release_event', self._on_release)
            self.canvas.mpl_connect('motion_notify_event', self._on_motion)
            self.canvas.mpl_connect('pick_event', self._on_pick)

    def _reset_ann(self):
        for ann, xy in zip(self.ann_list, self.xy_arr):
            ann.set_visible(False)
            ann.set_position(xy)
        self.canvas.draw_idle()

    def _on_press(self, event):
        # left mouse with move to enable ann; right to disable ann
        if event.inaxes != self.ax: return
        if event.button != 2:
            self.press = np.array([event.xdata, event.ydata])
            self.press_btn = event.button
            self.rect = matplotlib.patches.Rectangle(self.press, .0, .0, fill=False)
            self.ax.add_patch(self.rect)

    def _on_motion(self, event):
        # expand the rect
        if self.press is None: return
        if event.inaxes != self.ax: return

        dx, dy = [event.xdata, event.ydata] - self.press
        self.rect.set_height(dy)
        self.rect.set_width(dx)
        self.canvas.draw_idle()

    def _on_release(self, event):
        # toggle all selected annotation
        if self.press is None: return
        self.rect.set_visible(False)
        self.rect.remove()

        start = [min([event.xdata, self.press[0]]), min([event.ydata, self.press[1]])]
        end = [max([event.xdata, self.press[0]]), max([event.ydata, self.press[1]])]
        idx_list = np.where(np.logical_and(start < self.xy_arr, self.xy_arr < end).all(axis=1))[0]
        for idx in idx_list:
            self.ann_list[idx].set_visible(self.press_btn == 1)

        self.canvas.draw_idle()
        self.press = None

    def _on_pick(self, event):
        if event.mouseevent.inaxes is None or event.mouseevent.inaxes != self.ax:
            return

        for idx in event.ind: # left click - True; else - False
            self.ann_list[idx].set_visible(event.mouseevent.button == 1)
            a = self.ann_list[idx]
        self.canvas.draw_idle() # update canvas

    def plot(self, xy_arr=None, annotation=None, color_arr=None, color_name=None, special_point=None):
        '''
        plot the x-y array, with annotation, color array and name for different color
        if any not specified, random value will take the place
        only support scatter plot at the moment
        '''
        if xy_arr is None:
            xy_arr = np.random.randn(50).reshape(25,2)
        if color_arr is None:
            color_arr = np.random.randn(25)
            # color_arr = np.random.randint(0, 2, size=25)
            # color_arr = np.ones(25)
        if annotation is None:
            annotation = ['%-.2f,%-.2f - %-.2f' % (xy[0], xy[1], c) for xy,c in zip(xy_arr,color_arr)]

        self.ann_list = []
        self.xy_arr = xy_arr

        # new figure & annotation
        self.figure.clear()
        self.ax = self.figure.add_subplot(111)
        self._set_coordinate()
        
        # a circle outside special point
        if special_point is not None:
            special_xy_arr = xy_arr[np.where(special_point)]
            for xy in special_xy_arr:
                circ = matplotlib.patches.Circle(xy, radius=2 * self.visible_step, fill=False, color='b')
                self.ax.add_patch(circ)
        self.ax.scatter(xy_arr[:, 0], xy_arr[:, 1], marker='.', c=color_arr, cmap=self.cmap, vmin=0, vmax=1, picker=True)
        if color_name:
            handles = []
            scalarmap = matplotlib.cm.ScalarMappable(norm=matplotlib.colors.Normalize(vmin=0, vmax=1), cmap=self.cmap)
            for (val, name) in color_name.items():
                # TODO: change label from line to dot
                handles.append(matplotlib.lines.Line2D([], [], color=scalarmap.to_rgba(val), marker='.', label=name))
            self.ax.legend(handles=handles)

        # annotate
        for xy, ann in zip(xy_arr, annotation):
            self.ann_list.append(self.ax.annotate(ann, xy=xy))
            self.ann_list[-1].set_visible(False)
            self.ann_list[-1].draggable()

        self.canvas.draw()
        
    def _set_coordinate(self, lane_width=3.75):
        min_y = self.min_y
        max_y = self.max_y
        min_x = self.min_x
        max_x = self.max_x

        xmajorLocator = matplotlib.ticker.MultipleLocator(5)
        xminorLocator = matplotlib.ticker.MultipleLocator(1)

        ymajorLocator = matplotlib.ticker.MultipleLocator(10)
        yminorLocator = matplotlib.ticker.MultipleLocator(2)

        self.ax.xaxis.set_major_locator(xmajorLocator)
        self.ax.xaxis.set_minor_locator(xminorLocator)

        self.ax.yaxis.set_major_locator(ymajorLocator)
        self.ax.yaxis.set_minor_locator(yminorLocator)
        self.ax.xaxis.grid(True, which='major')
        self.ax.yaxis.grid(True, which='major')

        self.ax.set_xlim(max_y, min_y, 2)
        self.ax.set_ylim(min_x, max_x, 10)
        
        c = 'lime'
        ls = (0,(9,10,3,10)) # 
        for scaler in [-2.5, -1.5, -0.5, 0.5, 1.5, 2.5]:
            self.ax.vlines(lane_width * scaler, min_x, max_x, colors=c, linestyle=ls)

        self.ax.vlines(0, min_x, max_x, colors='k')
        self.ax.hlines(0, min_y, max_y, colors='k')


class Model_Viewer(qtwg.QMainWindow):
    '''
    view & analyze the model with input, label, its prediction and meta data
    '''
    def __init__(self, dataset_name, root_dir='../../', model_path=None):
        super(Model_Viewer, self).__init__()
        self.models = loader.Model_Loader(dataset_name, root_dir, model_path=model_path, verbose=True)  # load all models
        self.models.load_dataset(verbose=True)
        self.data_files = self.models.list_all_file_path()
        self.data_idx = 0
        feature_names = ' '.join(self.models.dataset[-1].feature_names)

        self.setWindowTitle('Model Viewer')
        self.setToolButtonStyle(qtc.Qt.ToolButtonTextUnderIcon)  # icon style
        self.setWindowFlags(self.windowFlags() | qtc.Qt.WindowSystemMenuHint | qtc.Qt.WindowMinMaxButtonsHint)

        # next/last - action
        next_example = qtwg.QAction('next', self)
        next_example.setStatusTip('next example')
        next_example.setShortcuts(Qt.QKeySequence('Right'))
        next_example.triggered.connect(self._show_next_example)

        last_example = qtwg.QAction('last_example', self)
        last_example.setStatusTip('last example')
        last_example.setShortcuts(Qt.QKeySequence('Left'))
        last_example.triggered.connect(self._show_last_example)

        # refresh all - action
        refresh = qtwg.QAction('refresh', self)
        refresh.setStatusTip('refresh to begining')
        refresh.triggered.connect(self._refresh)

        # current & all data files
        combo = qtwg.QComboBox()
        combo.addItems([os.path.join(d,f) for (d,f) in self.data_files])
        combo.currentIndexChanged.connect(lambda i: self._fill_layout(idx=i))
        combo.setEditable(False)
        self.path_widget = combo

        # play examples set-tup
        play_examples = qtwg.QAction('play', self)
        play_examples.setStatusTip('play through all examples as auto-ppt')
        play_examples.triggered.connect(self._play_all_examples)
        self.timer = qtc.QTimer()
        self.timer.timeout.connect(self._show_next_example)
        self.play_widget = play_examples
        play_setting = qtwg.QLineEdit()
        play_setting.setFixedWidth(20)
        play_setting.setMaxLength(3)
        play_setting.setPlaceholderText('5s')
        play_setting.setClearButtonEnabled(True)
        # play_setting.textChanged.connect()

        # config toolbar, statusbar
        toolbar = qtwg.QToolBar("My main toolbar")
        toolbar.addAction(last_example)
        toolbar.addAction(next_example)
        toolbar.addWidget(combo)
        toolbar.addAction(play_examples)
        toolbar.addWidget(play_setting)
        toolbar.addAction(refresh)
        self.addToolBar(toolbar)
        self.setStatusBar(Qt.QStatusBar(self))

        # layout framework
        layout_main = qtwg.QHBoxLayout()

        # image - image label
        img_label = qtwg.QLabel('image')
        img_label.setScaledContents(True)
        img_label.setPixmap(qtui.QPixmap())
        layout_main.addWidget(img_label)
        self.img_widget = img_label

        # input/label - matlibplot widget
        input_layout = qtwg.QVBoxLayout()
        input_metaline = qtwg.QLabel(feature_names)
        input_widget = Matlibplot_Widget('input')
        input_layout.addWidget(input_metaline)
        input_layout.addWidget(input_widget)
        layout_main.addLayout(input_layout)
        self.input_widget = input_widget

        # prediction - tab
        pred_tab = qtwg.QTabWidget()
        pred_tab.setDocumentMode(True)
        pred_tab.setTabPosition(Qt.QTabWidget.North)
        pred_tab.setMovable(True)
        layout_main.addWidget(pred_tab)
        self.pred_widget = pred_tab

        # set layout to central widget
        central_widget = qtwg.QWidget()
        central_widget.setLayout(layout_main)
        self.setCentralWidget(central_widget)

        self._fill_layout()

    def _fill_layout(self, idx=None):
        if idx == self.data_idx:  # replicated update
            return
        elif type(idx) is int:  # data idx if new idx passed
            self.data_idx = idx
        # default to update with set self.data_idx

        data_path_list = self.data_files[self.data_idx]
        data_dir = data_path_list[0]
        data_name = data_path_list[1]
        if type(self.models.model_name) is str:
            data_dict = self.models.dataset.load_with_metadata(data_dir, data_name, self.models.pred_dir)
        else:
            data_dict = self.models.dataset[-1].load_with_metadata(data_dir, data_name, self.models.pred_dir)
        
        # update combo list
        self.path_widget.setCurrentIndex(self.data_idx)

        # update image
        img = data_dict['img']  # rgb
        height, width, channel = img.shape
        bytes_per_line = width * channel
        img_format = qtui.QImage.Format_RGB888
        qimage = qtui.QImage(img, width, height, bytes_per_line, img_format)
        self.img_widget.setPixmap(qtui.QPixmap(qimage))

        # update input/label
        points = data_dict['input']
        labels = data_dict['label'][:,1]
        # get x-y & transform
        xy_arr = points[:, [1, 0]]
        self.input_widget.plot(xy_arr, annotation=[','.join(['%-.2f' % i for i in row]) for row in points], color_arr=labels, color_name={0:'other', 1:'car'})
        
        # update prediction
        pred = data_dict['pred']
        for name, idx in zip(pred.keys(), range(len(pred.keys()))): # pred per model
            cur_pred = pred[name]
            wrong_pred = ((cur_pred > 0.5) != labels)

            cur_page = self.pred_widget.widget(idx)
            # not working: get page by name
            # cur_page = self.pred_widget.findChildren(qtwg.QWidget, name) 
            if cur_page:  # existing model
                cur_page.plot(xy_arr=xy_arr, annotation=[str(n) for n in cur_pred], color_arr=cur_pred, special_point=wrong_pred)
            else:  # new model
                cur_page = Matlibplot_Widget()
                cur_page.plot(xy_arr=xy_arr, annotation=[str(n) for n in cur_pred], color_arr=cur_pred, special_point=wrong_pred)
                self.pred_widget.addTab(cur_page, name)

        self.update()

    def resizeEvent(self, event):
        '''
        re-plot figure when window resized
        '''
        self._fill_layout()
        return super(Model_Viewer, self).resizeEvent(event)


    def _play_all_examples(self):
        if self.play_widget.text() == 'pause': # pause the play
            self.play_widget.setText('continue')
            self.timer.stop()

        else:  # start playing
            if self.play_widget.text() == 'start':  # fresh start
                self.data_idx = -1
            self.play_widget.setText('pause')
            self.timer.start(5 * 1000)

    def _show_next_example(self):
        self.data_idx += 1
        if self.data_idx < len(self.data_files):
            self._fill_layout()
        else:
            self.data_idx = len(self.data_files) - 1

    def _show_last_example(self):
        self.data_idx -= 1
        if self.data_idx >= 0:
            self._fill_layout()
        else:
            self.data_idx = 0
    
    def _refresh(self):
        self.play_widget.setText('start')
        self.timer.stop()
        self.data_idx = 0
        self._fill_layout()

    def _delete_items(self, layout):
        '''
        recursively delete contents in a layout
        '''
        if layout is not None: 
            while layout.count(): 
                item = layout.takeAt(0) 
                widget = item.widget() 
                if widget is not None: 
                    widget.deleteLater()
                    widget.close() 
                else: 
                    self.deleteItems(item.layout())

if __name__ == '__main__':
    from optparse import OptionParser
    
    parser = OptionParser()
    parser.add_option('--root_dir', dest='root_dir')
    parser.add_option('--dataset_name', dest='dataset_name')
    (options, args) = parser.parse_args()

    root_dir = options.root_dir
    dataset_name = options.dataset_name
    sys.path.append(root_dir)

    app = qtwg.QApplication([])

    # at least one - app exits when the last one closed
    window = Model_Viewer(dataset_name, root_dir)
    window.showMaximized()

    # Start the event loop.
    app.exec_()