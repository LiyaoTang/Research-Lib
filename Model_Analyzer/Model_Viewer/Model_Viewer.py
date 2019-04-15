#!/usr/bin/env python
# coding: utf-8
'''
module: construct QT viewer for model analysis
script: display the viewer UI
'''


import sys
root_dir = '../../'
sys.path.append(root_dir)

if __name__ == '__main__':
    import matplotlib
    matplotlib.use('Qt5Agg')

import os
import gc

import numpy as np
import pandas as pd
import PyQt5.Qt as Qt
import PyQt5.QtGui as qtui
import PyQt5.QtCore as qtc
import PyQt5.QtOpenGL as qgl
import PyQt5.QtWidgets as qtwg
import matplotlib.pyplot as plt

from Model_Analyzer import Project_Loader
from matplotlib.figure import Figure
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas

class Matlibplot_Widget(qtwg.QWidget):
    '''
    qt widget to display a matplotlib image - can be interactive
    '''
    def __init__(self, interactive=True, cmap='rainbow'):
        super(Matlibplot_Widget, self).__init__()
        # plot setting
        self.cmap = plt.get_cmap(cmap)
        self.press = None

        # in cartesian coord
        self.max_x = 6
        self.min_x = -6
        self.max_y = 20  # 60
        self.min_y = -145  # -100

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
        for ann, xy in zip(self.ann_list, self.ann_xylist):
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
        idx_list = np.where(np.logical_and(start < self.ann_xylist, self.ann_xylist < end).all(axis=1))[0]
        for idx in idx_list:
            self.ann_list[idx].set_visible(self.press_btn == 1)

        self.canvas.draw_idle()
        self.press = None

    def _on_pick(self, event):
        if event.mouseevent.inaxes is None or event.mouseevent.inaxes != self.ax:
            return

        for idx in event.ind: # left click - True; else - False
            self.ann_list[idx].set_visible(event.mouseevent.button == 1)
        self.canvas.draw_idle() # update canvas

    def _append_new_annotation(self, xy_list, ann_list):
        for xy, ann in zip(xy_list, ann_list):
            self.ann_list.append(self.ax.annotate(ann, xy=xy))
            self.ann_xylist.append(xy)
            self.ann_list[-1].set_visible(False)
            self.ann_list[-1].draggable()

    def plot(self, points=None, boxes=None, circles=None, color_name=None):
        '''
        plot the given data:
            point_arr - scatter points
            box_list - rectangles
            circ_list - circles
        with annotation, color array and name for different color
        '''
        # if xy_arr is None:
        #     xy_arr = np.random.randn(50).reshape(25,2)
        # if color_arr is None:
        #     color_arr = np.random.randn(25)
        #     # color_arr = np.random.randint(0, 2, size=25)
        #     # color_arr = np.ones(25)

        self.ann_list = []
        self.ann_xylist = []
        self.point_arr = []

        # new figure & annotation
        self.figure.clear()
        self.ax = self.figure.add_subplot(111)
        self._set_coordinate()

        if points:
            xy_arr = points['def']
            self.ax.scatter(xy_arr[:, 0], xy_arr[:, 1], marker='.', c=points['color'], cmap=self.cmap, vmin=0, vmax=1, picker=True)
            if 'annotation' in points and points['annotation']:
                self._append_new_annotation(xy_arr, points['annotation'])

        if boxes:
            for box, c in zip(boxes['def'], boxes['color']):
                rect = matplotlib.patches.Rectangle(box[0], box[1], box[2], fill=False, color=c)
                self.ax.add_patch(rect)
            if 'annotation' in boxes and boxes['annotation']:
                self._append_new_annotation(np.array(boxes['def'])[:,[0,1]], boxes['annotation'])
            
        if circles:
            for circ, c in zip(circles['def'], circles['color']):
                circ = matplotlib.patches.Circle(circ[0], radius=circ[1], fill=False, color=c)
                self.ax.add_patch(circ)
            if 'annotation' in circles and circles['annotation']:
                self._append_new_annotation(np.array(circles['def'])[:,0], circles['annotation'])

        if color_name:
            handles = []
            scalarmap = matplotlib.cm.ScalarMappable(norm=matplotlib.colors.Normalize(vmin=0, vmax=1), cmap=self.cmap)
            for (val, name) in color_name.items():
                # TODO: change label from line to dot
                handles.append(matplotlib.lines.Line2D([], [], color=scalarmap.to_rgba(val), marker='.', label=name))
            self.ax.legend(handles=handles)

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

        self.ax.set_xlim(min_x, max_x, 2)
        self.ax.set_ylim(min_y, max_y, 10)
        
        c = 'lime'
        ls = (0,(9,10,3,10)) # self-defined dash line
        for scaler in [-2.5, -1.5, -0.5, 0.5, 1.5, 2.5]:
            self.ax.vlines(lane_width * scaler, min_y, max_y, colors=c, linestyle=ls)

        self.ax.vlines(0, min_y, max_y, colors='k')
        self.ax.hlines(0, min_x, max_y, colors='k')


class Model_Viewer(qtwg.QMainWindow):
    '''
    view & analyze the model with input, label, its prediction and meta data
    '''
    def __init__(self, dataset_name, dataset_dir, pred_dir, feeder, models=[], log_dir='./Log'):
        super(Model_Viewer, self).__init__()

        self.models = models
        self.feeder = feeder
        print(self.feeder)

        self.log_dir = log_dir
        self.pred_dir = pred_dir
        self.dataset_dir = os.path.join(dataset_dir.split(dataset_name)[0], dataset_name) + '/' # assure to be 'root_dir/Data/dataset'

        self.dataset_name = dataset_name
        self.data_files = sorted([os.path.join(d.split(dataset_name)[-1].strip('/'), f) for d, f in self.feeder.traverser.list_all_file_path()])
        self.data_list = []
        self.file_idx = 0
        self.data_idx = 0
        
        # self.input_type = self.project.input_type
        # self.output_type = self.project.output_type

        if dataset_name == 'corner':
            self._parse_data_dict_to_plot_def = self._parse_corner_data_dict_to_plot_def
        elif dataset_name == 'back':
            self._parse_data_dict_to_plot_def = self._parse_back_data_dict_to_plot_def
        elif dataset_name == 'fusion':
            self._parse_data_dict_to_plot_def = self._parse_back_data_dict_to_plot_def
        else:
            raise ValueError('not supported dataset %s' % dataset_name)

        self._init_ui()

    def _init_ui(self):
        # log setup TODO: define None Value - distinguish from empty
        os.makedirs(self.log_dir, exist_ok=True)
        self.cur_log_path = os.path.join(self.log_dir, self.dataset_name)
        self.tags = ['model err', 'model diff', 'label err']
        if os.path.isfile(self.cur_log_path):
            self.log_file = pd.read_csv(self.cur_log_path, index_col=['file', 'data'],)
            
            log_file_index = self.log_file.index.get_level_values('file')
            new_file = []
            for f in self.data_files:
                if f not in log_file_index:
                    new_file.append(f)
            if new_file:  # append index for new file into logs
                file_df = pd.DataFrame(index=pd.Index([(f, 0) for f in new_file], name=['file', 'data']), columns=self.tags)
                self.log_file = pd.concat([self.log_file, file_df], copy=False).sort_index()
            assert all(self.log_file.columns == self.tags)
        else:
            self.log_file = pd.DataFrame(index=pd.Index([(f, 0) for f in self.data_files], name=['file', 'data']), columns=self.tags)
        self.need_to_save = False

        # window config
        self.setWindowTitle('Model Viewer')
        self.setToolButtonStyle(qtc.Qt.ToolButtonTextUnderIcon)  # icon style
        self.setWindowFlags(self.windowFlags() | qtc.Qt.WindowSystemMenuHint | qtc.Qt.WindowMinMaxButtonsHint)

        # save log - action
        save_log = qtwg.QAction('save log', self)
        save_log.setStatusTip('save log')
        save_log.setShortcuts(Qt.QKeySequence('Ctrl+S'))
        save_log.setShortcutContext(qtc.Qt.ApplicationShortcut)
        save_log.triggered.connect(self._save_log)

        # next/last - action
        next_example = qtwg.QAction('next', self)
        next_example.setStatusTip('next example')
        next_example.setShortcuts(Qt.QKeySequence('Alt+Right'))
        next_example.setShortcutContext(qtc.Qt.ApplicationShortcut)
        next_example.triggered.connect(lambda: self._show_example(self.file_idx, self.data_idx + 1))

        last_example = qtwg.QAction('last_example', self)
        last_example.setStatusTip('last example')
        last_example.setShortcuts(Qt.QKeySequence('Alt+Left'))
        last_example.setShortcutContext(qtc.Qt.ApplicationShortcut)
        last_example.triggered.connect(lambda: self._show_example(self.file_idx, self.data_idx - 1))

        # refresh all - action
        refresh = qtwg.QAction('refresh', self)
        refresh.setStatusTip('refresh to begining')
        refresh.triggered.connect(self._refresh)

        # current & all data files - combo
        combo = qtwg.QComboBox()
        combo.addItems(self.data_files)
        combo.setEditable(False)
        self.path_widget = combo
        combo.currentIndexChanged.connect(lambda i: self._show_example(file_idx=i, data_idx=0))

        # current & all data in the current file - combo
        combo = qtwg.QComboBox()
        combo.setEditable(False)
        self.data_widget = combo # no data before file loaded
        combo.currentIndexChanged.connect(lambda i: self._show_example(file_idx=self.file_idx, data_idx=i))

        # play examples set-up
        play_examples = qtwg.QAction('play', self)
        play_examples.setStatusTip('play through all examples as auto-ppt')
        play_examples.triggered.connect(self._play_all_examples)
        self.timer = qtc.QTimer()
        self.timer.timeout.connect(lambda: self._show_example(file_idx=self.file_idx, data_idx=self.data_idx + 1))
        self.timer_step = 1 * 1000
        self.play_widget = play_examples

        play_setting = qtwg.QLineEdit()
        play_setting.setFixedWidth(40)
        play_setting.setMaxLength(6)
        play_setting.setPlaceholderText('1s')
        play_setting.textChanged.connect(self._set_timer_step)

        # tagging cur example - line
        self.tag_widget = {}
        for cur_t in self.tags:
            line = qtwg.QLineEdit()
            line.setFixedWidth(100)
            line.setClearButtonEnabled(True)
            self.tag_widget[cur_t] = line

        # config toolbar, statusbar
        toolbar = qtwg.QToolBar("My main toolbar")
        toolbar.addAction(last_example)
        toolbar.addAction(next_example)
        toolbar.addWidget(qtwg.QLabel(self.dataset_dir))
        toolbar.addWidget(self.path_widget)
        toolbar.addWidget(self.data_widget)
        toolbar.addAction(play_examples)
        toolbar.addWidget(play_setting)
        toolbar.addAction(refresh)
        toolbar.addSeparator()  # separate function zone
        for cur_t, t_wg in self.tag_widget.items():
            toolbar.addWidget(qtwg.QLabel(cur_t + ':'))
            toolbar.addWidget(t_wg)
        toolbar.addAction(save_log)
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
        label_layout = qtwg.QVBoxLayout()
        feature_names = ' '.join(self.feeder.feature_names)
        label_metaline = qtwg.QLabel(feature_names)
        label_metaline.setWordWrap(True)
        label_metaline.setAlignment(qtc.Qt.AlignCenter)
        label_widget = Matlibplot_Widget('input')
        label_layout.addWidget(label_metaline)
        label_layout.addWidget(label_widget)
        layout_main.addLayout(label_layout)
        self.label_widget = label_widget

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

        self._load_file(0)
        self._fill_layout()

    def _load_file(self, file_idx):
        data_path = os.path.join(self.dataset_dir, self.data_files[file_idx]).split('/')
        data_dir = os.path.join(*data_path[:-1])
        data_name = data_path[-1]
        
        self.data_list = []  # clear existing data
        self.data_widget.clear()
        gc.collect()

        # load new file
        self.data_list = self.feeder.load_with_metadata(data_dir, data_name, self.pred_dir)
        self.data_idx = 0
        self.file_idx = file_idx
        self.data_widget.addItems([str(i) for i in range(len(self.data_list))])

        cur_file = self.data_files[self.file_idx]
        data_idxlist = [*range(len(self.data_list))]
        if list(self.log_file.loc[cur_file].index) != data_idxlist:  # no index created before
            file_df = pd.DataFrame(index=pd.Index([(cur_file, i) for i in data_idxlist], name=['file', 'data']), columns=self.tags)
            # update to the newest one & preserve index order
            reserve_idx = self.log_file.index.get_level_values('file') != cur_file
            self.log_file = pd.concat([self.log_file.loc[reserve_idx], file_df], copy=False).sort_index()
        gc.collect()

    def _show_example(self, file_idx, data_idx):
        # check idx
        if file_idx == self.file_idx and data_idx == self.data_idx:  # replicated update
            return
        elif file_idx >= len(self.data_files) or file_idx < 0 or \
             data_idx >= len(self.data_list) or data_idx < 0:  # invalid idx chk
            return
        
        # new & valid idx
        if self._cleanup_cur_example():  # clean before update - continue only if successful
            
            if self.file_idx != file_idx:  # load new file if needed
                self._load_file(file_idx)  # reset data_idx to 0 after loading
            else:
                self.data_idx = data_idx
            self._fill_layout()  # fill with new content

    def _cleanup_cur_example(self):
        rst = True
        cur_tag_dict = {}
        cur_path = self.data_files[self.file_idx]
        cur_logidx = (cur_path, self.data_idx)
        for cur_t, t_wg in self.tag_widget.items():
            cur_tag_dict[cur_t] = t_wg.text()
        
        # existed tags - warnning box; else overwrite
        tag_existed = not all(self.log_file.loc[cur_logidx].isna())
        existed_tag = self.log_file.loc[cur_logidx].to_dict()  # good under unique index
        if tag_existed and cur_tag_dict != existed_tag:
            warn = qtwg.QMessageBox.warning(self, 'Warning',
                                            'Tags already existed:\n%s\n overwrite with %s ?' %
                                            (str(self.log_file.loc[cur_path].loc[self.data_idx]), str(cur_tag_dict)),
                                            qtwg.QMessageBox.Yes | qtwg.QMessageBox.No | qtwg.QMessageBox.Cancel,
                                            qtwg.QMessageBox.Yes)
            if warn == qtwg.QMessageBox.Yes: # overwrite
                self.log_file.loc[cur_logidx] = cur_tag_dict
                self.need_to_save = True
            elif warn == qtwg.QMessageBox.Cancel: # cancel current action
                rst = False
        elif cur_tag_dict != existed_tag:
            # print(cur_tag_dict)  # print new tags
            self.log_file.loc[cur_logidx] = cur_tag_dict
            self.need_to_save = True

        return rst

    def _parse_corner_data_dict_to_plot_def(self, data_dict):
        input_points = data_dict['input']
        input_xy = data_dict['input_xy']
        labels = data_dict['label'][:, 1]

        label_plotdef = {'points': {'def': input_xy, 'color': labels, 'annotation': [','.join(['%-.2f' % i for i in row]) for row in input_points]},
                         'boxes': None, 'circles': None}
        # color_arr=labels, color_name={0:'other', 1:'car'})
        
        # update prediction
        pred_plotdef = {}
        for name in data_dict['pred'].keys(): # pred per model
            cur_pred = data_dict['pred'][name]

            wrong_pred = ((cur_pred > 0.5) != labels)
            circ_num = wrong_pred.sum()

            pred_plotdef[name] = {'points': {'def': input_xy, 'color': cur_pred, 'annotation': [str(n) for n in cur_pred]},
                                  'circles': {'def': zip(input_xy[np.where(wrong_pred)], [1] * circ_num), 'color': ['r'] * circ_num, 'annotation': None},
                                  'boxes': None}
        return label_plotdef, pred_plotdef

    def _parse_back_data_dict_to_plot_def(self, data_dict):
        # {'input': cur_input, 'input_xy': input_xy, 'label': label_bboxlist, 'pred': pred_bboxlist, 'time_stamp': pred_time}
        # bboxdef {'xy': cart_xy, 'width': w, 'height': h, 'prob': [1], 'elem': elem, 'blockage'}
        input_points = data_dict['input']
        input_xy = data_dict['input_xy']
        if input_xy.size > 0:
            points = {'def': input_xy, 'color': None, 'annotation': [','.join(['%-.2f' % i for i in row]) for row in input_points]}
        else:
            points = None
        
        label_boxes = [[b['xy'], b['width'], b['height']] for b in data_dict['label']]
        label_plotdef = {'boxes': {'def': label_boxes, 'color': ['r'] * len(label_boxes), 'annotation': None},
                         'points': points, 'circles': None}

        pred_boxes = [[b['xy'], b['width'], b['height']] for b in data_dict['pred']]
        pred_plotdef = {'ext_model': {'boxes': {'def': pred_boxes + label_boxes,
                                                'color': ['g'] * len(pred_boxes) + ['r'] * len(label_boxes),
                                                'annotation': None},
                                      'points': points,
                                      'circles': None}}
        return label_plotdef, pred_plotdef

    def _fill_layout(self):
        # update file-data combo list
        if self.path_widget.currentIndex() != self.file_idx:
            self.path_widget.setCurrentIndex(self.file_idx)
        if self.data_widget.currentIndex() != self.data_idx:
            self.data_widget.setCurrentIndex(self.data_idx)

        # update tags from log file
        cur_tag = self.log_file.loc[(self.data_files[self.file_idx],self.data_idx)] # should not contain index cols under unique index
        existed_tag = cur_tag.notna()
        for t_n, t_wg in self.tag_widget.items():
            if existed_tag[t_n].item():
                t_wg.setText(str(cur_tag[t_n]))
            else:
                t_wg.setText(' ')

        # get & parse data dict
        data_dict = self.data_list[self.data_idx]
        label_plotdef, pred_plotdef = self._parse_data_dict_to_plot_def(data_dict)
        
        # update image
        if 'img' in data_dict and data_dict['img'] is not None:
            img = data_dict['img']  # assumed to be rgb
            height, width, channel = img.shape
            bytes_per_line = width * channel
            img_format = qtui.QImage.Format_RGB888
            qimage = qtui.QImage(img, width, height, bytes_per_line, img_format)
            self.img_widget.setPixmap(qtui.QPixmap(qimage))

        # update input/label
        self.label_widget.plot(points=label_plotdef['points'], boxes=label_plotdef['boxes'], circles=label_plotdef['circles'],
                               color_name=self.feeder.class_name)
        
        # update prediction
        for name, idx in zip(pred_plotdef.keys(), range(len(pred_plotdef.keys()))): # pred per model
            cur_pred = pred_plotdef[name]
            cur_page = self.pred_widget.widget(idx)
            # not working: get page by name
            # cur_page = self.pred_widget.findChildren(qtwg.QWidget, name)

            if cur_page:  # existing model
                cur_page.plot(points=cur_pred['points'], boxes=cur_pred['boxes'], circles=cur_pred['circles'],
                              color_name=self.feeder.class_name)
            else:  # new model
                cur_page = Matlibplot_Widget()
                cur_page.plot(points=cur_pred['points'], boxes=cur_pred['boxes'], circles=cur_pred['circles'],
                              color_name=self.feeder.class_name)
                self.pred_widget.addTab(cur_page, name)

        self.update()

    def _play_all_examples(self):
        if self.play_widget.text() == 'pause': # pause the play
            self.play_widget.setText('continue')
            self.timer.stop()

        else:  # start playing
            self.play_widget.setText('pause')
            self.timer.start(self.timer_step)
    def _set_timer_step(self, t):
        try:
            t = float(t)
            assert t >= 0
            self.timer_step = float(t) * 1000  # convert to second
        except:
            print('invalid change %s timer step still %f (s)' % (t, self.timer_step / 1000))

    def _refresh(self):
        self.play_widget.setText('start')
        self.timer.stop()
        self.data_idx = 0
        self._fill_layout()

    def _save_log(self):
        print('log file saved')
        if self._cleanup_cur_example():
            self.log_file.to_csv(self.cur_log_path)
            self.need_to_save = False

    def closeEvent(self, event):
        '''
        Reimplement the closeEvent() event handler: question dialog if unsaved worked found
        '''
        if self.need_to_save:
            reply = qtwg.QMessageBox.question(self, 'Message', 'Are you sure you want to quit? Any unsaved work will be lost.',
                                            qtwg.QMessageBox.Save | qtwg.QMessageBox.Discard | qtwg.QMessageBox.Cancel,
                                            qtwg.QMessageBox.Save)
            if reply == qtwg.QMessageBox.Save:
                self._save_log()
                event.accept()
            elif reply == qtwg.QMessageBox.Discard:
                warn_reply = qtwg.QMessageBox.warning(self, 'Warning', 'All unsaved work will be discard!',
                                                      qtwg.QMessageBox.Yes | qtwg.QMessageBox.No,
                                                      qtwg.QMessageBox.No)
                if warn_reply == qtwg.QMessageBox.Yes:
                    event.accept()
                else:
                    event.ignore()                    
            else:
                event.ignore()

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
    parser.add_option('--dataset_dir', dest='dataset_dir', default='')
    parser.add_option('--pred_dir', dest='pred_dir', default='')

    parser.add_option('--model_group', dest='model_group', default='.*')
    parser.add_option('--model_name', dest='model_name', default='.*')
    parser.add_option('--model_source', dest='model_source', default='loader')
    (options, args) = parser.parse_args()

    sys.path.append(root_dir)

    app = qtwg.QApplication([])

    if options.model_source == 'loader':
        project = Project_Loader(options.dataset_name, root_dir=options.root_dir, verbose=True)
        models, _ = project.load_models(options.model_group, options.model_name)
        feeder = project.feeder
        pred_dir = project.pred_dir
        dataset_dir = project.dataset_dir
        
    else:  # import model from thridparty: model_source = dir_path / py_file / class_name
        path_list = options.model_source.split('/')
        sys.path.append(os.path.join(*path_list[:-2]))  # include dir path in PATH
        loader = __import__(path_list[-1].split('.')[0], fromlist=[''])  # import module
        feeder_class = getattr(loader, path_list[-1])  # get the feeder class
        
        # pred_dir & dataset_dir must provided (used by feeder)
        dataset_dir = options.dataset_dir
        pred_dir = options.pred_dir
        try:
            feeder = feeder_class(dataset_dir)
        except:
            feeder = feeder_class()
            pass

    # at least one - app exits when the last one closed
    window = Model_Viewer(options.dataset_name, options.dataset_dir, pred_dir, feeder, models=[])
    window.showMaximized()

    # Start the event loop.
    app.exec_()