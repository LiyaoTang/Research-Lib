from PyQt5 import QtGui, QtCore, QtWidgets
import pandas as pd
import pyqtgraph as pg
import numpy as np
QVariant = lambda value=None: value


class Widget(QtWidgets.QWidget):

    def __init__(self, parent=None):
        super().__init__(parent)

        v_global_layout = QtWidgets.QVBoxLayout()
        v_global_layout.addWidget(TabDialog())
        v_global_layout.setAlignment(QtCore.Qt.AlignTop)

        self.setLayout(v_global_layout)


class TabDialog(QtWidgets.QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        tab_widget = QtWidgets.QTabWidget()

        # Have the tabs as this dialog's class properties
        self.tab1 = Tab1()
        self.tab2 = Tab2()

        tab_widget.addTab(self.tab1, "1")
        tab_widget.addTab(self.tab2, "2")

        self.tab1.a.sigClicked.connect(self.pointChanged)

        main_layout = QtWidgets.QVBoxLayout()
        main_layout.addWidget(tab_widget)
        self.setLayout(main_layout)

    def pointChanged(self, points):
        self.tab2.points_from_tab1_a = self.tab1.a


class Tab1(QtWidgets.QTabWidget):
    def __init__(self):
        super().__init__()
        layout = QtWidgets.QHBoxLayout()
        self.fig = pg.PlotWidget(name='Example: Selecting scatter points')

        self.plot_area = self.fig.plotItem
        self.a = pg.ScatterPlotItem(pxMode=False)
        spots = []
        for i in range(10):
            for j in range(10):
                spots.append({'pos': (1*i, 1*j), 'size': 1, 'pen': {'color': 'w', 'width': 2},
                              'brush': pg.intColor(i*10+j, 100)})
        self.a.addPoints(spots)

        self.plot_area.addItem(self.a)

        self.a.dataModel = DataFrameModel()
        self.a.dataTable = QtWidgets.QTableView()
        self.a.dataTable.setModel(self.a.dataModel)

        layout.addWidget(self.a.dataTable)
        layout.addWidget(self.fig)
        self.setLayout(layout)

        self.a.array = np.zeros((0, 2))

        def clicked(self, points):
            for p in points:
                p.setPen('b', width=2)
                position = p.viewPos()
                self.array = np.append(self.array, np.array([[position.x(), position.y()]]), axis=0)
            c = range(len(self.array))
            c = list(map(str, c))
            self.dataModel.signalUpdate(self.array, columns=c)
            self.dataModel.printValues() # also: print(self.array)
        self.a.sigClicked.connect(clicked)


class Tab2(QtWidgets.QTabWidget):
    def __init__(self):
        super().__init__()
        layout = QtWidgets.QHBoxLayout()

        self.points_from_tab1_a = []
        ##### Here I want to use Tab1.a and not inherit all the other stuff(layout) #####
        #print("values = ", Tab1.a.array) # a should change when a new point is selected in Tab1
        #####################################
        self.setLayout(layout)


class DataFrameModel(QtCore.QAbstractTableModel):
    """ data model for a DataFrame class """
    def __init__(self):
        super(DataFrameModel, self).__init__()
        self.df = pd.DataFrame()

    def signalUpdate(self, dataIn, columns):
        self.df = pd.DataFrame(dataIn, columns)
        self.layoutChanged.emit()

    def printValues(self):
        print("DataFrame values:\n", self.df.values)

    def values(self):
        return self.df.values

    #------------- table display functions -----------------
    def headerData(self, section, orientation, role=QtCore.Qt.DisplayRole):
        if role != QtCore.Qt.DisplayRole:
            return QVariant()

        if orientation == QtCore.Qt.Horizontal:
            try:
                return self.df.columns.tolist()[section]
            except (IndexError, ):
                return QVariant()
        elif orientation == QtCore.Qt.Vertical:
            try:
                # return self.df.index.tolist()
                return self.df.index.tolist()[section]
            except (IndexError, ):
                return QVariant()

    def data(self, index, role=QtCore.Qt.DisplayRole):
        if role != QtCore.Qt.DisplayRole:
            return QVariant()

        if not index.isValid():
            return QVariant()
        return QVariant(str(self.df.ix[index.row(), index.column()]))

    def rowCount(self, index=QtCore.QModelIndex()):
        return self.df.shape[0]

    def columnCount(self, index=QtCore.QModelIndex()):
        return self.df.shape[1]


if __name__ == '__main__':
    import sys
    app = QtWidgets.QApplication(sys.argv)

    main_window = Widget()
    main_window.setGeometry(100, 100, 640, 480)
    main_window.show()

    sys.exit(app.exec_())