from satlasaddon import RateModelDecay, RateModelPolar
from PyQt4 import QtCore, QtGui
import pyqtgraph as pg
import pyqtgraph.console as console
import sys

class ControllerApp(QtGui.QMainWindow):

    def __init__(self):
        super(ControllerApp, self).__init__()
        self.init_UI()
        self.show()

    def init_UI(self):
        self.central = QtGui.QSplitter()
        widget = QtGui.QWidget()
        self.central.addWidget(widget)
        layout = QtGui.QGridLayout(widget)
        self.setCentralWidget(self.central)

        self.mainTabs = QtGui.QTabWidget()
        layout.addWidget(self.mainTabs)

        self.create_plot_tab()
        self.controltab = QtGui.QWidget()
        self.controltablayout = QtGui.QGridLayout(self.controltab)

        self.plot_widget = pg.PlotWidget(name='plot1')
        self.plot_widget.plot()

        self.controltablayout.addWidget(self.plot_widget)

        self.mainTabs.addTab(self.controltab, 'Plot')

        self.console = console.ConsoleWidget()
        self.central.addWidget(self.console)

    def create_plot_tab(self):
        pass

def main():
    app = QtGui.QApplication(sys.argv)
    m = ControllerApp()
    sys.exit(app.exec_())
    # add freeze support


if __name__ == "__main__":
    main()
