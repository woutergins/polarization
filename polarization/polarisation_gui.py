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

        self.optical_tab = self.create_optical_tab()

        self.plottab = self.create_plot_tab()
        self.inputtab = self.create_input_tab()

        self.mainTabs.addTab(self.optical_tab, 'Optical')

        self.optical_tab.addTab(self.plottab, 'Plot')
        self.optical_tab.addTab(self.inputtab, 'Input')

        self.console = console.ConsoleWidget()
        self.central.addWidget(self.console)

    def create_optical_tab(self):
        opticaltab = QtGui.QTabWidget()
        # opticaltablayout = QtGui.QGridLayout(opticaltab)

        return opticaltab

    def create_plot_tab(self):
        plottab = QtGui.QWidget()
        plottablayout = QtGui.QGridLayout(plottab)

        plot_widget = pg.PlotWidget(name='plot1')
        plot_widget.plot()

        plottablayout.addWidget(plot_widget)

        return plottab

    def create_input_tab(self):
        inputtab = QtGui.QWidget()
        inputtablayout = QtGui.QGridLayout(inputtab)

        energy, energy_label = pg.SpinBox(value=30000, suffix='V', siPrefix=True), QtGui.QLabel('Beam energy')

        inputtablayout.addWidget(energy_label, 0, 0)
        inputtablayout.addWidget(energy, 0, 1)

        mass, mass_label = pg.SpinBox(value=30, decimals=9, suffix=' amu'), QtGui.QLabel('Mass of nucleus')

        inputtablayout.addWidget(mass_label, 1, 0)
        inputtablayout.addWidget(mass, 1, 1)

        length, length_label = pg.SpinBox(value=1.5, decimals=9, suffix='m', siPrefix=True), QtGui.QLabel('Interaction length')

        inputtablayout.addWidget(length_label, 2, 0)
        inputtablayout.addWidget(length, 2, 1)

        integration_length, integration_length_label = pg.SpinBox(value=100e-3, decimals=9, suffix='m', siPrefix=True), QtGui.QLabel('Interaction integration')

        inputtablayout.addWidget(integration_length_label, 3, 0)
        inputtablayout.addWidget(integration_length, 3, 1)

        return inputtab

def main():
    app = QtGui.QApplication(sys.argv)
    m = ControllerApp()
    sys.exit(app.exec_())
    # add freeze support


if __name__ == "__main__":
    main()
