# ------------------------------------------------------
# ---------------------- main.py -----------------------
# ------------------------------------------------------
from turtle import delay, done
from PyQt5.QtWidgets import*
from PyQt5.uic import loadUi

from matplotlib.backends.backend_qt5agg import (NavigationToolbar2QT as NavigationToolbar)

import numpy as np
import pandas as pd
import random
import csv
import sys
from lib import pointLib as pLib
from lib import coolingLib as cLib
from lib import meltPoolLib as mLib
from keras.models import load_model

class MatplotlibWidget(QMainWindow):
    def __init__(self):
        QMainWindow.__init__(self)
        loadUi("demo.ui",self)
        self.setWindowTitle("Demo")
        self.btn1.clicked.connect(self.Cooling_Rate_Prediction)
        self.btn2.clicked.connect(self.Melt_Prediction)
        self.btn3.clicked.connect(self.Point_Prediction)
        self.btn4.clicked.connect(self.Predict_All)
        self.btn5.clicked.connect(self.Cancel)
        self.addToolBar(NavigationToolbar(self.MplWidget.canvas, self))

    def Cooling_Rate_Prediction(self):
            # input
            q = float(self.input1.text())
            T_s = float(self.input2.text())
            T_c = float(self.input3.text())
            pointID = 1500
            # load data
            model = load_model("C:/Users/TTA/Desktop/Project_Virtual_Lab/DED/data/model.h5")
            time_orig = pd.read_csv('C:/Users/TTA/Desktop/Project_Virtual_Lab/DED/data/Timestep.csv').values
            df = pd.read_csv('C:/Users/TTA/Desktop/Project_Virtual_Lab/DED/data/DED_data.csv', header=None, index_col=None,  dtype=np.float16)
            ref_ = np.loadtxt('C:/Users/TTA/Desktop/Project_Virtual_Lab/DED/data/ref.txt')
            # get scaler
            scaler_X, scaler_y = pLib.get_scaler(df)
            # create inp for prediction
            inp_ = pLib.create_inp(q, T_s, T_c, ref_, scaler_X)
            # predict the data
            print('Predict ...')
            pred_ = pLib.pred(model, inp_, scaler_y)
            # make result in order for all mesh
            res_ = pLib.makeRes(pred_) # Should save this one in db if possible
            # plot if needed
            point_x = time_orig
            point_y = res_[:,pointID]
            coolingRate =  cLib.cooling_rate(time_orig, point_y)
            self.output1.display(coolingRate)


    def Melt_Prediction(self):
        # Input
        q = float(self.input1.text())
        T_s = float(self.input2.text())
        T_c = float(self.input3.text())
        # Load neccessary data
        print('Loading ...')
        model = load_model("data/model.h5")
        time_orig = pd.read_csv('data/Timestep.csv').values
        df = pd.read_csv('data/DED_data.csv', header=None, index_col=None,  dtype=np.float16)
        ref_ = np.loadtxt('data/ref.txt')
        mshfilename= 'data/simu2_gid.msh'
        [coord, Element, row, col, mesh]=mLib.read_msh_file(mshfilename)
        T_melt = 1676.15
        # get scaler
        scaler_X, scaler_y = pLib.get_scaler(df)
        # create inp for prediction
        inp_ = pLib.create_inp(q, T_s, T_c, ref_, scaler_X)
        # predict the data
        print('Predict ...')
        pred_ = pLib.pred(model, inp_, scaler_y)
        # make result in order for all mesh
        res_ = pLib.makeRes(pred_) # Should save this one in db if possible
        meltWidth = np.zeros((1978,1))
        meltDepth = np.zeros((1978,1))
        meltArea = np.zeros((1978,1))
        for my_dt in range(1978):
            id = 0
            for dt in range(2520):
                nodeID = dt
                nodeT = float(res_[my_dt][dt])
                coord[nodeID][2] = nodeT
            meltWidth[my_dt] = mLib.melt_width(T_melt, coord, row)
            meltDepth[my_dt] = mLib.melt_depth(T_melt, coord, col)
            meltArea[my_dt] = mLib.melt_size(T_melt, coord, Element, mesh)
        
        self.MplWidget.canvas.axes.clear()
        self.MplWidget.canvas.axes.plot(meltWidth)
        self.MplWidget.canvas.axes.plot(meltDepth)
        self.MplWidget.canvas.axes.plot(meltArea)
        self.MplWidget.canvas.axes.legend(('meltWidth', 'meltDepth', 'meltArea'),loc='upper right')
        self.MplWidget.canvas.axes.set_title('Mel Prediction')
        self.MplWidget.canvas.draw()
    
    def Point_Prediction(self):
        # Input
        q = float(self.input1.text())
        T_s = float(self.input2.text())
        T_c = float(self.input3.text())
        pointID = 1500 # can modify to any point in FE - Choose by user
        # Load neccessary data
        print('Loading ...')
        model = load_model("data/model.h5")
        time_orig = pd.read_csv('data/Timestep.csv').values
        df = pd.read_csv('data/DED_data.csv', header=None, index_col=None,  dtype=np.float16)
        ref_ = np.loadtxt('data/ref.txt')
        # get scaler
        scaler_X, scaler_y = pLib.get_scaler(df)
        # create inp for prediction
        inp_ = pLib.create_inp(q, T_s, T_c, ref_, scaler_X)
        # predict the data
        print('Predict ...')
        pred_ = pLib.pred(model, inp_, scaler_y)
        # make result in order for all mesh
        res_ = pLib.makeRes(pred_) # Should save this one in db if possible
        # plot if needed
        point_x = time_orig
        point_y = res_[:,pointID]
        self.MplWidget.canvas.axes.clear()
        self.MplWidget.canvas.axes.plot(point_x, point_y)
        self.MplWidget.canvas.axes.legend(('Point_Prediction'),loc='upper right')
        self.MplWidget.canvas.axes.set_title('Point Prediction')
        self.MplWidget.canvas.draw()


    def Predict_All(self):
        # Input
        q = float(self.input1.text())
        T_s = float(self.input2.text())
        T_c = float(self.input3.text())
        pointID = 1500 # can modify to any point in FE - Choose by user
        # Load neccessary data
        print('Loading ...')
        model = load_model("data/model.h5")
        time_orig = pd.read_csv('data/Timestep.csv').values
        df = pd.read_csv('data/DED_data.csv', header=None, index_col=None,  dtype=np.float16)
        ref_ = np.loadtxt('data/ref.txt')
        mshfilename= 'data/simu2_gid.msh'
        [coord, Element, row, col, mesh]=mLib.read_msh_file(mshfilename)
        T_melt = 1676.15
        # get scaler
        scaler_X, scaler_y = pLib.get_scaler(df)

        # create inp for prediction
        inp_ = pLib.create_inp(q, T_s, T_c, ref_, scaler_X)

        # predict the data
        print('Predict ...')
        pred_ = pLib.pred(model, inp_, scaler_y)

        # make result in order for all mesh
        res_ = pLib.makeRes(pred_) # Should save this one in db if possible, so that user can choose another point in pointID to continue the calculation

        # Cal melt width, depth, and area
        meltWidth = np.zeros((1978,1))
        meltDepth = np.zeros((1978,1))
        meltArea = np.zeros((1978,1))

        for my_dt in range(1978):
            id = 0
            print(my_dt)
            for dt in range(2520):
                nodeID = dt
                nodeT = float(res_[my_dt][dt])
                coord[nodeID][2] = nodeT
            meltWidth[my_dt] = mLib.melt_width(T_melt, coord, row)
            meltDepth[my_dt] = mLib.melt_depth(T_melt, coord, col)
            meltArea[my_dt] = mLib.melt_size(T_melt, coord, Element, mesh)

        print('done')
        point_x = time_orig
        point_y = res_[:,pointID]
        coolingRate =  cLib.cooling_rate(time_orig, point_y)

        self.output1.display(coolingRate)
        
        self.MplWidget.canvas.axes.clear()
        self.MplWidget.canvas.axes.plot(meltWidth)
        self.MplWidget.canvas.axes.plot(meltDepth)
        self.MplWidget.canvas.axes.plot(meltArea)
        self.MplWidget.canvas.axes.legend(('meltWidth', 'meltDepth', 'meltArea'),loc='upper right')
        self.MplWidget.canvas.axes.set_title('Mel Prediction')
        self.MplWidget.canvas.draw()

        plot_prosessing(self, meltWidth, meltDepth, meltArea)

        # self.MplWidget_2.canvas.axes.clear()
        # self.MplWidget_2.canvas.axes.plot(meltWidth)
        # self.MplWidget_2.canvas.axes.plot(meltDepth)
        # self.MplWidget_2.canvas.axes.plot(meltArea)
        # self.MplWidget_2.canvas.draw()           
        # self.MplWidget_2.canvas.axes.legend(('meltWidth', 'meltDepth', 'meltArea'),loc='upper right')
        # self.MplWidget_2.canvas.axes.set_title('Mel Prediction Proceesing')

    def Cancel(self):
        sys.exit(-1) 


def plot_prosessing(self, meltWidth, meltDepth, meltArea):
    a = []
    b = []
    c = []
    for my_dt in range(1978):
        print(my_dt)
        a.append(meltWidth[my_dt])
        b.append(meltDepth[my_dt])
        c.append(meltArea[my_dt]) 
        self.MplWidget_2.canvas.axes.clear()
        self.MplWidget_2.canvas.axes.plot(meltWidth)
        self.MplWidget_2.canvas.axes.plot(meltDepth)
        self.MplWidget_2.canvas.axes.plot(meltArea)
        self.MplWidget_2.canvas.draw()           
        self.MplWidget_2.canvas.axes.legend(('meltWidth', 'meltDepth', 'meltArea'),loc='upper right')
        self.MplWidget_2.canvas.axes.set_title('Mel Prediction Proceesing')


app = QApplication([])
window = MatplotlibWidget()
window.show()
app.exec_()
