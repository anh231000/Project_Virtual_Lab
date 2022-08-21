from re import X
from tkinter import Y
from turtle import delay, done
from PyQt5.QtWidgets import*
from PyQt5.uic import loadUi

from PyQt5 import QtCore, QtGui, QtWidgets, QtMultimedia

from matplotlib.backends.backend_qt5agg import (NavigationToolbar2QT as NavigationToolbar)

import numpy as np
import pandas as pd
import random
import csv
import sys

from pyparsing import col
from traitlets import Int
from lib import pointLib as pLib
from lib import coolingLib as cLib
from lib import meltPoolLib as mLib
from keras.models import load_model
import matplotlib.animation as animation
from PyQt5.QtMultimedia import *
from PyQt5.QtMultimediaWidgets import *



class MatplotlibWidget(QMainWindow):
    def __init__(self):
        super(MatplotlibWidget, self).__init__()
        loadUi("demo.ui",self)
        self.btn1.clicked.connect(self.Cooling_Rate_Prediction)
        self.btn2.clicked.connect(self.Melt_Prediction)
        self.btn3.clicked.connect(self.Point_Prediction)
        self.btn4.clicked.connect(self.Predict_All)
        self.btn5.clicked.connect(self.Cancel)
        self.addToolBar(NavigationToolbar(self.MplWidget_2.canvas, self))
        a = QVideoWidget(self)
        vbox = QVBoxLayout()
        vbox.addWidget(a)
        self.video.setLayout(vbox)
        self.player = QMediaPlayer()
        self.player.setVideoOutput(a)

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
        print('done')
        self.video_media()


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

        with open('Melt_Prediction.csv', 'w', encoding='UTF8', newline='') as f:
            writer = csv.writer(f)
            header = ['meltWidth', 'meltDepth', 'meltArea']
            writer.writerow(header)
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
                data = [meltWidth[my_dt], meltDepth[my_dt], meltArea[my_dt]]
                writer.writerow(data)
        
        self.ani = animation.FuncAnimation(self.MplWidget_2, self.update_axes, 
        self.update_graph, interval= 0.1, repeat=False)
        self.MplWidget_2.canvas.draw()
        print('done')
        self.video_media()
    
    
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

        with open('Point_Prediction.csv', 'w', encoding='UTF8', newline='') as f:
            writer = csv.writer(f)
            header = ['point_x', 'point_y']
            writer.writerow(header)
            for i in range(len(point_x)):
                data = [point_x[i],point_y[i]]
                writer.writerow(data)

        self.ani = animation.FuncAnimation(self.MplWidget_2, self.update_axesa, 
        self.update_grapha, interval= 0.1, repeat=False)
        self.MplWidget_2.canvas.draw()
        print('done')
        self.video_media()


    def Predict_All(self):
        self.MplWidget_2.canvas.axes.clear()
        self.MplWidget_2.canvas.draw() 
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

        with open('Melt_Prediction.csv', 'w', encoding='UTF8', newline='') as f:
            writer = csv.writer(f)
            header = ['meltWidth', 'meltDepth', 'meltArea']
            writer.writerow(header)
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
                data = [meltWidth[my_dt], meltDepth[my_dt], meltArea[my_dt]]
                writer.writerow(data)

        self.ani = animation.FuncAnimation(self.MplWidget_2, self.update_axes, 
        self.update_graph, interval = 0.1, repeat=False)
        self.MplWidget_2.canvas.draw()

        point_x = time_orig
        point_y = res_[:,pointID]
        coolingRate =  cLib.cooling_rate(time_orig, point_y)
        self.output1.display(coolingRate)
        print('done')
        self.video_media()


    def video_media(self):
        url = QtCore.QUrl("http://clips.vorwaerts-gmbh.de/VfE_html5.mp4")
        self.player.setMedia(QtMultimedia.QMediaContent(url))
        self.player.play()

    def video_mediab(self):
        a = QVideoWidget(self)
        vbox = QVBoxLayout()
        vbox.addWidget(a)
        self.video.setLayout(vbox)
        self.player = QMediaPlayer()
        self.player.setVideoOutput(a)
        url = QtCore.QUrl("http://clips.vorwaerts-gmbh.de/VfE_html5.mp4")
        self.player.setMedia(QtMultimedia.QMediaContent(url))
        self.player.play()

    def Cancel(self):
        sys.exit(-1)

    def update_grapha(self):
        X = []
        Y = []
        for i in range(1978):
            df = pd.read_csv("Point_Prediction.csv")
            a1 = df._get_value(i, 'point_x')
            b1 = df._get_value(i, 'point_y')
            X.append(float(a1[1:(len(a1)-1)]))
            Y.append(b1)
            yield X, Y

    def update_axesa(self, update):
        X, Y = update[0], update[1]
        self.MplWidget_2.canvas.axes.clear()
        self.MplWidget_2.canvas.axes.plot(X, Y)
        self.MplWidget_2.canvas.axes.legend(('Point_Prediction'),loc='upper right')
        self.MplWidget_2.canvas.axes.set_title('Point Prediction')
        self.MplWidget_2.canvas.draw()


    def update_graph(self):
        a = []
        b = []
        c = []
        t = []
        for i in range(1978):
            df = pd.read_csv("Melt_Prediction.csv")
            a1 = df._get_value(i, 'meltWidth')
            b1 = df._get_value(i, 'meltDepth')
            c1 = df._get_value(i, 'meltArea')
            a.append(float(a1[1:(len(a1)-1)]))
            b.append(float(b1[1:(len(b1)-1)]))
            c.append(float(c1[1:(len(c1)-1)]))
            t.append(i)
            yield t, a, b, c  

    def update_axes(self, update):
        t, a, b, c = update[0], update[1], update[2], update[3]
        self.MplWidget_2.canvas.axes.clear()
        self.MplWidget_2.canvas.axes.plot(t, a)
        self.MplWidget_2.canvas.axes.plot(t, b)
        self.MplWidget_2.canvas.axes.plot(t, c)
        self.MplWidget_2.canvas.axes.legend(('meltWidth', 'meltDepth', 'meltArea'),loc='upper right')
        self.MplWidget_2.canvas.axes.set_title('Mel Prediction Processing')            
        self.MplWidget_2.canvas.draw()


# main
app = QApplication([])
window = MatplotlibWidget()
window.show()
app.exec_()

