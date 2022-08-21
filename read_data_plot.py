# ------------------------------------------------------
# ---------------------- main.py -----------------------
# ------------------------------------------------------
import time
from turtle import color, delay, done
from typing_extensions import Self
from PyQt5.QtWidgets import*
from PyQt5.uic import loadUi
from matplotlib import pyplot as plt
from matplotlib.backends.backend_qt5agg import (NavigationToolbar2QT as NavigationToolbar)
import numpy as np
import pandas as pd
import random
import sys
import csv
import io
from lib import pointLib as pLib
from lib import coolingLib as cLib
from lib import meltPoolLib as mLib
from keras.models import load_model
import matplotlib.animation as animation
from PyQt5 import QtCore, QtGui, QtWidgets, QtMultimedia
from PyQt5.QtMultimedia import *
from PyQt5.QtMultimediaWidgets import *

class MatplotlibWidget(QMainWindow):
    def __init__(self):
        QMainWindow.__init__(self)
        loadUi("demo.ui",self)
        self.setWindowTitle("AA")
        self.btn1.clicked.connect(self.video_media)
        self.addToolBar(NavigationToolbar(self.MplWidget_2.canvas, self)) 
    

    def setup(self):
        self.MplWidget = self.makeVideoWidget()
        self.mediaPlayer = self.makeMediaPlayer()

    def makeMediaPlayer(self):
        mediaPlayer = QMediaPlayer(self)
        mediaPlayer.setVideoOutput(self.MplWidget)
        return mediaPlayer
    
    def makeVideoWidget(self):
        videoOutput = QVideoWidget(self)
        vbox = QVBoxLayout()
        vbox.addWidget(videoOutput)
        self.MplWidget.setLayout(vbox)
        return videoOutput

    def video_media(self):
        a = QVideoWidget(self)
        vbox = QVBoxLayout()
        vbox.addWidget(a)
        self.video.setLayout(vbox)
        self.player = QMediaPlayer()
        self.player.setVideoOutput(a)
        url = QtCore.QUrl("http://clips.vorwaerts-gmbh.de/VfE_html5.mp4")
        self.player.setMedia(QtMultimedia.QMediaContent(url))
        # self.player.setPosition(0) # to start at the beginning of the video every time
        # self.video.show()
        self.player.play()

    def update_animation(self):
        self.ani = animation.FuncAnimation(self.MplWidget, self.update_axes, 
        self.update_graph, interval=10, repeat=False)
        self.MplWidget.canvas.draw()

    def update_graph(self):
        a = []
        b = []
        c = []
        t = []
        for i in range(1920):
            df = pd.read_csv("data.csv")
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
        self.MplWidget.canvas.axes.clear()
        self.MplWidget.canvas.axes.plot(t, a)
        self.MplWidget.canvas.axes.plot(t, b)
        self.MplWidget.canvas.axes.plot(t, c)
        self.MplWidget.canvas.axes.legend(('meltWidth', 'meltDepth', 'meltArea'),loc='upper right')
        self.MplWidget.canvas.axes.set_title('Mel Prediction Processing')            
        self.MplWidget.canvas.draw()

    # def demo(self):
    #     df = pd.read_csv("data.csv")
    #     a = []
    #     b = []
    #     c = []
    #     e = []
    #     # for i in range(1920):
    #     #     a1 = df._get_value(i, 'meltWidth')
    #     #     b1 = df._get_value(i, 'meltDepth')
    #     #     c1 = df._get_value(i, 'meltArea')
    #     #     a.append(float(a1[1:(len(a1)-1)]))
    #     #     b.append(float(b1[1:(len(b1)-1)]))
    #     #     c.append(float(c1[1:(len(c1)-1)]))
    #     #     e.append(i)

    #     size = 100
    #     x_vec = np.linspace(0,1,size+1)[0:-1]
    #     y_vec1 = np.random.randn(len(x_vec))
    #     y_vec2 = np.random.randn(len(x_vec))
    #     y_vec3 = np.random.randn(len(x_vec))
    #     line1 = []
    #     line2 = []
    #     line3 = []

    #     for i in range(1920):
    #         a1 = df._get_value(i, 'meltWidth')
    #         b1 = df._get_value(i, 'meltDepth')
    #         c1 = df._get_value(i, 'meltArea')
    #         a.append(float(a1[1:(len(a1)-1)]))
    #         b.append(float(b1[1:(len(b1)-1)]))
    #         c.append(float(c1[1:(len(c1)-1)]))
    #         e.append(i)

    #         y_vec1[-1] = float(a1[1:(len(a1)-1)])
    #         line1 = live_plotter(x_vec,y_vec1,line1)
    #         y_vec1 = np.append(y_vec1[1:],0.0)
            
            
    #         y_vec2[-1] = (float(b1[1:(len(b1)-1)]))
    #         line2 = live_plotter(x_vec,y_vec2, line2)
    #         y_vec2 = np.append(y_vec2[1:],0.0)

    #         y_vec3[-1] = (float(c1[1:(len(c1)-1)]))
    #         line3 = live_plotter(x_vec,y_vec3, line3)
    #         y_vec3 = np.append(y_vec3[1:],0.0)

    #         # self.MplWidget_2.canvas.axes.clear()
    #         self.MplWidget_2.canvas.axes.plot(e, a, color='r')
    #         self.MplWidget_2.canvas.axes.plot(e, b, color='g')
    #         self.MplWidget_2.canvas.axes.plot(e, c, color='y')
    #         self.MplWidget_2.canvas.axes.legend(('meltWidth', 'meltDepth', 'meltArea'),loc='upper right')
    #         self.MplWidget_2.canvas.axes.set_title('Mel Prediction Processing')            
    #         self.MplWidget_2.canvas.draw() 
    #         self.MplWidget_2.canvas.axes.pause(0.1)  

    #     self.MplWidget.canvas.axes.clear()
    #     self.MplWidget.canvas.axes.plot(a)
    #     self.MplWidget.canvas.axes.plot(b)
    #     self.MplWidget.canvas.axes.plot(c)
    #     self.MplWidget.canvas.axes.legend(('meltWidth', 'meltDepth', 'meltArea'),loc='upper right')
    #     self.MplWidget.canvas.axes.set_title('Mel Prediction Processing')
    #     self.MplWidget.canvas.draw()  


app = QApplication([])
window = MatplotlibWidget()
window.show()
app.exec_()
