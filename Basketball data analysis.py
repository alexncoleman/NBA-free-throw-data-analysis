#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun  6 16:31:53 2022

@author: alexcoleman
"""

import pandas as pd 
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit
from matplotlib import style

plt.style.use('ggplot')

#read in data
data = pd.read_excel('2016 basketball shooting data.xlsx')
#creating arrays of all the statistics
height = np.array(pd.DataFrame(data, columns=['height_v'])).flatten()
ft = np.array(pd.DataFrame(data, columns=['FT%'])).flatten()
angle = np.array(pd.DataFrame(data, columns=['launch angle'])).flatten()
std_ang = np.array(pd.DataFrame(data, columns=['stdev'])).flatten()

#%%%%%%% ft% against height

plt.scatter(height, ft)
plt.grid()
plt.ylabel('Free Throw Percentage')
plt.xlabel('Height in inches')
plt.title('Free Throw Percentage agaisnt Height')
plt.xlim(70,88)

#fitting a Gaussian
def gaus(x, a, x0, sigma,c):
    return a * np.exp(-(x-x0)**2/(2*sigma**2)) + c



initial_guess1 = [0.1,78,4,0]
fit1,cov1 = curve_fit(gaus, height, ft, p0=initial_guess1)

xdata = np.linspace(70,88,500)
plt.plot(xdata,gaus(xdata,*fit1), color='black')
plt.legend(['Gaussian fit', 'Data'])
plt.show()

print('The optimal height is %.1f and has a free throw percentage of %.2f' %(fit1[1],np.max(ft)))

#%%%%%%% ft% against angle
#color coding
iteration = 0
for i in range(len(height)):
    if height[iteration] < 75:
       small = plt.scatter(angle[iteration], ft[iteration], color ='yellow', label = "Shorter than 6'3")
    if 75 <= height[iteration] < 80:
        medium = plt.scatter(angle[iteration], ft[iteration], color ='orange', label ="Between 6'3 and 6'8")
    if height[iteration] >= 80:
        large = plt.scatter(angle[iteration], ft[iteration], color ='red', label = "6'8 or Taller")
    
    iteration = iteration +1
    

plt.grid()
plt.ylabel('Free Throw Percentage')
plt.xlabel('Launch Angle in degrees')
plt.title('Free Throw Percentage agaisnt Launch Angle')

#fitting a gaussian
initial_guess2 = [0.1,52,4,0.7]
fit2,cov2 = curve_fit(gaus, angle, ft, p0=initial_guess2, sigma=std_ang)

xdata = np.linspace(45,58, 500)
plt.plot(xdata, gaus(xdata, *fit2), color='black')
legend1 = plt.legend(['Gaussian Fit'])
legend2 = plt.legend((small,medium,large), ["Shorter than 6'3", "Between 6'3 and 6'8", "6'8 or Taller"], loc='lower right')
plt.gca().add_artist(legend1)
plt.show()

print('The average optimal launch angle is %.2g with an uncertainty %.1g' %(fit2[1],np.sqrt(cov2[1,0]*cov2[1,2])))
print('The standard deviation of a normal distribution fit to the launch angle is %.2g' %fit2[2])

#%%%%%%%%% angle error vs angle 
#color coding
iteration = 0
for i in range(len(height)):
    if height[iteration] < 75:
       small = plt.scatter(angle[iteration], std_ang[iteration], color ='yellow', label = "Shorter than 6'3")
    if 75 <= height[iteration] < 80:
        medium = plt.scatter(angle[iteration], std_ang[iteration], color ='orange', label ="Between 6'3 and 6'8")
    if height[iteration] >= 80:
        large = plt.scatter(angle[iteration], std_ang[iteration], color ='red', label = "6'8 or Taller")
    
    iteration = iteration +1


plt.grid()
plt.ylabel('Angle Error')
plt.xlabel('Launch Angle in Degrees')
plt.title('Error of Launch Angle vs Launch Angle')

#fitting gaussian 
initial_guess3 = [0.1,52,4,1]
fit3,cov3 = curve_fit(gaus, angle, std_ang, p0=initial_guess3)

#fitting linear
def linear(x,m,c):
    return m*x+c
guesses = [-0.125,2.8]
fit_l,cov_l = curve_fit(linear,angle,std_ang)



xdata= np.linspace(45,58,500)
#plt.plot(xdata, gaus(xdata,*fit3), color='r', linewidth='2')
plt.plot(xdata, linear(xdata,*fit_l), color='black', linewidth='2')
legend1 =plt.legend(['Linear Fit'])
legend2 = plt.legend((small,medium,large), ["Shorter than 6'3", "Between 6'3 and 6'8", "6'8 or Taller"], loc='upper left')
plt.gca().add_artist(legend1)

plt.show()

print('The gradient of the linear fit is %.3f +/- %.3f' %(fit_l[0],np.sqrt(cov_l[0][0])))

#%%%%%% free throw percentage vs uncertainty in angle

#Fitting linear
initial_guess4 = [0.25,1]
fit_l2, cov_l2 = curve_fit(linear, std_ang, ft, p0=initial_guess4)

#color coding data points based on height
iteration = 0
for i in range(len(height)):
    if height[iteration] < 75:
       small = plt.scatter(std_ang[iteration], ft[iteration], color ='yellow', label = "Shorter than 6'3")
    if 75 <= height[iteration] < 80:
        medium = plt.scatter(std_ang[iteration], ft[iteration], color ='orange', label ="Between 6'3 and 6'8")
    if height[iteration] >= 80:
        large = plt.scatter(std_ang[iteration], ft[iteration], color ='red', label = "6'8 or Taller")
    
    iteration = iteration +1
    
    
    
plt.grid()
plt.ylabel('Free Throw Percentage')
plt.xlabel('Uncertainty in Launch angle')
plt.title('Free Throw Percentage against Launch Angle Uncertainty')

xdata = np.linspace(1.25,3.25,500)
plt.plot(xdata, linear(xdata, *fit_l2), color ='black')
legend1 = plt.legend(['Linear fit'], loc= 'lower right')
legend2 = plt.legend((small,medium,large), ["Shorter than 6'3", "Between 6'3 and 6'8", "6'8 or Taller"], loc='upper right')
plt.gca().add_artist(legend1)
plt.show()

print('The gradient of the linear fit is %.2f +/- %.2f' %(fit_l2[0],np.sqrt(cov_l2[0][0])))

#%%%%%%%%%%% std error vs height
#plt.scatter(height, std_ang)

initial_guess5= [0.075, 0]
fit_l3,cov_l3 = curve_fit(linear, height, std_ang, p0=initial_guess5)

iteration = 0
for i in range(len(ft)):
    if ft[iteration] < 0.75:
       bad = plt.scatter(height[iteration], std_ang[iteration], color ='yellow')
    if 0.75 <= ft[iteration] < 0.80:
        medium = plt.scatter(height[iteration], std_ang[iteration], color ='orange')
    if ft[iteration] >= 0.80:
        good = plt.scatter(height[iteration], std_ang[iteration], color ='red')
    
    iteration = iteration +1


plt.ylabel('Standard Deviation in Release Angle')
plt.xlabel('Height (in inches)')
plt.title('Error in Release Angle vs Height')

xdata = np.linspace(69,86,600)
plt.plot(xdata, linear(xdata, *fit_l3), color ='black')
legend1 = plt.legend(['Linear fit'], loc= 'upper right')
legend2 = plt.legend((bad,medium,good), ["ft < 75%", "75% < ft < 80%", "ft > 80%"], loc='upper left')
plt.gca().add_artist(legend1)
plt.show()


print('The gradient of the linear fit is %.3f +/- %.3f' %(fit_l3[0],np.sqrt(cov_l3[0][0])))
