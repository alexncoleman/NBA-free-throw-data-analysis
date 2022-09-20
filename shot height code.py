#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun  3 13:07:18 2022

@author: alexcoleman
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit

data = pd.read_excel("Height and Free throws.xlsx")
#create a variable for the free throws
ft = np.array(pd.DataFrame(data, columns= ['FT%'])).flatten()
#variable for height
height = np.array(pd.DataFrame(data, columns=['height'])).flatten()
#creating variable for the position
pos = np.array(pd.DataFrame(data, columns=['pos'])).flatten()


#plotting graph
plt.scatter(height, ft*100)
plt.grid()
plt.ylabel('Free Throw Percentage')
plt.xlabel('Height in m')
plt.title('Free Throw Percentage against Height')
plt.show()

#Now coding the graph to have different colours based on position
iteration = 0
for i in range(len(pos)):
    if pos[iteration] == 'G':
       g = plt.scatter(height[iteration], ft[iteration], color ='yellow', label = 'Guards')
    if pos[iteration] == 'F':
        f = plt.scatter(height[iteration], ft[iteration], color ='orange', label ='Forwards')
    if pos[iteration] == 'C':
        c = plt.scatter(height[iteration], ft[iteration], color ='red', label = 'Centres')
    
    iteration = iteration + 1 

plt.grid()
plt.xlabel('Height in m')
plt.ylabel('Free Throw Percentage')
plt.xlim(1.6,2.25)
plt.title('Free Throw Percentage against Height')
plt.annotate('Stephen Curry', (height[1],ft[1]))


#Now curvefitting to work out best height
def gaus(x, a, x0, sigma,c):
    return a * np.exp(-(x-x0)**2/(2*sigma**2)) + c

initial_a = 1
initial_mu = 1.85
initial_sig = 0.4
initial_c = -0.3
p0= [initial_a, initial_mu, initial_sig, initial_c]


fit1,cov1 = curve_fit(gaus, height, ft, p0)

xplot = np.linspace(1.7,2.25,100)
G = plt.plot(xplot, gaus(xplot, *fit1), color = 'black', label = 'Gaussian fit')
print(fit1)


legend1 = plt.legend((g,f,c),[ 'Guards','Forwards', 'Centres'], loc= 'lower left')
legend2 = plt.legend(['Gaussian Fit'], loc ='upper left')
plt.gca().add_artist(legend1)
plt.show()

print('The optimal height is %.1f m and has a free throw percentage of %.2f' %(p0[1],np.max(ft)))
print('The standard deviation of a normal distribution fit to the free throw percentage vs height is %.2g m' %p0[2])









    