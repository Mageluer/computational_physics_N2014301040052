#!usr/bin/env/ python
#-*- coding=utf-8 -*-

"""
created on Dec,11 2016
"""

from __future__ import division
import pylab as pl
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
from copy import deepcopy

class relaxation:
    def __init__(self, L=12, alpha=1):
        self.L, self.N, self.deltaV, self.convergence, self.alpha, self.V= L, 0, 1, 1e-5, alpha, [[0 for i in range(L+1)] for j in range(L+1)]
        self.Ex,self.Ey,self.E,self.V[int(L/3)],self.V[int(L*2/3)]=deepcopy(self.V),deepcopy(self.V),deepcopy(self.V),[0]*int(L/3)+[2]*(L+1-2*int(L/3))+[0]*int(L/3),[0]*int(L/3)+[-1]*(L+1-2*int(L/3))+[0]*int(L/3)
    def calculate(self):
        while abs(self.deltaV) > self.convergence:
            self.deltaV=0
            for i in range(1,self.L):
                for j in range(1,self.L):
                    if not (i in (int(self.L/3), int(self.L*2/3)) and j in range(int(self.L/3),self.L+1-int(self.L/3))):
                        self.deltaV,self.V[i][j]=self.deltaV+abs((self.V[i-1][j]+self.V[i+1][j]+self.V[i][j-1]+self.V[i][j+1])/4-self.V[i][j]),self.alpha*(self.V[i-1][j]+self.V[i+1][j]+self.V[i][j-1]+self.V[i][j+1])/4+(1-self.alpha)*self.V[i][j]
            self.N+=1
            print self.N,self.deltaV
        for i in range(1,self.L):
            for j in range(1,self.L):
                self.Ex[i][j],self.Ey[i][j]=(self.V[i-1][j]-self.V[i+1][j])/2,(self.V[i][j-1]-self.V[i][j+1])/2
                self.E[i][j]=np.sqrt(self.Ex[i][j]**2+self.Ey[i][j]**2)
        
#######  不如来画图  #########
aa=relaxation(L=90,alpha=1)
aa.calculate()
x=np.linspace(-1,1,aa.L+1)
y=np.linspace(-1,1,aa.L+1)
X,Y=np.meshgrid(x,y)
fig1=pl.figure(1)
pl.ax=Axes3D(fig1)
pl.ax.plot_surface(X, Y, aa.V, rstride=5, cstride=5, cmap='rainbow')
pl.ax.set_xlabel("X")  
pl.ax.set_ylabel("Y")  
pl.ax.set_zlabel("V")  
pl.title('Potential versus x-y plane')
fig2=pl.figure(2)
pl.contourf(X,Y,aa.V)
pl.colorbar()
fig3=pl.figure(3)
surf3=pl.contour(X,Y,aa.V)
pl.clabel(surf3,inline=1, fontsize=10, cmap='jet')
fig4=pl.figure(4)
pl.quiver(X,Y,aa.Ex,aa.Ey,aa.E,cmap='rainbow',linewidth=2,headlength=7)
pl.colorbar()
pl.show()
