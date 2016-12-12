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
        self.V[int(L/3)],self.V[int(L*2/3)]=[0]*int(L/3)+[1]*7+[-1]*7+[1]*7+[0]*int(L/3),[0]*int(L/3)+[-1]*7+[1]*7+[-1]*7+[0]*int(L/3)
        #self.V[int(L/3)],self.V[int(L*2/3)]=[0]*int(L/3)+[1]*(L+1-2*int(L/3))+[0]*int(L/3),[0]*int(L/3)+[-1]*(L+1-2*int(L/3))+[0]*int(L/3)
    def calculate(self):
        while abs(self.deltaV) > self.convergence:
            self.deltaV,self.N=0,self.N+1
            for i in range(1,self.L):
                for j in range(1,self.L):
                    if not (i in (int(self.L/3), int(self.L*2/3)) and j in range(int(self.L/3),self.L+1-int(self.L/3))):
                        self.deltaV,self.V[i][j]=self.deltaV+abs((self.V[i-1][j]+self.V[i+1][j]+self.V[i][j-1]+self.V[i][j+1])/4-self.V[i][j]),self.alpha*(self.V[i-1][j]+self.V[i+1][j]+self.V[i][j-1]+self.V[i][j+1])/4+(1-self.alpha)*self.V[i][j]
            print self.N,self.deltaV
        
#######  不如来画图  #########
aa=relaxation(L=60,alpha=1.5)
aa.calculate()
x=np.linspace(-1,1,aa.L+1)
y=np.linspace(-1,1,aa.L+1)
X,Y=np.meshgrid(x,y)
Ex,Ey,E=deepcopy(aa.V),deepcopy(aa.V),deepcopy(aa.V)
for i in range(1,aa.L):
    for j in range(1,aa.L):
        Ex[i][j],Ey[i][j]=(aa.V[i-1][j]-aa.V[i+1][j])/2,(aa.V[i][j-1]-aa.V[i][j+1])/2
        E[i][j]=np.sqrt(Ex[i][j]**2+Ey[i][j]**2)
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
pl.quiver(X,Y,Ex,Ey,E,cmap='rainbow',linewidth=2,headlength=7)
pl.colorbar()
"""
for L,style in zip([30,36,42],['r-s','g-o','b-d']):
    rap,rN=[],[]
    for alpha in sorted(np.append(np.linspace(1.5,1.9,20),2/(1+np.pi/L))):
        aa=relaxation(L=L,alpha=alpha)
        aa.calculate()
        rap.append(alpha)
        rN.append(aa.N)
    pl.plot(rap,rN,style,label=r'L=%d' % L)
pl.xlabel(r"$\alpha$")
pl.ylabel(r'$N$')
pl.legend(loc='best')
pl.title(r'convergence speed versus $\alpha$')
"""
pl.show()
