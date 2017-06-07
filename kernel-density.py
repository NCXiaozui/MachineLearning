# -*- coding: utf-8 -*-
'''
 the code for kernel density estimate
'''

import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import KernelDensity

np.random.seed(1)
N = 20
#np.concatenate将所有的array联合在一起。在这里我们用了两个高斯分布，一个是标准正态分布，一个是均值为5,方差为1的正态分布。
#[:,np.newaxis]是一种增加维度的方式，np.newaxis其实是一个None
x = np.concatenate((np.random.normal(0,1,int(0.3*N)),np.random.normal(5,1,int(0.7*N))))[:,np.newaxis]

print x[:,0]

#np.linspace是生成一串数字
x_plot = np.linspace(-5,10,1000)[:,np.newaxis]
bins = np.linspace(-5,10,10)
print bins

fig,ax = plt.subplots(2,2,sharex=True,sharey=True)
fig.subplots_adjust(hspace=0.05,wspace=0.05)

#Histogram
#x[:,0]减掉一个维度
ax[0,0].hist(x[:,0],bins=bins,fc='#AAAAFF',normed=True)
ax[0,0].text(-3.5,0.31,'Histogram')

#Histogram,bins shifted
ax[0,1].hist(x[:,0],bins=bins + 0.75,fc='#AAAAFF',normed=True)
ax[0,1].text(-3.5,0.31,'Hisstogram, bins shifted')

#kernel density estimate of tophat
kde = KernelDensity(kernel = 'tophat',bandwidth = 0.75).fit(x)
log_dens = kde.score_samples(x_plot)

ax[1,0].fill(x_plot[:,0],np.exp(log_dens),fc='#AAAAFF')
ax[1,0].text(-3.5,0.31,'Tophat Kernel Density')

#kernel density estimate of Gaussian
kde = KernelDensity(kernel = 'gaussian',bandwidth= 0.75).fit(x)
log_dens = kde.score_samples(x_plot)

ax[1,1].fill(x_plot[:,0],np.exp(log_dens),fc='#AAAAFF')
ax[1,1].text(-3.5,0.31,'Gaussian Kernel Density')

for axi in ax.ravel():
    axi.plot(x[:,0],np.zeros(x.shape[0]),'+k')
    axi.set_xlim(-4,9)
    axi.set_ylim(-0.02,0.34)

for axi in ax[:,0]:
    axi.set_ylabel('Normalized Density')

for axi in ax[1,:]:
    axi.set_xlabel('x')
plt.show()
