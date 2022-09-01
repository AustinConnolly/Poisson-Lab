

import numpy as np
import scipy as sp
import scipy.stats as stats
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
from scipy.special import gamma
yourfilepath=["/Users/arlenekemp/Desktop/Poisson Lab/mu4.txt","/Users/arlenekemp/Desktop/Poisson Lab/zaydmu10.txt","/Users/arlenekemp/Desktop/Poisson Lab/mu30.txt","/Users/arlenekemp/Desktop/Poisson Lab/mu100.txt"]

S=len(yourfilepath)

def binwidth(bins):
    return max(bins)/10


#for y in range(S):
fp=yourfilepath[2]
f=np.genfromtxt(fp, int, "#", skip_header=0)
    
#binw=binwidth(f)
#binw=
#binss=np.arange(np.floor(min(f)),np.floor(max(f))+1,binw)

histdata,histbin=np.histogram(f,bins=10,density=False)

histdata1,histbin1=np.histogram(f,bins=10,density=True)

A=histdata[0]/histdata1[0]

def poissondis(x):
    return (((31.13)**x)/gamma(x+1))*np.exp(-31.13)

xx=np.linspace(25,39.4,100)

#tot=np.sum(histdata)
#A=tot*(1.2)/(6.29)

print(A*histdata1)
print(histbin1)
print(A)


#b=histdata1[-1]
#histdata1=np.delete(histdata1,len(histdata1)-1)
#histdata1[-1]=histdata1[-1]+b

#b=histdata1[-1]
#histdata1=np.delete(histdata1,len(histdata1)-1)
#histdata1[-1]=histdata1[-1]+b



histdata1[1]=histdata1[1]+histdata1[0]
histdata1=np.delete(histdata1,0)

#histbin1=np.delete(histbin1,len(histbin1)-1)
#histbin1=np.delete(histbin1,len(histbin1)-1)
histbin1=np.delete(histbin1,0)
##-------------------
print(A*histdata1)
print(histbin1)
print(A)


cntrhistbin=np.zeros(len(histdata1))
h=0
for i in range(len(histdata1)):
    cntrhistbin[h]=(histbin1[i]+histbin1[i+1])/2
    h=h+1

u_4=np.zeros(len(histdata1))

for i in range(len(histdata1)):
    u_4[i]=np.sqrt(A*histdata1[i]*(1-(A*histdata1[i])/(100)))

print(cntrhistbin)
print(len(cntrhistbin))


N=100
meanCount=31.13

step=cntrhistbin[1]-cntrhistbin[0]
yUpper=[(1-stats.poisson.cdf(cntrhistbin[-2], meanCount))*N]*2
yLower=[(stats.poisson.cdf(cntrhistbin[1], meanCount))*N]*2
plt.plot([cntrhistbin[-2], cntrhistbin[-1]+step], yUpper,color='r')
plt.plot([cntrhistbin[0]-step, cntrhistbin[1]], yLower,color='r')



plt.ylabel("Number of Trials")
plt.xlabel("Counts per Trial")
plt.title("$\mu \\approx 30$")

plt.scatter(cntrhistbin,A*histdata1)
plt.errorbar(cntrhistbin,A*histdata1,xerr=None,yerr=u_4,fmt='', marker='_', ls = 'None',capsize=2.3, ecolor = 'b',label='Data points')
plt.plot(xx,A*poissondis(xx),color='r',label='Theoretical Poisson Distribution')

plt.tick_params(direction='in',top=True,right=True)
plt.legend(fancybox=True,edgecolor='k')

plt.grid()

plt.show()