#
import numpy as np
import scipy as sp
import scipy.stats as stats
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
from scipy.special import gamma
yourfilepath=["/Users/arlenekemp/Desktop/Poisson Lab/mu4.txt","/Users/arlenekemp/Desktop/Poisson Lab/mu10.txt","/Users/arlenekemp/Desktop/Poisson Lab/mu30.txt","/Users/arlenekemp/Desktop/Poisson Lab/mu100.txt"]

S=len(yourfilepath)

def binwidth(bins):
    return max(bins)/10

#no=[]

#for y in range(S):
fp=yourfilepath[0]
f=np.genfromtxt(fp, int, "#", skip_header=0)
    
binw=binwidth(f)
binss=np.arange(np.floor(min(f)),np.floor(max(f))+1,binw)

histdata,histbin=np.histogram(f,binss,density=False)

histdata1,histbin1=np.histogram(f,binss,density=True)

A=histdata[0]/histdata1[0]

def poissondis(x):
    return (((6.29)**x)/gamma(x+1))*np.exp(-6.29)

xx=np.linspace(2.8,8.8,100)

#tot=np.sum(histdata)
#A=tot*(1.2)/(6.29)

print(A*histdata1)
print(histbin1)
print(A)

b=histdata1[-1]
histdata1=np.delete(histdata1,len(histdata1)-1)
histdata1[-1]=histdata1[-1]+b

histbin1=np.delete(histbin1,len(histbin1)-1)
#histbin1=np.delete(histbin1,0)

print(A*histdata1)
print(histbin1)
print(A)

u_4=np.zeros(len(histdata1))

for i in range(len(histdata1)):
    u_4[i]=np.sqrt(A*histdata1[i]*(1-(A*histdata1[i])/(100)))

print(len(u_4))

cntrhistbin=[1.6,2.8,4,5.2,6.4,7.6,8.8,10]



#plt.bar(histbin1[:-1],A*histdata1)
#plt.scatter(histbin1[:-1],A*histdata1)

#plt.bar(cntrhistbin,A*histdata1)


N=100
meanCount=6.29
#step=A*histdata1[1]-A*histdata1[0]
step=cntrhistbin[1]-cntrhistbin[0]
yUpper=[(1-stats.poisson.cdf(cntrhistbin[-2], meanCount))*N]*2
yLower=[(stats.poisson.cdf(cntrhistbin[1], meanCount))*N]*2
plt.plot([cntrhistbin[-2], cntrhistbin[-1]+step], yUpper,color='r')
plt.plot([cntrhistbin[0]-step, cntrhistbin[1]], yLower,color='r')

plt.ylabel("Number of Trials")
plt.xlabel("Counts per Trial")
#plt.label()

plt.scatter(cntrhistbin,A*histdata1)
plt.errorbar(cntrhistbin,A*histdata1,xerr=None,yerr=u_4,fmt='', marker='_', ls = 'None',capsize=2.3, ecolor = 'b',label='Data points')
plt.title("$\mu \\approx 4$")
plt.tick_params(direction='in',top=True,right=True)

#plt.bar(histbin1,A*histdata1)
#plt.scatter(histbin1,A*histdata1)
#plt.errorbar(histbin1[:-1],A*histdata1,xerr=None,yerr=u_4,fmt='', marker='_', ls = 'None',capsize=2.3, ecolor = 'b')

plt.plot(xx,A*poissondis(xx),color='r',label='Theoretical Poisson Distribution')
plt.grid()
plt.legend(fancybox=True,edgecolor='k')
plt.show()


#print(histbin)
#histdata=[ 7 , 9 ,11 ,13 ,27 ,17,  5,  7]
#histbin=[ 1., 2.2 , 3.4 , 4.6 , 5.8 , 7. ,  8.2,  9.4, 10.6 ]
#newhistbin=np.zeros(8)
#for i in range(0,8):
    #newhistbin[i]=(histbin[i-1]+histbin[i])/2
    
#print(newhistbin)


#plt.scatter(newhistbin,histdata)
##plt.hist(f,binss)

###calculating uncertainty:
#u_4=np.zeros(len(histdata))

#for i in range(len(histdata)):
    #u_4[i]=np.sqrt(histdata[i]*(1-(histdata[i])/(100)))
    

    
#def poissondis(x):
    #return (((6.29)**x)/gamma(x+1))*np.exp(-6.29)

#xx=np.linspace(3,9.4,100)
#A= (histdata[1]-histdata[0])*(100)
#N=100
#meanCount=6.29
#step=histdata[1]-histdata[0]
#yUpper=[(1-stats.poisson.cdf(newhistbin[-2], meanCount))*N]*2
#yLower=[(stats.poisson.cdf(newhistbin[1], meanCount))*N]*2
#plt.plot([newhistbin[-2], newhistbin[-2]+step], yUpper)
#plt.plot([newhistbin[1]-step, newhistbin[1]], yLower)

#plt.errorbar(newhistbin,histdata,xerr=None,yerr=u_4,fmt='', marker='_', ls = 'None',capsize=2.3, ecolor = 'b')
#plt.plot(xx,A*poissondis(xx),color='b')
#plt.draw()
#plt.show()

    
    
    
