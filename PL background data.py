from matplotlib import rc
import numpy as np
import scipy as sp
import scipy.stats as stats
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
from scipy.special import gamma
yourfilepath=["/Users/arlenekemp/Desktop/Poisson Lab/1710305_BKG.txt"]

s_x1=np.zeros(len(yourfilepath))
s_x2=np.zeros(len(yourfilepath))
u_s=np.zeros(len(yourfilepath))
SS=np.zeros(len(yourfilepath))
varia = np.zeros(len(yourfilepath))
#f=np.zeros(len(yourfilepath))
u1=np.zeros(len(yourfilepath))
u2=np.zeros(len(yourfilepath))
umu=np.zeros(len(yourfilepath))
mu=np.zeros(len(yourfilepath))

#munam=["$\mu\\approx 4$","$\mu\\approx 10$","$\mu\\approx 30$","$\mu\\approx 100$"]


for y in range(len(yourfilepath)):
    fp=yourfilepath[y]
    f=np.genfromtxt(fp, int, "#", skip_header=0)
    
    j=np.linspace(1,100,len(f))
    
    N=len(f)
    
    r=np.zeros(N)
    xi=np.zeros(N)
    ui=np.zeros(N)
    
    for i in range(len(j)):
        k=int(j[i])
        xi[i]=np.sum(f[0:k])
        r[i]=(1/(i+1))*xi[i]
        ui[i]=r[i]/np.sqrt(len(f[0:k]))
        
    # REMEMBER TO COME BACK AND PLOT THESE!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    #plt.errorbar(j,r,yerr=ui,xerr=None,fmt='', marker='_', ls = 'None',capsize=2.3, ecolor = 'b',label="Mean Values")
    #plt.scatter(j,r)
    #plt.legend(fancybox=True,edgecolor='k')
    #plt.grid()    
    #plt.tick_params(direction='in',top=True,right=True)
    #plt.title(munam[y])
    #plt.xlabel("Number of runs")
    #plt.ylabel("Average mean value")
    #xmin,xmax,ymin,ymax=plt.axis([0,100,0,max(r)+max(ui)+1])
    #plt.show()
    mu[y]=r[-1]
    umu[y]=np.sqrt(mu[y]/N)
   
    
    u_s[y]=np.sqrt(((2*N*mu[y]**2)+(N-1)*mu[y])/(N*(N-1)))
    
    #for i in range(N):
    
    
    
    SS[y]=(1/(N-1))*(np.sum(f**2)-np.sum(2*f*mu[y])+N*mu[y]**2)
    
    n , ( xmin , xmax ) , m , v , s , k = stats . describe ( f )
    varia[y]=v
    print("Run",y,"is",v)
    print(mu)
    
    s_x1[y]=SS[y]/mu[y]
    s_x2[y]=varia[y]/mu[y]
    u1[y]=(SS[y]/mu[y])*np.sqrt((u_s[y]/SS[y])**2+(umu[y]/mu[y])**2)
    u2[y]=(varia[y]/mu[y])*np.sqrt((u_s[y]/varia[y])**2+(umu[y]/mu[y])**2)

print(SS)
print(varia)
print(u_s)
print(mu)
print(umu)
print(s_x2)
print(u2)
c=[1,1]
xs=[mu,mu+0.2]
plt.errorbar(mu,s_x2,yerr=u2,xerr=None,fmt='', marker='_', ls = 'None',capsize=2.3, ecolor = 'b',label="Variance over mean value points")
plt.scatter(mu,s_x2)
plt.plot(xs,c,label="Value of 1")
plt.tick_params(direction='in',top=True,right=True)
plt.legend(fancybox=True,edgecolor='k')
plt.xlabel("Mean Value")
plt.ylabel("$s^2/\\bar{x}$")
plt.grid()

plt.show()
