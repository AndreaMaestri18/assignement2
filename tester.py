import scipy.integrate as spi
import numpy as np
import pylab as pl

n=4 # number of subpopulations
N = 1000
beta=1.0*np.ones(n)
gamma=0.1*np.ones(n)
X0=0.1*np.ones(n)
Y0=0.0*np.ones(n); Y0[0]=0.0001
print(f"Y0: {Y0}")
rho=0.001*np.ones((n,n)); rho=rho-np.diag(np.diag(rho))
print("rho\n", f"{rho}")
ND=MaxTime=2910.0
TS=1.0

INPUT=np.hstack((X0,Y0))#---------------------> V
# print("stack\n", f"{INPUT}")

def diff_eqs(INP,t):  
	'''The main set of equations'''
	Y=np.zeros((2*n))
	V = INP   
	for i in range(n):
        lmbda_i = 0
		Y[i] =  - beta[i]*V[i]*V[n+i] # This is dS, V[i] = S, V[n + i] = I
		Y[n+i] = beta[i]*V[i]*V[n+i] - gamma[i]*V[n+i] # This is dI
		for j in range(n):
			Y[i]+=rho[i][j]*V[j] - rho[j][i]*V[i] # X
			Y[n+i]+=rho[i][j]*V[n+j] - rho[j][i]*V[n+i] # Y
            lmbdai += 
	return Y   # For odeint

t_start = 0.0; t_end = ND; t_inc = TS
t_range = np.arange(t_start, t_end+t_inc, t_inc)
RES = spi.odeint(diff_eqs,INPUT,t_range)

#Ploting
pl.subplot(211)
for i in range(n):
	pl.plot(t_range/365.0, RES[:,i], color=(0.0,0.3,i/5))
pl.xlabel('Time (Years)')
pl.ylabel('Susceptibles')
pl.subplot(212)
for i in range(n):
	pl.plot(t_range/365.0, RES[:,i+n], color=(0.7,0.0,i/5))
pl.ylabel('Infected')
pl.xlabel('Time (Years)')

# pl.show()