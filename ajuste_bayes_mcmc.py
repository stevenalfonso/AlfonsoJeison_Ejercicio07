import numpy as np
import matplotlib.pyplot as plt

data = np.loadtxt("notas_andes.dat", skiprows=1)

def model(x, b1, b2, b3, b4, b5):
    y = b1 + b2*x[:,0] + b3*x[:,1] + b4*x[:,2] + b5*x[:,3]
    return y
sigma = 0.1
n_ = 69
def like(x, y, sigma, b1, b2, b3, b4, b5, n_):
    a = (-1/(2 * sigma**2))*np.sum((y - model(x, b1, b2, b3, b4, b5))**2)
    return a

N = 20000
def metropolis(N):
    Y = data[:,4]
    X = data[:,:4]
    b1, b2, b3, b4, b5 = np.zeros(5)
    #b1, b2, b3, b4, b5 = np.random.random(5)
    p = like(X, Y, sigma, b1, b2, b3, b4, b5, n_)
    b1l = []
    b2l = []
    b3l = []
    b4l = []
    b5l = []
    for i in range(N): # variar betas
        b1_n = b1 + np.random.normal(0,0.05)
        b2_n = b2 + np.random.normal(0,0.05)
        b3_n = b3 + np.random.normal(0,0.05)
        b4_n = b4 + np.random.normal(0,0.05)
        b5_n = b5 + np.random.normal(0,0.05)
        p_n = like(X, Y, sigma, b1_n, b2_n, b3_n, b4_n, b5_n, n_)
        u = np.random.rand()
        if u < min(1, np.exp(p_n - p)):
            p = p_n
            b1 = b1_n
            b2 = b2_n
            b3 = b3_n
            b4 = b4_n
            b5 = b5_n
        b1l.append(b1)
        b2l.append(b2)
        b3l.append(b3)
        b4l.append(b4)
        b5l.append(b5)
    return b1l, b2l, b3l, b4l, b5l
b1l, b2l, b3l, b4l, b5l = metropolis(N)
betas = np.array([b1l, b2l, b3l, b4l, b5l])

plt.figure(figsize=(8,8))
for i in range(5):
    plt.subplot(2,3,i+1)
    plt.hist(betas[i,10000:20000])
    plt.title(r'$\beta_%1.0f $='%i  + '%0.4f' %np.mean(betas[i,10000:20000]) + r'$\pm$ %0.4f' %np.std(betas[i,10000:20000]))
    plt.xlabel(r'$\beta_%1.0f $'%i)
    plt.tight_layout()
plt.savefig('ajuste_bayes_mcmc.png')
plt.show()