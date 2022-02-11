import numpy as np
import matplotlib.pyplot as plt
import scipy as sc
from tqdm.auto import tqdm

def DZoutBayes(y, omega, V, Delta):
    return np.exp(-(omega-y)**2/(2*(Delta+V))) * (omega - y)/np.sqrt(2*np.pi*(V+Delta))

def ZoutBayes(y, omega, V, Delta):
    return np.exp(-(omega-y)**2/(2*(Delta+V))) / np.sqrt(2*np.pi*(V+Delta))

def foutBayes():
    return 

def foutL2(y, omega, V):
    return (y - omega) / (1 + V)

def DfoutL2(y, omega, V):
    return - 1.0 / (1 + V)

def foutL1():
    return  

def DfoutL1():
    return
