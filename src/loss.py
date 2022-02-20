import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
import scipy.stats as stats
import numerical_functions as numfun

class Prior():
    def __init__(self):
        pass

    def Zw(self, gamma, Lambda):
        pass

    def fw(self, gamma, Lambda):
        pass

    def Dfw(self, gamma, Lambda):
        pass

class Channel():
    def __init__(self, reg_param):
        pass
    
    def Zout(self, y, omega, V):
        pass

    def fout(self, y, omega, V):
        pass

    def Dfout(self, y, omega, V):
        pass

    def __repr__(self):
        return "reg_param : {:.4f}".format()

class GaussianPrior(Prior):
    name = "Gaussian Prior"
    tex_string = r"$L(y,z) = \frac{1}{2} (y - z)^2 + \frac{lambda}{2} w^2$"

    def __init__(self):
        pass

    def Zw(self, gamma, Lambda):
        return 

    def fw(self, gamma, Lambda):
        return 

    def Dfw(self, gamma, Lambda):
        return 

class L2Loss(Channel):
    name = "L2 loss"
    tex_string = r"$L(y,z) = \frac{1}{2} (y - z)^2 + \frac{lambda}{2} w^2$"

    def __init__(self, reg_param):
        self.reg_param = reg_param

    def fout(self, y, omega, V):
        return (y - omega) / (1 + V)

    def Dfout(self, y, omega, V):
        return - 1.0 / (1 + V)


class L1Loss(Channel):
    name = "L1 loss"
    tex_string = r"$L(y,z) = \frac{1}{2} |y - z| + \frac{lambda}{2} w^2$"

    def __init__(self, reg_param):
        self.reg_param = reg_param

    def fout(self, y, omega, V):
        pass

    def Dfout(self, y, omega, V):
        pass

class HuberLoss(Channel):
    name = "Huber loss"
    tex_string = r"$L(y,z) = ... + \frac{lambda}{2} w^2$"

    def __init__(self, reg_param):
        self.reg_param = reg_param

    def fout(self, y, omega, V):
        pass

    def Dfout(self, y, omega, V):
        pass