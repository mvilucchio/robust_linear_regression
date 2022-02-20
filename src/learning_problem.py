from contextlib import redirect_stderr
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
import scipy.stats as stats
import numerical_functions as numfun
from tqdm.auto import tqdm


class LearningProlem():
    def __init__(self, loss, reg, channel, prior) -> None:
        self.loss = loss
        self.reg = reg
        self.channel = channel
        self.prior = prior
