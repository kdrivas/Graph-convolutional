import numpy as np
import time
import math
import matplotlib.pyplot as plt

def as_minutes(s):
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)

def time_since(since, percent):
    now = time.time()
    s = now - since
    es = s / (percent)
    rs = es - s
    return '%s (- %s)' % (as_minutes(s), as_minutes(rs))

def show_plot(points):
    plt.figure()
    fig, ax = plt.subplots()
    loc = ticker.MultipleLocator(base=1) # put ticks at regular intervals
    ax.yaxis.set_major_locator(loc)
    plt.plot(points)
    
def plot_losses(train_loss, val_loss, scale):
    plt.figure(figsize=(10,5))
    plt.plot(train_loss)
    plt.plot([(x + 1) * scale - 1 for x in range(len(val_loss))], val_loss)
    plt.legend(['train loss', 'validation loss'])
