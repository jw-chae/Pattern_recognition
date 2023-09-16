import numpy as np
from IPython import display
import matplotlib.pyplot as plt
np.random.seed(42)

def make_sample(w=1, b=0, size=100, noise=1.0): #w = weight b = bias
    x = np.random.rand(size)
    y = w * x + b
    noise = np.random.uniform(-abs(noise), abs(noise), size=y.shape)
    yy = y + noise
    return x, yy
def make_multi_sample(num_inputs,num_features): #m = size n = features
    # num_inputs = 2
    # num_features = 1000
    true_w = np.array([1 for i in range(0,num_features)])
    true_b = 0
    features = np.random.normal(scale=1, size=(num_features, num_inputs))
    labels = np.dot(features, true_w) + true_b
    labels += np.random.normal(scale=0.01, size=labels.shape)
    return features, labels
x,y = make_multi_sample(100,100)
print(x.shape,y.shape)
#print(x)
def use_svg_display():
    # Display in vector graphics
    display.set_matplotlib_formats('svg')

def set_figsize(figsize=(3.5, 2.5)):
    use_svg_display()
    # Set the size of the graph to be plotted
    plt.rcParams['figure.figsize'] = figsize


def main():
    set_figsize()
    plt.figure(figsize=(10, 6))
    plt.scatter(x[:, 99], y, 1)
    plt.show()

if __name__==main():
    main()