import numpy as np
import matplotlib.pyplot as plt

#
def generate_train_and_val_indices(dataset_size,np_seed,validation_split=0.2):
    if np_seed==True:
        np.random.seed(123)
    indices = list(range(dataset_size))
    split = int(np.floor(validation_split * dataset_size))
    train_indices, val_indices = indices[split:], indices[:split]
    return np.array(train_indices),np.array(val_indices)

class IndexTracker(object):
    def __init__(self, ax, X,map='gray'):
        self.ax = ax
        ax.set_title('use scroll wheel to navigate images')
        self.X = X
        rows, cols, self.slices = X.shape
        self.ind = self.slices//2

        self.im = ax.imshow(self.X[:, :, self.ind],cmap=map)
        self.update()

    def onscroll(self, event):
        print("%s %s" % (event.button, event.step))
        if event.button == 'up':
            self.ind = (self.ind + 1) % self.slices
        else:
            self.ind = (self.ind - 1) % self.slices
        self.update()

    def update(self):
        self.im.set_data(self.X[:, :, self.ind])
        self.ax.set_ylabel('slice %s' % self.ind)
        self.im.axes.figure.canvas.draw()

#
def ViewImage(img):
    fig, ax = plt.subplots(1, 1)
    tracker = IndexTracker(ax, img)
    plt.show()
    fig.canvas.mpl_connect('scroll_event', tracker.onscroll)


def bbox_3D(img):
    r = np.any(img, axis=(1, 2))
    c = np.any(img, axis=(0, 2))
    z = np.any(img, axis=(0, 1))
    rmin, rmax = np.where(r)[0][[0, -1]]
    cmin, cmax = np.where(c)[0][[0, -1]]
    zmin, zmax = np.where(z)[0][[0, -1]]
    return rmin, rmax, cmin, cmax, zmin, zmax
