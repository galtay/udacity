"""
Classification on the notMNIST data set
http://yaroslavvb.blogspot.com/2011/09/notmnist-dataset.html

We have two datasets notMNIST_large and notMNIST_small
Each is a directory with 10 sub-directories.  Each sub-directory
contains images of a letter from various fonts.

Images of letters [A, B, C, D, E, F, G, H, I, J]

Note the convention in scikitlearn for the X array is to put the
features as columns and the samples in rows
X.shape = (N_SAMPLES, N_FEATURES)
"""

import os
import mahotas
import numpy as np

import sklearn
from sklearn.cross_validation import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report

import matplotlib.pyplot as plt
from matplotlib import cm


NPIX = 28
LABELS = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J']
RANDOM_STATE = 101


class DataSet(object):

    def __init__(self, data_dir):
        self.data_dir = data_dir
        self.labels = LABELS
        self.counts = self.get_image_counts()
        self.nimgs = sum(self.counts.values())

    def split_train_validate_test(self, random_state=RANDOM_STATE):
        # split off the training set
        X_train, X_vt, y_train, y_vt = train_test_split(
            self.Xs, self.y, test_size=0.2, random_state=random_state)
        # split the vt set into validation and test
        X_val, X_test, y_val, y_test = train_test_split(
            X_vt, y_vt, test_size=0.5, random_state=random_state)
        X = {'train': X_train, 'val': X_val, 'test': X_test}
        y = {'train': y_train, 'val': y_val, 'test': y_test}
        return X, y

    def attach_X_y(self, shuffle=True, random_state=RANDOM_STATE):
        y = []
        imgs = []
        nimgs = 0
        for i_label, label in enumerate(self.labels):
            imgs_1 = self.load_images_for_label(label)
            nimgs_1 = imgs_1.shape[0]
            imgs.append(imgs_1)
            y_1 = np.array([i_label] * nimgs_1, dtype=np.uint8)
            y.append(y_1)
        y = np.concatenate(y)
        imgs = np.concatenate(imgs)
        X = imgs.reshape(imgs.shape[0], NPIX*NPIX)
        if shuffle:
            X, y = sklearn.utils.shuffle(X, y, random_state=random_state)
        self.X = X
        self.y = y

    def standardize_X(self):
        scaler = StandardScaler()
        self.Xs = scaler.fit_transform(self.X)

    def get_image_counts(self):
        """count the number of image files in each label directory"""
        counts = {}
        for label_name in self.labels:
            label_path = os.path.join(self.data_dir, label_name)
            file_names = os.listdir(label_path)
            nfiles = len(file_names)
            counts[label_name] = nfiles
            print('label dir {} has {} files'.format(label_path, nfiles))
        return counts

    def load_images_for_label(self, label):
        """return an ndarray with all images in a label directory"""
        label_path = os.path.join(self.data_dir, label)
        print('loading images from {}'.format(label_path))
        file_names = os.listdir(label_path)
        nfiles = len(file_names)
        imgs = np.zeros((nfiles, NPIX, NPIX), dtype=np.uint8)
        igood = 0
        for file_name in file_names:
            file_path = os.path.join(label_path, file_name)
            try:
                imgs[igood,:,:] = mahotas.imread(file_path)
                igood += 1
            except OSError:
                print('skipping {}'.format(file_path))
        imgs = imgs[0:igood,:,:]
        print('skipped {} of {}'.format(nfiles-igood, nfiles))
        return imgs

    def plot_random_selection(self, imgs, nrows=5, ncols=5):
        """plot a random selection of images in the `imgs` array"""
        nimgs = imgs.shape[0]
        fig, axs = plt.subplots(
            nrows, ncols, sharex=True, sharey=True, figsize=(7,7))
        for ir in range(nrows):
            for ic in range(ncols):
                iplt = np.random.randint(low=0, high=nimgs)
                axs[ir,ic].imshow(imgs[iplt,:,:], cmap=cm.gray)
        fig.subplots_adjust(hspace=0.1, wspace=0.1)





#data_dir = 'notMNIST_small'
data_dir = 'notMNIST_large'
ds = DataSet(data_dir)
ds.attach_X_y()
ds.standardize_X()
X, y = ds.split_train_validate_test()
model = LogisticRegression(random_state=RANDOM_STATE, verbose=0)
for n_samples in [50, 100, 1000, 5000, None]:
    XX = X['train'][:n_samples,:]
    yy = y['train'][:n_samples]
    print('training with {} samples.'.format(len(yy)))
    model.fit(XX, yy)

    y_pred = model.predict(X['val'])
    print(classification_report(y['val'], y_pred))
    print()
