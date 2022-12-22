from sklearn.preprocessing import normalize
from sklearn.datasets import load_digits
import numpy as np
import skimage.measure

class DownsampledMNIST:
    def __init__(self, classes = None):
        self.data = load_digits()
        self.X = self.data.data
        self.Y = self.data.target
        self.classes = classes

        self.normalize()
        self.preprocessing()
        self.class_selection()

    def normalize(self):
        self.X = normalize(self.X, axis = 0, norm = 'max')

    def preprocessing(self):
        samples = []
        for i in range(len(self.X)):
            img = skimage.measure.block_reduce(self.X[i], 4, np.mean)
            img = img.flatten()
            samples.append(img)
        
        self.X = np.array(samples)

    def concatenate(self,datas,labels):
        con_data=datas[0]
        con_label=labels[0]
        for i in range(1,len(datas)):
            con_data=np.concatenate((con_data,datas[i]),axis=0)
            con_label=np.concatenate((con_label,labels[i]),axis=0)
        return con_data,con_label

    def class_selection(self):
        datas, labels = [], []
        for item in self.classes:
            data = self.X[np.array(self.Y) == item]
            datas.append(data)
            labels.append(np.full((data.shape[0]), item))

        X, Y = self.concatenate(datas, labels)
        self.X = X
        self.Y = Y

if __name__ == "__main__":
    classes = [3, 6, 7]
    data = DownsampledMNIST(classes = classes)
    print("X data:", data.X.shape)
    print("Y data:", data.Y.shape)