import numpy as np

class MNISTDataset:

    def __init__(self,path):
        self.path = path
        self.train_examples = 60000
        self.test_examples = 10000
        self.image_size = 28
        self.train_images = self.read_training_images()
        self.train_labels = self.read_training_labels()
        self.test_images = self.read_test_images()
        self.test_labels = self.read_test_labels()

    def read_training_images(self):
        with open(self.path+'train-images-idx3-ubyte', encoding = "ISO-8859-1") as f:
            f.read(16)
            buff = f.read(self.image_size**2 * self.train_examples)
            print(buff[200:3000])
            data = np.frombuffer(buff, dtype = np.uint8).astype(np.float32)
            train_data = data.reshape(self.train_examples, self.image_size, self.image_size, 1)

        return train_data

    def read_training_labels(self):
        with open(self.path+'train-labels-idx1-ubyte') as f:
            f.read(8)
            buff = f.read(self.train_examples)
            data = np.frombuffer(buff, dtype = np.uint8).astype(np.float32)
            train_labels = data.reshape(self.train_examples, 1, 1)

        return train_labels

    def read_test_images(self):
        with open(self.path+'t10k-images-idx3-ubyte') as f:
            f.read(16)
            buff = f.read(self.image_size**2 * self.test_examples)
            data = np.frombuffer(buff, dtype = np.uint8).astype(np.float32)
            test_data = data.reshape(self.test_examples, self.image_size, self.image_size, 1)

        return test_data

    def read_test_labels(self):
        with open(self.path+'t10k-labels-idx1-ubyte') as f:
            f.read(8)
            buff = f.read(self.test_examples)
            data = np.frombuffer(buff, dtype = np.uint8).astype(np.float32)
            train_labels = data.reshape(self.test_examples, 1, 1)

        return test_labels
if __name__=="__main__":
    x = MNISTDataset('')
    import matplotlib.pyplot as plt
    image = np.asarray(x.train_images[2]).squeeze()
    plt.imshow(image)
    plt.show()
