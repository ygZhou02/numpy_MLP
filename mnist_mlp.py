import numpy as np
import matplotlib.pyplot as plt
import struct
import os

from layers import FullyConnectedLayer, ReLULayer, SigmoidLayer, SoftmaxLossLayer


class MNIST_MLP:
    def __init__(self, batch_size=100, input_size=784, hidden1=256, hidden2=128,
                 hidden3=64, hidden4=32, hidden5=16, out_classes=10,
                 lr=0.0005, max_epoch=1, print_iter=100, active_fun="relu"):
        self.batch_size = batch_size
        self.input_size = input_size
        self.hidden1 = hidden1
        self.hidden2 = hidden2
        self.hidden3 = hidden3
        self.hidden4 = hidden4
        self.hidden5 = hidden5
        self.out_classes = out_classes
        self.lr = lr
        self.max_epoch = max_epoch
        self.print_iter = print_iter
        self.update_layer_list = []
        self.active_fun = active_fun

    def build_model(self):
        self.fc1 = FullyConnectedLayer(self.input_size, self.hidden1)
        self.fc2 = FullyConnectedLayer(self.hidden1, self.hidden2)
        self.fc3 = FullyConnectedLayer(self.hidden2, self.hidden3)
        self.fc4 = FullyConnectedLayer(self.hidden3, self.hidden4)
        self.fc5 = FullyConnectedLayer(self.hidden4, self.hidden5)
        self.fc6 = FullyConnectedLayer(self.hidden5, self.out_classes)
        self.softmax = SoftmaxLossLayer()
        if self.active_fun == "sigmoid":
            self.activate1 = SigmoidLayer()
            self.activate2 = SigmoidLayer()
            self.activate3 = SigmoidLayer()
            self.activate4 = SigmoidLayer()
            self.activate5 = SigmoidLayer()
        elif self.active_fun == "relu":
            self.activate1 = ReLULayer()
            self.activate2 = ReLULayer()
            self.activate3 = ReLULayer()
            self.activate4 = ReLULayer()
            self.activate5 = ReLULayer()
        self.update_layer_list = [self.fc1, self.fc2, self.fc3, self.fc4, self.fc5, self.fc6]

    def init_model(self):
        for layer in self.update_layer_list:
            layer.init_param()

    def forward(self, x):
        x = self.fc1.forward(x)
        x = self.activate1.forward(x)
        x = self.fc2.forward(x)
        x = self.activate2.forward(x)
        x = self.fc3.forward(x)
        x = self.activate3.forward(x)
        x = self.fc4.forward(x)
        x = self.activate4.forward(x)
        x = self.fc5.forward(x)
        x = self.activate5.forward(x)
        x = self.fc6.forward(x)
        x = self.softmax.forward(x)
        return x

    def backward(self):
        dloss = self.softmax.backward()
        dh6 = self.fc6.backward(dloss)
        dh5 = self.activate5.backward(dh6)
        dh5 = self.fc5.backward(dh5)
        dh4 = self.activate4.backward(dh5)
        dh4 = self.fc4.backward(dh4)
        dh3 = self.activate3.backward(dh4)
        dh3 = self.fc3.backward(dh3)
        dh2 = self.activate2.backward(dh3)
        dh2 = self.fc2.backward(dh2)
        dh1 = self.activate1.backward(dh2)
        dh1 = self.fc1.backward(dh1)

    def update(self, lr):
        for layer in self.update_layer_list:
            layer.update_param(lr)

    def save_model(self, param_dir):
        params = {}
        params['w1'], params['b1'] = self.fc1.save_param()
        params['w2'], params['b2'] = self.fc2.save_param()
        params['w3'], params['b3'] = self.fc3.save_param()
        params['w4'], params['b4'] = self.fc4.save_param()
        params['w5'], params['b5'] = self.fc5.save_param()
        params['w6'], params['b6'] = self.fc6.save_param()
        np.save(param_dir, params)

    def load_mnist(self, file_dir, is_images=True):
        bin_file = open(file_dir, 'rb')
        bin_data = bin_file.read()
        bin_file.close()
        if is_images:
            fmt_header = '>iiii'
            magic, num_images, num_rows, num_cols = struct.unpack_from(fmt_header, bin_data, 0)
        else:
            fmt_header = '>ii'
            magic, num_images = struct.unpack_from(fmt_header, bin_data, 0)
            num_rows, num_cols = 1, 1
        data_size = num_images * num_rows * num_cols
        mat_data = struct.unpack_from('>' + str(data_size) + 'B', bin_data, struct.calcsize(fmt_header))
        mat_data = np.reshape(mat_data, [num_images, num_rows * num_cols])
        return mat_data

    def load_data(self):
        MNIST_DIR = 'mnist'
        TRAIN_DATA = 'train-images.idx3-ubyte'
        TRAIN_LABELS = 'train-labels.idx1-ubyte'
        TEST_DATA = 't10k-images.idx3-ubyte'
        TEST_LABELS = 't10k-labels.idx1-ubyte'
        train_images = self.load_mnist(os.path.join(MNIST_DIR, TRAIN_DATA), True)
        train_labels = self.load_mnist(os.path.join(MNIST_DIR, TRAIN_LABELS), False)
        test_images = self.load_mnist(os.path.join(MNIST_DIR, TEST_DATA), True)
        test_labels = self.load_mnist(os.path.join(MNIST_DIR, TEST_LABELS), False)
        self.train_data = np.append(train_images, train_labels, axis=1)
        self.test_data = np.append(test_images, test_labels, axis=1)

    def shuffle_data(self):
        np.random.shuffle(self.train_data)

    def train(self):
        print('start training')
        max_batch = int(self.train_data.shape[0] / self.batch_size)
        losses = []
        accuracy_list = []
        for idx_epoch in range(self.max_epoch):
            self.shuffle_data()
            for idx_batch in range(max_batch):
                batch_images = self.train_data[idx_batch * self.batch_size:(idx_batch + 1) * self.batch_size, :-1] / 255
                batch_labels = self.train_data[idx_batch * self.batch_size:(idx_batch + 1) * self.batch_size, -1]
                prob = self.forward(batch_images)
                loss = self.softmax.get_loss(batch_labels)
                losses.append(loss)
                self.backward()
                self.update(self.lr)
                if idx_batch % self.print_iter == 0:
                    print('Epoch %d, iter %d, loss: %.6f' % (idx_epoch, idx_batch, loss))
            accuracy = self.evaluate()
            accuracy_list.append(accuracy)
        return losses, accuracy_list

    def load_model(self, param_dir):
        params = np.load(param_dir, allow_pickle=True).item()
        self.fc1.load_param(params['w1'], params['b1'])
        self.fc2.load_param(params['w2'], params['b2'])
        self.fc3.load_param(params['w3'], params['b3'])
        self.fc4.load_param(params['w4'], params['b4'])
        self.fc5.load_param(params['w5'], params['b5'])
        self.fc6.load_param(params['w6'], params['b6'])

    def evaluate(self):
        pred_results = np.zeros([self.test_data.shape[0]])
        for idx in range(int(self.test_data.shape[0] / self.batch_size)):
            batch_images = self.test_data[idx * self.batch_size:(idx + 1) * self.batch_size, :-1]
            prob = self.forward(batch_images)
            pred_labels = np.argmax(prob, axis=1)
            pred_results[idx * self.batch_size:(idx + 1) * self.batch_size] = pred_labels
        accuracy = np.mean(pred_results == self.test_data[:, -1])
        print('Accuracy in test set: %f' % accuracy)
        return accuracy


def build_mnist_mlp(h1, h2, h3, h4, h5, e, lr, active_fun):
    mlp = MNIST_MLP(hidden1=h1, hidden2=h2, hidden3=h3, hidden4=h4, hidden5=h5, max_epoch=e, lr=lr, active_fun=active_fun)
    mlp.load_data()
    mlp.build_model()
    mlp.init_model()
    # mlp.load_model("exp/不同学习率的对比/32-16-100epoch-01lr-ReLU-Gaussian/mlp-64-32-16-100epoch-001lr.npy")
    losses, acc_list = mlp.train()
    mlp.save_model('mlp-%d-%d-%d-%d-%d-%depoch-%flr.npy' % (h1, h2, h3, h4, h5, e, lr))
    # BASELINE:: 32-16 100epoch 0.01lr ReLU normal Acc=93.58
    # 32-16 100epoch 0.01lr ReLU Kaiming Acc=95.62
    # 32-16 100epoch 0.01lr ReLU Xavier Acc=95.77
    # 32-16 100epoch 0.01lr ReLU uniform Acc=95.47
    # 32-16 100epoch 0.01lr ReLU 0 Acc=11.35
    # 32-16 100epoch 0.1lr ReLU normal Acc=95.14
    # 32-16 100epoch 1lr ReLU normal Acc=89.17
    # 32-16 100epoch 0.001lr ReLU normal Acc=10.61
    # 32-16 100epoch 0.01lr sigmoid normal Acc=11.35
    # 32-16 100epoch 0.1lr sigmoid normal Acc=90.99
    # 32-16 100epoch 0.01lr sigmoid normal momentum Acc=89.84
    # 32-16 100epoch 0.1lr sigmoid normal momentum Acc=86.23
    # 32-16 100epoch 0.01lr ReLU normal momentum Acc=95.01
    # 32-16 100epoch 0.1lr ReLU normal momentum Acc=95.07
    return mlp, losses, acc_list


if __name__ == '__main__':
    np.random.seed(43)
    h1, h2, h3, h4, h5, e, lr = 256, 128, 64, 32, 16, 100, 0.01
    active_fun = "relu"
    mlp, losses, acc_list = build_mnist_mlp(h1, h2, h3, h4, h5, e, lr, active_fun)
    mlp.evaluate()
    plt.figure()

    ax = plt.axes()
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    plt.xlabel('iters')  # x轴标签
    plt.ylabel('Acc')  # y轴标签

    plt.plot(np.arange(len(acc_list)), acc_list, linewidth=1, linestyle="solid", label="test accuracy")
    plt.legend()
    plt.title('Test Acc curve')
    plt.savefig('test_acc_curve_'+str(h1)+'-'+str(h2)+'-'+str(h3)+'-'+str(h4)+'-'+str(h5)+'-'+'-'+str(e)+'epoch-'+str(lr)+'lr.png', bbox_inches='tight')

    plt.show()

    plt.figure()

    ax = plt.axes()
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    plt.xlabel('iters')  # x轴标签
    plt.ylabel('Loss')  # y轴标签

    plt.plot(np.arange(len(losses)), losses, linewidth=1, linestyle="solid", label="Loss")
    plt.legend()
    plt.title('Loss curve')
    plt.savefig('Loss_curve_'+str(h1)+'-'+str(h2)+'-'+str(h3)+'-'+str(h4)+'-'+str(h5)+'-'+'-'+str(e)+'epoch-'+str(lr)+'lr.png', bbox_inches='tight')

    plt.show()

