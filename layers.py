import numpy as np


class FullyConnectedLayer:
    def __init__(self, num_input, num_output):
        self.num_input = num_input
        self.num_output = num_output
        self.last_dweight = 0
        self.last_dbias = 0

    def init_param(self, std=0.01):
        # 高斯随机初始化
        # self.weight = np.random.normal(loc=0.0, scale=std, size=(self.num_input, self.num_output))
        # self.bias = np.random.normal(loc=0.0, scale=std, size=(1, self.num_output))
        # 零初始化
        # self.weight = np.zeros_like(self.weight)
        # self.bias = np.zeros([1, self.num_output])
        # 均匀随机初始化
        # a = np.sqrt(1. / self.num_input)
        # self.weight = np.random.uniform(low=-a, high=a, size=(self.num_input, self.num_output))
        # self.bias = np.random.uniform(low=-a, high=a, size=(1, self.num_output))
        # xavier normal初始化
        # std = np.sqrt(2. / (self.num_input + self.num_output))
        # print(std)
        # self.weight = np.random.normal(loc=0.0, scale=std, size=(self.num_input, self.num_output))
        # self.bias = np.random.normal(loc=0.0, scale=std, size=(1, self.num_output))
        # kaiming normal初始化
        std = np.sqrt(2. / (self.num_input))
        print(std)
        self.weight = np.random.normal(loc=0.0, scale=std, size=(self.num_input, self.num_output))
        self.bias = np.random.normal(loc=0.0, scale=std, size=(1, self.num_output))

    def forward(self, x):
        self.x = x
        self.output = np.dot(x, self.weight) + self.bias
        return self.output

    def backward(self, top_diff):
        self.d_weight = np.dot(self.x.T, top_diff)
        self.d_bias = top_diff.sum(axis=0)
        bottom_diff = np.dot(top_diff, self.weight.T)
        return bottom_diff

    def update_param_simple(self, lr):
        self.weight = self.weight - lr * self.d_weight
        self.bias = self.bias - lr * self.d_bias

    def update_param(self, lr):
        momentum = 0.9
        v = momentum * self.last_dweight + lr * self.d_weight
        self.weight = self.weight - v
        self.last_dweight = v
        u = momentum * self.last_dbias + lr * self.d_bias
        self.bias = self.bias - u
        self.last_dbias = u

    def load_param(self, weight, bias):
        assert self.weight.shape == weight.shape
        assert self.bias.shape == bias.shape
        self.weight = weight
        self.bias = bias

    def save_param(self):
        return self.weight, self.bias


class ReLULayer:
    def forward(self, x):
        self.x = x
        output = np.maximum(0, self.x)
        return output

    def backward(self, top_diff):
        bottom_diff = np.zeros_like(self.x)
        mask = self.x >= 0
        np.putmask(bottom_diff, mask, top_diff)
        return bottom_diff


class SigmoidLayer:
    def sigma(self, x):
        return 1/(1 + np.exp(-x))

    def forward(self, x):
        self.x = x
        self.output = self.sigma(x)
        return self.output

    def backward(self, top_diff):
        bottom_diff = top_diff * self.output * (1 - self.output)
        return bottom_diff


class SoftmaxLossLayer:
    def forward(self, x):
        x_max = np.max(x, axis=1, keepdims=True)
        x_exp = np.exp(x - x_max)
        self.prob = x_exp / (x_exp.sum(axis=1).reshape(-1, 1))
        return self.prob

    def get_loss(self, label):
        self.batch_size = self.prob.shape[0]
        self.label_onehot = np.zeros_like(self.prob)
        self.label_onehot[np.arange(self.batch_size), label] = 1.0
        self.prob[self.prob <= 0] = 1e-5
        loss = -np.sum(np.log(self.prob) * self.label_onehot) / self.batch_size
        return loss

    def backward_simple(self):
        # CrossEntropy 和 Softmax 的求导放一块了
        bottom_diff = (- self.label_onehot + self.prob) / self.batch_size
        return bottom_diff

    def backward(self):
        # CrossEntropy 和 Softmax 的求导放一块了
        bottom_diff = (- self.label_onehot + self.prob) / self.batch_size
        return bottom_diff

