## 代码结构说明

### mnist_mlp.py文件

class MNIST_MLP为神经网络的核心代码，定义了如下方法：

build_model：建立模型，在这块可以调整模型的结构、层数等。可选的参数有激活函数是sigmoid还是ReLU。

init_model：调用layers.py的方法，逐层初始化模型参数。

forward：神经网络的前向传播。

backward：神经网络的反向传播，计算导数并保存。

update：神经网络的学习，更新每层的权重参数。

save_model：保存模型参数为npy文件。

load_mnist：用二进制方法加载数据集，返回数据列表。

load_data：加载训练接和测试集。

shuffle_data：随机打乱数据集，在train中有调用。

train：模型训练。可以调整训练迭代数、batch_size等。

load_model：加载已经训练好的模型参数。

evaluate：验证模型在测试集上的正确率。

除此之外，还有build_mnist_mlp函数：先后调用了模型构建、数据集引入、搭建神经网络、初始化网络参数、模型训练和模型保存等步骤。

最后的main函数中，可以更改随机数种子、模型层数和每层参数、激活函数等。还有两个绘图函数，调用了matplotlib。

### layers.py文件

实现了全连接层、ReLU激活函数、Sigmoid激活函数、Softmax+Loss层等4个类。每个类的核心代码在实验报告中有所阐述，在此不做赘述。全连接层类里的init_param方法中，实现了5个实验中的神经网络初始化方法，可以逐一调用。

## 压缩包整体说明

exp文件夹下为各项实验的结果，包括正确率损失图像、代码文件以及模型文件。

mnist文件夹下为下载好的MNIST数据集。

外层有两个代码文件，前面叙述过，还有一个PDF文档，为本次实验的报告。