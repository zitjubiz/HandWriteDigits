import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

# 获取数据（如果存在就读取，不存在就下载完再读取,网络问题,请先下载）
#"one-hot vectors"。 一个one-hot向量除了某一位的数字是1以外其余各维度数字都是0。
# 数字n将表示成一个只有在第n维度（从0开始）数字为1的10维向量。比如，标签3将表示成([0,0,0,1,0,0,0,0,0,0,0])
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

# 输入
x = tf.placeholder("float", [None, 784]) #输入占位符（每张手写数字784个像素点）
y_ = tf.placeholder("float", [None,10]) #输入占位符（这张手写数字具体代表的值，0-9对应矩阵的10个位置）

# 计算分类softmax会将xW+b分成10类，对应0-9
W = tf.Variable(tf.zeros([784,10])) #权重
b = tf.Variable(tf.zeros([10])) #偏置
y = tf.nn.softmax(tf.matmul(x,W) + b) # 输入矩阵x与权重矩阵W相乘，加上偏置矩阵b，然后求softmax（sigmoid函数升级版，可以分成多类）

# 计算偏差和
cross_entropy = -tf.reduce_sum(y_*tf.log(y))

# 使用梯度下降法（步长0.01），来使偏差和最小
train_step = tf.train.GradientDescentOptimizer(0.01).minimize(cross_entropy)

# 初始化变量
init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)
#每一个MNIST数据单元有两部分组成：一张包含手写数字的图片和一个对应的标签。我们把这些图片设为“xs”，把这些标签设为“ys”
# 训练数据集和测试数据集都包含xs和ys，比如训练数据集的图片是 mnist.train.images ，训练数据集的标签是 mnist.train.labels
for i in range(100): # 训练次数
    batch_xs, batch_ys = mnist.train.next_batch(100) # 随机取100个手写数字图片
    sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys}) # 执行梯度下降算法，输入值x：batch_xs，输入值y：batch_ys

# 计算训练精度
correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))

accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
print(accuracy)
print(sess.run(accuracy, feed_dict={x: mnist.test.images, y_: mnist.test.labels})) #运行精度图，x和y_从测试手写图片中取值