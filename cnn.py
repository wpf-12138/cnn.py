import tensorflow as tf


class TextCNN(object):

    """文本分类，CNN模型"""

    def __init__(self, config):
        self.config = config

        # 三个待输入的数据
        self.input_x = tf.placeholder(tf.int32, [None, self.config.seq_length], name='input_x')
        self.input_y = tf.placeholder(tf.float32, [None, self.config.num_classes], name='input_y')
        self.keep_prob = tf.placeholder(tf.float32, name='keep_prob')

        self.cnn()

    def cnn(self):
        """CNN模型"""
        # 词向量映射
        with tf.name_scope("embedding"):
            embedding = tf.get_variable('embedding', [self.config.vocab_size, self.config.embedding_dim])
            embedding_inputs = tf.nn.embedding_lookup(embedding, self.input_x)

        with tf.name_scope("cnn1"):
            # CNN layer
            conv1 = tf.layers.conv1d(embedding_inputs, 10, 5, name='conv1')
            # global max pooling layer
            # gmp1 = tf.reduce_max(conv1, reduction_indices=[1], name='gmp1')
            maxpool1 = tf.layers.max_pooling1d(conv1, pool_size=2, strides=2, name="maxpool1")
            gmp1 = tf.nn.relu(maxpool1)

        with tf.name_scope("cnn2"):
            # CNN layer
            conv2 = tf.layers.conv1d(gmp1, 10, 3, name='conv2')
            maxpool2 = tf.layers.max_pooling1d(conv2, pool_size=2, strides=2, name="maxpool2")
            gmp2 = tf.nn.relu(maxpool2)

        with tf.name_scope("cnn3"):
            # CNN layer
            conv3 = tf.layers.conv1d(gmp2, 10, 3, name='conv3')
            maxpool3 = tf.layers.max_pooling1d(conv3, pool_size=2, strides=2, name="maxpool3")
            gmp3 = tf.nn.relu(maxpool3)

        with tf.name_scope("score"):
            # 全连接层，后面接dropout以及relu激活
            fc = tf.layers.flatten(gmp3)
            fc = tf.layers.dense(fc, 64, name='fc1')
            fc = tf.contrib.layers.dropout(fc, self.keep_prob)
            fc = tf.nn.relu(fc)

            # 分类器
            self.logits = tf.layers.dense(fc, self.config.num_classes, name='fc2')
            self.softmax = tf.nn.softmax(self.logits)
            self.y_pred_cls = tf.argmax(tf.nn.softmax(self.logits), 1)  # 预测类别

        with tf.name_scope("optimize"):
            # 损失函数，交叉熵
            cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=self.logits, labels=self.input_y)
            self.loss = tf.reduce_mean(cross_entropy)
            # 优化器
            self.optim = tf.train.AdamOptimizer(learning_rate=self.config.learning_rate).minimize(self.loss)

        with tf.name_scope("accuracy"):
            # 准确率
            correct_pred = tf.equal(tf.argmax(self.input_y, 1), self.y_pred_cls)
            self.acc = tf.reduce_mean(tf.cast(correct_pred, tf.float32))