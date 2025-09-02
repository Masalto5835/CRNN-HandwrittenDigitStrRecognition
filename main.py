import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import layers, Model, callbacks
import matplotlib.pyplot as plt
import cv2
import random
import time
from tensorflow.keras.preprocessing.sequence import pad_sequences

# 配置参数
IMAGE_HEIGHT = 32           # 固定高度为 32
IMAGE_WIDTH  = 256          # 固定宽度为 228
BATCH_SIZE = 64
EPOCHS = 30                 # 训练步数 30 步
MAX_SEQ_LENGTH = 8          # 最大字符序列长度
NUM_CLASSES = 11            # 10个字符 + 1个空白类
CHARACTERS = "0123456789"   # 10个字符

# 加载并预处理EMNIST数据集
def load_and_preprocess_emnist(path):
    data = pd.read_csv(path, header=None)
    images = data.iloc[:, 1:].values.astype('float32')
    labels = data.iloc[:, 0].values.astype('int32')

    # 重塑图像并旋转（EMNIST图像需要旋转）
    images = images.reshape(-1, 28, 28, 1)
    images = np.transpose(images, [0, 2, 1, 3])  # 转置

    # 归一化
    images = images / 255.0

    return images, labels

# 加载训练和测试数据
print("加载EMNIST数据集...")
train_images, train_labels = load_and_preprocess_emnist('datasets/emnist-digits-train.CSV')
test_images, test_labels = load_and_preprocess_emnist('datasets/emnist-digits-test.CSV')
print(f"训练集大小: {len(train_images)}, 测试集大小: {len(test_images)}")

# 创建字符序列图像生成器
def generate_sequence_images(images, labels, num_samples, max_seq_len=MAX_SEQ_LENGTH):
    sequence_images = []
    sequence_labels = []
    label_lengths = []

    for _ in range(num_samples):
        # 随机确定序列长度 (1 到 max_seq_len)
        seq_len = random.randint(1, max_seq_len)

        # 随机选择字符
        char_indices = np.random.choice(len(images), seq_len, replace=True)

        # 创建空白画布
        canvas = np.zeros((IMAGE_HEIGHT, IMAGE_WIDTH, 1), dtype=np.float32)

        # 计算字符间距和起始位置
        total_char_width = sum(images[idx].shape[1] for idx in char_indices)
        spacing = (IMAGE_WIDTH - total_char_width) / (seq_len + 1) if seq_len > 0 else 0
        x_pos = max(0, int(spacing))

        # 在画布上放置字符
        char_labels = []
        for idx in char_indices:
            char_img = images[idx].squeeze() # 移除单通道维度
            char_h, char_w = char_img.shape  # 获取原始高度和宽度

            # 调整字符大小以匹配画布高度
            scale_factor = (IMAGE_HEIGHT - 4) / char_h  # 计算缩放比例
            new_w = max(5, int(char_w * scale_factor))  # 计算新宽度，至少保证5像素
            resized_char = cv2.resize(char_img, (new_w, IMAGE_HEIGHT - 4))  # 缩放图像到新尺寸

            # 随机垂直位置
            y_pos = random.randint(2, 4)       # 添加小范围垂直偏移，模仿真实文本的基线波动

            # 放置字符
            end_x = min(x_pos + new_w, IMAGE_WIDTH) # 确保不超出画像边界
            actual_w = end_x - x_pos                # 计算实际防止宽度
            if actual_w > 0:                        # 将图像复制到画布指定位置
                canvas[y_pos:y_pos + IMAGE_HEIGHT - 4, x_pos:end_x, 0] = resized_char[:, :actual_w]

            # 更新位置和标签
            x_pos += actual_w + random.randint(1, max(1, int(spacing)))
            char_labels.append(labels[idx])

        sequence_images.append(canvas)
        sequence_labels.append(char_labels)
        label_lengths.append(seq_len)

    return np.array(sequence_images), sequence_labels, np.array(label_lengths)

# 生成训练和测试序列数据集
print("生成序列图像...")
X_train, y_train, train_label_lengths = generate_sequence_images(train_images, train_labels, 100000)
X_test, y_test, test_label_lengths = generate_sequence_images(test_images, test_labels, 20000)
print(f"训练序列图像: {X_train.shape}, 测试序列图像: {X_test.shape}")

# 将标签填充为固定长度
def prepare_labels(labels, max_len=MAX_SEQ_LENGTH):
    # 使用-1填充标签序列
    padded_labels = pad_sequences(labels, maxlen=max_len, padding='post', value=-1)
    return padded_labels

y_train_padded = prepare_labels(y_train)
y_test_padded  = prepare_labels(y_test)

# 构建CRNN模型
def build_crnn_model():
    # 输入层 指定输入尺寸为 IMAGE_HEIGHT * IMAGE_WIDTH * 1
    input_image = layers.Input(shape=(IMAGE_HEIGHT, IMAGE_WIDTH, 1), name='image')

    # CNN特征提取
    x = layers.Conv2D(32, (3, 3), activation='relu', padding='same')(input_image)
    x = layers.BatchNormalization()(x)  # IMAGE_HEIGHT * IMAGE_WIDTH * 32
    x = layers.MaxPooling2D((2, 2))(x)  # IMAGE_HEIGHT/2 * IMAGE_WIDTH/2 * 32

    x = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(x)
    x = layers.BatchNormalization()(x)  # IMAGE_HEIGHT/2 * IMAGE_WIDTH/2 * 64
    x = layers.MaxPooling2D((2, 2))(x)  # IMAGE_HEIGHT/4 * IMAGE_WIDTH/4 * 64

    x = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(x)
    x = layers.BatchNormalization()(x)  # IMAGE_HEIGHT/4 * IMAGE_WIDTH/4 * 128
    x = layers.MaxPooling2D((1, 2))(x)  # IMAGE_HEIGHT/4 * IMAGE_WIDTH/8 * 128 (池化(1, 2)表示高度不变，宽度减半)

    x = layers.Conv2D(256, (3, 3), activation='relu', padding='same')(x)
    x = layers.BatchNormalization()(x)  # IMAGE_HEIGHT/4 * IMAGE_WIDTH/8  * 256
    x = layers.MaxPooling2D((1, 2))(x)  # IMAGE_HEIGHT/4 * IMAGE_WIDTH/16 * 256 (池化(1, 2)表示高度不变，宽度减半)

    # 准备RNN输入 (时间步长, 特征维度)
    x = layers.Reshape((IMAGE_WIDTH // 16, IMAGE_HEIGHT // 4 * 256))(x) # batch_size, IMAGE_WIDTH // 16(timestep), IMAGE_HEIGHT // 4 * 256
    x = layers.Dense(256)(x)                                            # batch_size, IMAGE_WIDTH // 16(timestep), 256

    # RNN序列建模
    x = layers.Bidirectional(layers.LSTM(128, return_sequences=True))(x)    # batch_size, IMAGE_WIDTH // 16(timestep), 256
    x = layers.Bidirectional(layers.LSTM(128, return_sequences=True))(x)    # batch_size, IMAGE_WIDTH // 16(timestep), 256

    # 输出层
    output = layers.Dense(NUM_CLASSES, activation='softmax', name='output')(x)   # batch_size, IMAGE_WIDTH // 16(timestep), NUM_CLASSES

    return Model(inputs=input_image, outputs=output)

# 自定义CTC损失函数
def ctc_loss(y_true, y_pred):
    # y_true形状为(batch_size, max_label_length) y_pred形状为(batch_size, timesteps, NUM_CLASSES)
    batch_size = tf.shape(y_pred)[0]
    input_length = tf.ones(shape=(batch_size, 1), dtype=tf.int32) * tf.shape(y_pred)[1]
    label_length = tf.ones(shape=(batch_size, 1), dtype=tf.int32) * tf.shape(y_true)[1]

    # 处理填充值
    y_true = tf.where(y_true == -1, NUM_CLASSES - 1, y_true)

    # 计算CTC损失
    loss = tf.keras.backend.ctc_batch_cost(
        y_true, y_pred, input_length, label_length)
    return loss

# 解码预测结果
def decode_predictions(preds, characters=CHARACTERS):
    """ preds:模型输出,形状为 (batch_size, timestep, NUM_CLASSES)
        characters:字符映射表,默认使用全局变量 CHARACTERS """
    # 在NUM_CLASSES维度上找出最大值的索引值(在每个时间步上选择概率最高的类别索引), 输出形状为 (batch_size, timestep), 并转换为numpy数组
    pred_indices = tf.argmax(preds, axis=2).numpy()

    decoded_texts = []
    for pred in pred_indices:
        prev_char = None    # 记录上一个非空白字符的索引
        text = []           # 存储当前样本的识别结果字符
        for idx in pred:    # 遍历时间步上的每个预测
            char_idx = idx
            if char_idx != (NUM_CLASSES - 1) and char_idx != prev_char: # 检查当前索引不是空白类且检查当前字符与上一个非空白字符不同
                if char_idx < len(characters): # 确保索引在字符映射表范围内防止索引越界错误
                    text.append(characters[char_idx]) # 将索引转换为实际字符添加到当前样本的结果列表
            prev_char = char_idx if char_idx != (NUM_CLASSES - 1) else None # 如果当前字符不是空白,则更新 prev_char否则重置为None
        decoded_texts.append(''.join(text)) # 将字符列表组合成字符串添加到批次结果列表

    return decoded_texts    # 输出形状为(batch_size, )

# 自定义回调函数用于监控正确率
class OCRMetrics(callbacks.Callback):
    def __init__(self, val_data, sample_size=200):
        super().__init__()
        self.val_images = val_data[0]   # 验证集图像
        self.val_labels = val_data[1]   # 验证集标签
        self.sample_size = min(sample_size, len(val_data[0]))   # 锁定最大样本大小
        self.history = {'val_cer': [], 'val_accuracy': []}      # 存储历史指标
        self.best_val_accuracy = 0.0    # 跟踪最佳准确率

    def on_epoch_end(self, epoch, logs=None):
        # 随机选择样本子集进行评估
        indices = np.random.choice(len(self.val_images), self.sample_size, replace=False)   # 从完整验证集中选择sample_size个样本, 且不重复选择
        sample_images = self.val_images[indices]    # 样本图像
        sample_labels = self.val_labels[indices]    # 样本标签

        # 预测
        preds = self.model.predict(sample_images, verbose=0)    # 使用当前模型进行预测 verbose=0表示不显示预测进度
        decoded_preds = decode_predictions(preds)               # 解码预测结果

        # 准备真实标签
        true_texts = []
        for labels in sample_labels:                # 过滤掉填充值 拼接为完整字符串
            # 过滤填充值并转换为字符
            text = ''.join([CHARACTERS[idx] for idx in labels if idx != -1])
            true_texts.append(text)

        # 计算字符错误率(CER - Character Error Rate)和序列准确率
        cer_total = 0
        char_count = 0
        correct_count = 0

        for true, pred in zip(true_texts, decoded_preds):   # 将真实标签(true - true_texts)和预测标签(pred - decoded_preds)进行匹配
            # 计算编辑距离
            distance = levenshtein_distance(true, pred)     # 使用 Levenshtein距离计算差异
            cer_total += distance
            char_count += len(true)                         # 字符总数

            # 检查完全匹配
            if true == pred:
                correct_count += 1

        # 计算指标
        val_cer = cer_total / max(1, char_count)            # CER = 总编辑距离 / 总字符数
        val_accuracy = correct_count / self.sample_size     # 衡量完全匹配的序列比例 = 完全匹配的序列数 / 总样本数

        # 保存指标
        self.history['val_cer'].append(val_cer)
        self.history['val_accuracy'].append(val_accuracy)

        # 更新最佳准确率
        if val_accuracy > self.best_val_accuracy:
            self.best_val_accuracy = val_accuracy           # 更新最佳准确率
            self.model.save('ocr_best_model.keras')         # 保存最佳模型

        # 打印指标
        print(f"\nEpoch {epoch + 1} - val_CER: {val_cer:.4f}, val_accuracy: {val_accuracy:.4f}")

        # 可视化一些结果
        self.visualize_results(sample_images[:5], true_texts[:5], decoded_preds[:5], epoch)

    def visualize_results(self, images, true_texts, pred_texts, epoch):
        # 创建包含最多5个子图的图形/显示原始图像(灰度)/在标题中显示真实文本和预测文本/添加主标题显示当前epoch/保存为图像文件/关闭图形释放内存
        plt.figure(figsize=(15, 2))             # 设置图形大小为15英寸宽×3英寸高
        for i in range(min(5, len(images))):    # 最多显示5个样本, 确保不超出可用样本数量
            plt.subplot(1, 5, i + 1)                                                # 创建子图布局,1行5列的子图网格,i+1是当前子图位置
            plt.imshow(images[i].squeeze(), cmap='gray')                                  # 获取第i个样本的图像数据,移除单维度,使用灰度图显示图像
            plt.title(f"True: {true_texts[i]}\nPred: {pred_texts[i]}", fontsize=10) # 设置子图标题和标题字体大小
            plt.axis('off')                     # 关闭坐标轴显示

        plt.suptitle(f"Epoch {epoch + 1} Predictions", fontsize=14) # 设置主标题和标题字体大小
        plt.tight_layout()                                            # 自动调整子图参数使它们适应图形区域 防止标题和标签重叠 优化子图间距
        plt.savefig(f'epoch_{epoch + 1}_predictions.png')             # 将图形保存为PNG文件
        plt.close()                                                   # 释放图形资源

    def plot_history(self):
        plt.figure(figsize=(12, 8))   # 设置图像窗口大小为12英寸宽x8英寸高

        # CER历史
        plt.subplot(2, 1, 1)    # 创建2行1列的子图网格 定位到第一个子图
        plt.plot(self.history['val_cer'], 'r-', label='Validation CER') # 获取验证CER值列表,红色实线样式,设置标签为Validation CER
        plt.title('Character Error Rate (CER)') # 设置子图标题为 Character Error Rate (CER)
        plt.xlabel('Epoch')                     # 设置x轴标签为"Epoch"（训练周期）
        plt.ylabel('CER')                       # 设置y轴标签为"CER"（字符错误率）
        plt.legend()                            # 显示图例（右上角的"Validation CER"标签）
        plt.grid(True)                          # 添加网格线，增强可读性

        # 准确率历史
        plt.subplot(2, 1, 2)    # 创建2行1列的子图网格 定位到第二个子图
        plt.plot(self.history['val_accuracy'], 'b-', label='Validation Accuracy')   # 从历史记录中获取验证准确率值列表，蓝色实线样式,设置标签为Validation Accuracy
        plt.title('Sequence Accuracy')          # 设置子图标题为 Sequence Accuracy
        plt.xlabel('Epoch')                     # 设置x轴标签为"Epoch"（训练周期）
        plt.ylabel('Accuracy')                  # 设置y轴标签为"Accuracy"（准确率）
        plt.ylim(0, 1)                    # 设置y轴范围为0到1
        plt.legend()                            # 显示图例
        plt.grid(True)                          # 添加网格线，增强可读性

        plt.tight_layout()                      # 自动调整子图参数 使它们适应图形区域
        plt.savefig('training_history.png')     # 将图形保存为PNG文件
        plt.show()                              # 显示生成的图表

# Levenshtein距离计算
def levenshtein_distance(s1, s2):
    """ Levenshtein距离(也称为编辑距离)是衡量两个字符串相似度的经典算法,用于计算将一个字符串转换为另一个字符串所需的最少单字符编辑操作次数。
        包含插入、删除、替换三种操作"""
    # s1:第一个字符串   s2:第二个字符串
    if len(s1) < len(s2):
        return levenshtein_distance(s2, s1)

    if len(s2) == 0:                    # 如果s2是空字符串,距离就是s1的长度
        return len(s1)

    previous_row = range(len(s2) + 1)   # 创建初始行,表示从空字符串转换为s2各前缀所需的操作数
    for i, c1 in enumerate(s1):         # i为索引值 c1为元素
        current_row = [i + 1]           #
        for j, c2 in enumerate(s2):     # j为索引值 c2为元素
            insertions = previous_row[j + 1] + 1
            deletions = current_row[j] + 1
            substitutions = previous_row[j] + (c1 != c2)
            current_row.append(min(insertions, deletions, substitutions))
        previous_row = current_row

    return previous_row[-1]

# 构建模型
print("构建CRNN模型...")
model = build_crnn_model()

# 编译模型并展示模型结构
model.compile(optimizer='adam', loss=ctc_loss)
model.summary()

# 准备回调函数
val_data = (X_test[:1000], y_test_padded[:1000])
metrics_callback = OCRMetrics(val_data, sample_size=200)

# 学习率调度器
lr_scheduler = callbacks.ReduceLROnPlateau(
    # 用于在训练过程中动态调整学习率
    monitor='val_loss', # 监控的指标
    factor=0.5,         # 学习率衰减因子
    patience=3,         # 等待的epoch数(如果验证损失在3个连续epoch内没有改善则触发学习率衰减)
    verbose=1,          # 日志输出级别(0为静默模式,1为显示调试信息)
    min_lr=1e-6         # 最小学习率下限
)

# 模型检查点
checkpoint = callbacks.ModelCheckpoint(
    # 用于在训练过程中保存模型检查点
    'ocr_checkpoint.keras',  # 保存模型的文件路径
    save_best_only=True,            # True只保存最好的模型 False会在每个epoch后保存模型
    monitor='val_loss',             # 指定要监控的指标名称
    mode='min',                     # 指定监控指标的模式 min表示越小越好
    verbose=1                       # 日志输出级别(0为静默模式,1为显示调试信息)
)

# 训练模型
print("开始训练模型...")
start_time = time.time()    # 模型开始训练的时间

history = model.fit(
    X_train, y_train_padded,                                # 输入数据和标签
    validation_data=(X_test[:5000], y_test_padded[:5000]),  # 手动指定的验证集
    batch_size=BATCH_SIZE,                                  # 每个批次的样本数
    epochs=EPOCHS,                                          # 训练步数
    callbacks=[metrics_callback, lr_scheduler, checkpoint]  # 回调函数
)

end_time = time.time()      # 模型结束训练的时间
print(f"训练完成，耗时: {end_time - start_time:.2f}秒")        # 输出模型训练所用的时间

# 可视化训练历史
metrics_callback.plot_history()

# 评估最佳模型
print("加载最佳模型进行评估...")
best_model = tf.keras.models.load_model('ocr_best_model.keras', custom_objects={'ctc_loss': ctc_loss})


# 在完整测试集上评估
def evaluate_model(model, images, labels, sample_size=1000):
    indices = np.random.choice(len(images), sample_size, replace=False) # 从整个测试集中随机选择指定数量的样本并且不重复
    sample_images = images[indices]     # 获取对应的图像
    sample_labels = labels[indices]     # 获取对应的标签

    # 预测
    preds = model.predict(sample_images, verbose=0) # 使用模型对采样图像进行预测
    decoded_preds = decode_predictions(preds)       # 将预测结果解码为可读的文本

    # 准备真实标签
    true_texts = []
    for labels in sample_labels:
        text = ''.join([CHARACTERS[idx] for idx in labels if idx != -1])
        true_texts.append(text)

    # 计算指标
    cer_total = 0
    char_count = 0
    correct_count = 0

    for true, pred in zip(true_texts, decoded_preds):
        distance = levenshtein_distance(true, pred)
        cer_total += distance
        char_count += len(true)

        if true == pred:
            correct_count += 1

    cer = cer_total / max(1, char_count)
    accuracy = correct_count / sample_size

    print(f"测试集评估结果 (样本数={sample_size}):")
    print(f"字符错误率 (CER): {cer:.4f}")
    print(f"序列准确率: {accuracy:.4f}")

    # 显示一些预测示例
    plt.figure(figsize=(15, 10))
    for i in range(min(10, len(sample_images))):
        plt.subplot(2, 5, i + 1)
        plt.imshow(sample_images[i].squeeze(), cmap='gray')
        plt.title(f"True: {true_texts[i]}\nPred: {decoded_preds[i]}", fontsize=9)
        plt.axis('off')

    plt.suptitle("测试集预测示例", fontsize=16)
    plt.tight_layout()
    plt.savefig('test_predictions.png')
    plt.show()

    return cer, accuracy

# 评估模型
test_cer, test_accuracy = evaluate_model(best_model, X_test, y_test_padded)

# 保存最终模型
best_model.save('ocr_final_model.keras')
print("最终模型已保存为 ocr_final_model.keras")