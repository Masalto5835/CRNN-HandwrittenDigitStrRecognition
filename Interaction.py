import tkinter as tk
import tensorflow as tf
from tkinter import filedialog, messagebox, scrolledtext
import os
import numpy as np
from tensorflow import keras
from tensorflow.keras.preprocessing import image
from PIL import Image, ImageTk

NUM_CLASSES = 11            # 10个字符 + 1个空白类
CHARACTERS = "0123456789"   # 10个字符

model = keras.models.load_model('ocr_best_model.keras', compile=False)
image_references=[]

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

def process_file(filepath):
    try:
        img = image.load_img(filepath, target_size=(32,256), color_mode='grayscale')       # 加载图像并转换为灰度
        img_array = image.img_to_array(img)
        img_array = img_array[:, :, 0]
        img_array = img_array.reshape(1, 32, 256, 1)
        img_array = 1 - img_array.astype('float32') / 255.0

        prediction = model.predict(img_array, verbose=0)           # 进行模型推理
        predicted_digit = decode_predictions(prediction)
        result = [predicted_digit]

        return result
    except UnicodeDecodeError:
        return "错误：无法解码文本文件（请尝试其他编码）"
    except Exception as e:
        return f"处理错误: {str(e)}"


def main():
    # 创建主窗口
    root = tk.Tk()
    root.title("手写数字识别-CRNN模型")
    root.geometry("800x600")
    root.resizable(True, True)

    # 设置应用程序图标
    try:
        root.iconbitmap(default='CRNN.ico')  # 可替换为实际图标文件
    except:
        pass

    # 创建主框架
    main_frame = tk.Frame(root, padx=20, pady=20)
    main_frame.pack(fill=tk.BOTH, expand=True)

    # 文件选择区域
    file_frame = tk.LabelFrame(main_frame, text="文件选择", padx=10, pady=10)
    file_frame.pack(fill=tk.X, pady=(0, 10))

    file_path_var = tk.StringVar()
    tk.Entry(file_frame, textvariable=file_path_var, width=70).pack(
        side=tk.LEFT, padx=(0, 10), fill=tk.X, expand=True)

    def browse_file():
        filetypes = (
            [('图片文件.png', '*.png')]
        )
        filepath = filedialog.askopenfilename(
            title="选择图片",
            initialdir=os.path.expanduser('~'),  # 从用户主目录开始
            filetypes=filetypes
        )
        if filepath:
            file_path_var.set(filepath)

    tk.Button(file_frame, text="浏览目录", command=browse_file, width=10).pack(side=tk.RIGHT)

    # 处理按钮
    def process_and_show():
        filepath = file_path_var.get()
        if not filepath:
            messagebox.showwarning("输入错误", "请先选择文件")
            return
        if not os.path.isfile(filepath):
            messagebox.showerror("文件错误", f"文件不存在: {filepath}")
            return

        result = process_file(filepath)
        '''output_text.delete(1.0, tk.END)'''
        img = Image.open(filepath)
        img.thumbnail((200,200))
        test_image = ImageTk.PhotoImage(img)
        image_references.append(test_image)
        output_text.image_create(tk.END, image=test_image)
        output_text.insert(tk.END, '\n模型推理结果：')
        output_text.insert(tk.END, result[0])
        """
        output_text.insert(tk.END, '\n各结果预测概率：\n')
        for i in range(10):
            output_text.insert(tk.END, str(i)+'：')
            output_text.insert(tk.END, result[1][i])
            output_text.insert(tk.END, '\n')"""

    tk.Button(main_frame, text="进行推理", command=process_and_show, width=10).pack(pady=10)

    # 输出区域
    output_frame = tk.LabelFrame(main_frame, text="推理结果", padx=10, pady=10)
    output_frame.pack(fill=tk.BOTH, expand=True)

    output_text = scrolledtext.ScrolledText(output_frame, wrap=tk.WORD, font=("Consolas", 10))
    output_text.pack(fill=tk.BOTH, expand=True)

    # 运行主循环
    root.mainloop()


if __name__ == "__main__":
    main()