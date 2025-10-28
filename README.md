# ==========================================================
# 如何一步步用Google Colaboratory訓練自己的模型？
# ==========================================================

# 本文介紹如何利用Google Colaboratory的免費GPU資源進行高效的深度學習模型訓練，
# 並對比了GPU與CPU在訓練過程中的性能差異。
# 使用Google Colaboratory必須翻牆

# 搞深度學習就必須要使用GPU，否則訓練太慢了，
# 自己用CPU訓練一個AlexNet網絡，資料量大了記憶體不夠，
# 資料量小模型訓練不夠，訓練時辦公也還特別卡，
# 搞了2天，結果訓練準確率才到0.5，
# 程式員嗎，遇山開山遇水搭橋，於是想個辦法突破自己電腦限制。

# 為啥選擇Colab呢？有以下幾個原因：
# 第一、自己搞的是TensorFlow（Pytorch也可以用它，後期也準備削它）；
# 第二，最重要的免費使用GPU，提高訓練模型的效率；
# 第三，不消耗自己電腦的pc，可以幹其他事情。
# 第四，不用配置環境。

# 如果你跟我一樣，也需要免費GPU來訓練模型的話，那麼一起Colab。


# 登錄Google雲端硬碟
# 單擊雲端硬碟，用Google帳號登錄。
# 在這裡插入圖片描述
# 單擊左上角新建，得到如下的界面：
# 在這裡插入圖片描述
# 你可以選擇上傳檔案或者資料夾，
# 你可以把訓練模型的資料集和配置等上傳上去，
# 上傳完之後，我們在點擊更多\Google Colaboratory，進入下面頁面：
# 在這裡插入圖片描述
# 單擊紅色框內按鈕，裝載Google雲端硬碟，或者運行下面代碼。

from google.colab import drive
drive.mount('/content/drive')

# python運行
# 掛載成功後，我們可以在右側中選擇content/drive/MyDrive，找到雲端硬碟，見下圖:
# 在這裡插入圖片描述


# 用GPU運行模型
# Colab就是一個編輯器，其格式為.ipynb，只不過它支援了TensorFlow，Pytorch等深度學習框架，
# 還提供了免費GPU，對於AI從業人員來說是福音，
# 但是要使用TPU的話就需要Colab pro，需要付費。
# 下面我開始構建模型和編譯模型。

# 單擊代碼/新建代碼單元格，開始寫自己的模型。
# 默認是CPU，若你想使用GPU，可以在代碼執行程序/更改運行時類型/CPU進行設置。

# 下面用個簡單的實例說明如何使用的：

# Simple CNN model for CIFAR-10  
import numpy  
from tensorflow.keras.datasets import cifar10  
from tensorflow.keras.layers import Conv2D  
from tensorflow.keras.layers import Dense  
from tensorflow.keras.layers import Dropout  
from tensorflow.keras.layers import Flatten  
from tensorflow.keras.layers import MaxPooling2D  
from tensorflow.keras.models import Sequential  
from tensorflow.keras.optimizers import SGD  
from tensorflow.python.keras.utils import np_utils  
  
# fix random seed for reproducibility  
seed = 7  
numpy.random.seed(seed)  
  
# load data  
(X_train, y_train), (X_test, y_test) = cifar10.load_data()  
  
# normalize inputs from 0-255 to 0.0-1.0  
X_train = X_train.astype("float32")  
X_test = X_test.astype("float32")  
X_train = X_train / 255.0  
X_test = X_test / 255.0  
  
# one hot encode outputs  
y_train = np_utils.to_categorical(y_train)  
y_test = np_utils.to_categorical(y_test)  
num_classes = y_test.shape[1]  
  
# Create the model  
model = Sequential()  
model.add(Conv2D(filters=32, kernel_size=(3, 3),  
                 input_shape=(32, 32, 3),  
                 padding="same",  
                 activation="relu"))  
model.add(Dropout(0.2))  
model.add(Conv2D(filters=32,  
                 kernel_size=(3, 3),  
                 activation="relu",  
                 padding="same"))  
model.add(MaxPooling2D(pool_size=(2, 2)))  
model.add(Flatten())  
model.add(Dense(512, activation="relu"))  
model.add(Dropout(0.5))  
model.add(Dense(num_classes, activation="softmax"))  
# Compile mode  
epochs = 25  
lrate = 0.01  
decay = lrate / epochs  
sgd = SGD(lr=lrate, momentum=0.9, decay=decay, nesterov=False)  
model.compile(loss="categorical_crossentropy", optimizer=sgd, metrics=["accuracy"])  
print(model.summary())  
  
# Fit the model  
model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=epochs,  
          batch_size=32, verbose=2)  
# Final evaluation of the model  
scores = model.evaluate(X_test, y_test, verbose=0)  
print("Accuracy: %.2f%%" % (scores[1] * 100))  

# python運行

# 採用CPU訓練結果如下，每步需要135ms，一個迭代需要211s。

# Epoch 1/25 1563/1563 - 211s - loss: 1.7122 - accuracy: 0.3809 - val_loss: 1.4475 - val_accuracy: 0.4919 - 211s/epoch - 135ms/step 
# Epoch 2/25 1563/1563 - 212s - loss: 1.3752 - accuracy: 0.5056 - val_loss: 1.2571 - val_accuracy: 0.5479 - 212s/epoch - 135ms/step

# json
# 接下來我們看看GPU模式下，每步耗時5ms，每個迭代耗時7s，
# 與CPU的211s相比，性能提升太多了。省下來的時間寫個代碼它不香嗎？

# Epoch 1/25 1563/1563 - 23s - loss: 1.7370 - accuracy: 0.3743 - val_loss: 1.4892 - val_accuracy: 0.4618 - 23s/epoch - 14ms/step 
# Epoch 2/25 1563/1563 - 7s - loss: 1.3938 - accuracy: 0.4956 - val_loss: 1.2771 - val_accuracy: 0.5397 - 7s/epoch - 5ms/step 
# Epoch 3/25 1563/1563 - 7s - loss: 1.2359 - accuracy: 0.5555 - val_loss: 1.1794 - val_accuracy: 0.5852 - 7s/epoch - 5ms/step 
# Epoch 4/25 1563/1563 - 7s - loss: 1.1201 - accuracy: 0.6021 - val_loss: 1.0750 - val_accuracy: 0.6226 - 7s/epoch - 4ms/step 
# Epoch 5/25 1563/1563 - 7s - loss: 1.0240 - accuracy: 0.6385 - val_loss: 1.0332 - val_accuracy: 0.6346 - 7s/epoch - 4ms/step 
# Epoch 6/25 1563/1563 - 7s - loss: 0.9512 - accuracy: 0.6615 - val_loss: 1.0049 - val_accuracy: 0.6417 - 7s/epoch - 4ms/step

# json

# 如果你覺得有更好的工具，不妨告訴，讓我也感受提升效率的快感！！！
# 歡迎點贊，收藏！
