# 如何一步步用 Google Colaboratory 訓練自己的模型？

本文介紹如何利用 Google Colaboratory 的免費 GPU 資源進行高效的深度學習模型訓練，並對比了 GPU 與 CPU 在訓練過程中的性能差異。  
> ⚠️ 使用 Google Colaboratory 必須翻牆。

---

## 為什麼要使用 GPU？

搞深度學習就必須要使用 GPU，否則訓練太慢了。  
自己用 CPU 訓練一個 AlexNet 網絡，資料量大了記憶體不夠，資料量小模型訓練不夠。  
訓練時電腦也特別卡，搞了 2 天，結果訓練準確率才到 0.5。  
程式員嘛，遇山開山、遇水搭橋，於是想個辦法突破自己電腦限制。

---

## 為啥選擇 Colab 呢？

有以下幾個原因：

1. 自己搞的是 TensorFlow（Pytorch 也可以用它，後期也準備學它）；
2. 最重要的是免費使用 GPU，提高訓練模型的效率；
3. 不消耗自己電腦的資源，可以幹其他事情；
4. 不用配置環境。

如果你跟我一樣，也需要免費 GPU 來訓練模型的話，那麼一起用 Colab 吧。

---

## 登錄 Google 雲端硬碟

單擊雲端硬碟，用 Google 帳號登錄。界面如下： 
<img width="1505" height="980" alt="image" src="https://github.com/user-attachments/assets/a65e7e80-7d8c-45e0-9c03-b91f0740b6be" />
單擊左上角新建，得到如下的界面：
<img width="1502" height="974" alt="image" src="https://github.com/user-attachments/assets/b9d28ed5-598a-400e-86e0-5f56da90cf5d" />
你可以選擇或者，你可以把訓練模型的數據集和配置等上傳上去，上傳完之後，我們在點擊，進入下面頁面：上传文件文件夹更多\Google Colaboratory
<img width="1919" height="842" alt="image" src="https://github.com/user-attachments/assets/38e02c47-1235-4026-8f15-3705f9162098" />

單擊紅色框內按鈕，裝載 Google 雲端硬碟，或者運行下面代碼：

```python
from google.colab import drive
drive.mount('/content/drive')
```
掛載成功后，我們可以在右側種選擇，找到雲端硬碟，見下圖：content/drive/MyDrive
<img width="1917" height="911" alt="螢幕擷取畫面 2025-10-28 154148" src="https://github.com/user-attachments/assets/e65424ac-1176-4530-b621-6274610a1d9f" />

# 用GPU運行模型
Colab就是一個編輯器，其格式為.ipynb，只不過它支援了TensorFlow，Pytorch等深度學習框架，
還提供了免費GPU，對於AI從業人員來說是福音，但是要使用TPU的話就需要Colab pro，需要付費。
下面我開始構建模型和編譯模型。

單擊代碼/新建代碼單元格，開始寫自己的模型。默認是CPU，若你想使用GPU，可以在代碼執行程序/更改運行時類型/CPU進行設置。
下面用個簡單的實例說明如何使用的：

```pyton
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
from tensorflow.keras.utils import to_categorical

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
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)
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
sgd = SGD(learning_rate=lrate, momentum=0.9, decay=decay, nesterov=False)
model.compile(loss="categorical_crossentropy", optimizer=sgd, metrics=["accuracy"])
print(model.summary())

# Fit the model
model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=epochs,
          batch_size=32, verbose=2)
# Final evaluation of the model
scores = model.evaluate(X_test, y_test, verbose=0)
print("Accuracy: %.2f%%" % (scores[1] * 100))
```

採用CPU訓練結果如下，每步都需要120ms以上，一個迭代需要211s。
```
Epoch 1/25
1563/1563 - 196s - 125ms/step - accuracy: 0.3834 - loss: 1.6963 - val_accuracy: 0.5029 - val_loss: 1.3784

Epoch 2/25
1563/1563 - 207s - 132ms/step - accuracy: 0.5180 - loss: 1.3604 - val_accuracy: 0.5411 - val_loss: 1.2707
```
接下來我們看看GPU模式下，每步耗時5ms，每個迭代耗時7s，與CPU的211s相比，性能提升太多了。省下來的時間寫個代碼它不香嗎？
```
Epoch 1/25
1563/1563 - 13s - 8ms/step - accuracy: 0.3633 - loss: 1.7546 - val_accuracy: 0.4874 - val_loss: 1.4514
Epoch 2/25
1563/1563 - 7s - 4ms/step - accuracy: 0.4883 - loss: 1.4132 - val_accuracy: 0.5351 - val_loss: 1.2998
Epoch 3/25
1563/1563 - 6s - 4ms/step - accuracy: 0.5599 - loss: 1.2361 - val_accuracy: 0.5818 - val_loss: 1.1777
Epoch 4/25
1563/1563 - 7s - 4ms/step - accuracy: 0.6108 - loss: 1.0985 - val_accuracy: 0.6122 - val_loss: 1.1027
Epoch 5/25
1563/1563 - 6s - 4ms/step - accuracy: 0.6535 - loss: 0.9824 - val_accuracy: 0.6252 - val_loss: 1.0640
Epoch 6/25
1563/1563 - 7s - 4ms/step - accuracy: 0.6904 - loss: 0.8805 - val_accuracy: 0.6575 - val_loss: 0.9684
Epoch 7/25
1563/1563 - 7s - 4ms/step - accuracy: 0.7228 - loss: 0.7910 - val_accuracy: 0.6657 - val_loss: 0.9692
Epoch 8/25
1563/1563 - 6s - 4ms/step - accuracy: 0.7504 - loss: 0.7076 - val_accuracy: 0.6742 - val_loss: 0.9804
Epoch 9/25
1563/1563 - 7s - 4ms/step - accuracy: 0.7776 - loss: 0.6374 - val_accuracy: 0.6723 - val_loss: 1.0063
Epoch 10/25
1563/1563 - 6s - 4ms/step - accuracy: 0.7962 - loss: 0.5828 - val_accuracy: 0.6683 - val_loss: 1.0513
Epoch 11/25
1563/1563 - 7s - 4ms/step - accuracy: 0.8162 - loss: 0.5262 - val_accuracy: 0.6772 - val_loss: 1.0628
Epoch 12/25
1563/1563 - 6s - 4ms/step - accuracy: 0.8335 - loss: 0.4773 - val_accuracy: 0.6776 - val_loss: 1.0634
Epoch 13/25
1563/1563 - 7s - 4ms/step - accuracy: 0.8470 - loss: 0.4405 - val_accuracy: 0.6866 - val_loss: 1.1001
Epoch 14/25
1563/1563 - 6s - 4ms/step - accuracy: 0.8596 - loss: 0.4035 - val_accuracy: 0.6878 - val_loss: 1.1317
Epoch 15/25
1563/1563 - 7s - 4ms/step - accuracy: 0.8669 - loss: 0.3843 - val_accuracy: 0.6785 - val_loss: 1.2197
Epoch 16/25
1563/1563 - 6s - 4ms/step - accuracy: 0.8786 - loss: 0.3521 - val_accuracy: 0.6765 - val_loss: 1.2206
Epoch 17/25
1563/1563 - 7s - 4ms/step - accuracy: 0.8843 - loss: 0.3373 - val_accuracy: 0.6781 - val_loss: 1.2941
Epoch 18/25
1563/1563 - 7s - 4ms/step - accuracy: 0.8894 - loss: 0.3247 - val_accuracy: 0.6805 - val_loss: 1.3519
Epoch 19/25
1563/1563 - 7s - 4ms/step - accuracy: 0.8980 - loss: 0.3029 - val_accuracy: 0.6761 - val_loss: 1.2920
Epoch 20/25
1563/1563 - 7s - 4ms/step - accuracy: 0.8978 - loss: 0.3021 - val_accuracy: 0.6784 - val_loss: 1.3887
Epoch 21/25
1563/1563 - 6s - 4ms/step - accuracy: 0.9059 - loss: 0.2779 - val_accuracy: 0.6786 - val_loss: 1.5030
Epoch 22/25
1563/1563 - 7s - 4ms/step - accuracy: 0.9062 - loss: 0.2808 - val_accuracy: 0.6850 - val_loss: 1.3259
Epoch 23/25
1563/1563 - 6s - 4ms/step - accuracy: 0.9103 - loss: 0.2665 - val_accuracy: 0.6832 - val_loss: 1.3664
Epoch 24/25
1563/1563 - 7s - 4ms/step - accuracy: 0.9161 - loss: 0.2535 - val_accuracy: 0.6784 - val_loss: 1.4950
Epoch 25/25
1563/1563 - 6s - 4ms/step - accuracy: 0.9166 - loss: 0.2526 - val_accuracy: 0.6783 - val_loss: 1.4801
Accuracy: 67.83%
```
