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


