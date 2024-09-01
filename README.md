### Kohonen Self-Organizing Feature Map (SOFM)

#### 摘要

這是一個實作 Kohonen Self-Organizing Feature Map (SOFM) 的 Python 程式碼，用於資料分類和視覺化。在這個專案中，我們使用了一個包含葡萄酒相關屬性的數據集，並將其分成兩個部分：訓練集和測試集。

#### Kohonen 網路訓練

```python
# 在 kohonen.ipynb 中的程式碼片段
import numpy as np
import pandas as pd

# ... （省略部分程式碼）

# 初始化 Kohonen 神經網路
kohonen_layer = 5
k1 = kohonen()
k1.initial(feature_dim, kohonen_layer)

# 訓練 Kohonen 神經網路
for index, row in data.iterrows():
    max_value, max_index = k1.forward(row.values)
    k1.backward(max_value, max_index, row.values, 0.1)

# 輸出權重矩陣
print(k1.W)
```

#### Self-Organizing Feature Map (SOFM) 分類

```python
# 在 prhw3.ipynb 中的程式碼片段
import numpy as np
import pandas as pd

# ... （省略部分程式碼）

# 初始化 SOFM
feature_dim = 13
sofm_layer = 100
sofm = SOFM()
sofm.initial(feature_dim, sofm_layer)
winner_counts = np.zeros((sofm.grid_size, sofm.grid_size))

# 訓練 SOFM
sofm.forward(train_features, winner_counts)
sofm_output = sofm.forward(test_features, winner_counts)

# 輸出結果
print("SOFM輸出：")
print(sofm_output)
print(sofm.labelgrid)
```

### 視覺化結果

我們使用了 matplotlib 繪製了 SOFM 的熱力圖，顯示了每個單元格的輸出和相應的數字標籤。

```python
# 在 prhw3.ipynb 中的程式碼片段
import matplotlib.pyplot as plt

# ... （省略部分程式碼）

# 繪製SOFM輸出的熱力圖和標籤
plt.subplot(1, 2, 1)
plt.title('SOFM Output')
plt.imshow(sofm_output, cmap='hot', interpolation='nearest')
plt.colorbar()

# 在每個單元格中顯示數字標籤
for i in range(sofm_output.shape[0]):
    for j in range(sofm_output.shape[1]):
        label = str(int(sofm_output[i][j])) + ',' + ','.join(map(str, sofm.labelgrid[i][j])) if sofm.labelgrid[i][j] else str(int(sofm_output[i][j]))
        plt.text(j, i, label, ha='center', va='center', color='black')

# 調整子圖之間的間距
plt.tight_layout()

# 顯示圖形
plt.show()
```
