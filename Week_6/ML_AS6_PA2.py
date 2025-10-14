import xml.etree.ElementTree as ET
import numpy as np
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
import matplotlib.pyplot as plt

# 設定 Matplotlib 使用英文，避免字體報錯
plt.rcParams['font.sans-serif'] = ['DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False


# ----------------------------------------------------
# 數據讀取與準備 (使用 Assignment 4 程式碼)
# ----------------------------------------------------
# 讀取 XML 資料
tree = ET.parse("O-A0038-003.xml")
root = tree.getroot()
ns = {"cwa": "urn:cwa:gov:tw:cwacommon:0.1"}
content = root.find(".//cwa:Content", ns).text
values = [float(x) for x in content.replace("\n", ",").split(",") if x.strip()]

# 轉成 2D array (120x67)
grid = np.array(values).reshape(120, 67)
row_num, col_num = grid.shape

# 設定經緯度基準與偏移量
baseLong = 120
baseLat = 21.88
offset = 0.03

classification_data_full = []
classification_labels = []
regression_data = []
regression_values = []
all_data_2d = [] # 用於最終繪圖的 Long 和 Lat 矩陣

for i in range(row_num):
    for j in range(col_num):
        longitude = baseLong + j * offset
        latitude = baseLat + i * offset
        temperature = grid[i][j]
        
        features = [longitude, latitude]
        
        # 收集所有點的分類資料
        classification_data_full.append(features)
        classification_labels.append(0 if temperature == -999.0 else 1)
        all_data_2d.append(features)
        
        # 收集有效點的回歸資料
        if temperature != -999.0:
            regression_data.append(features)
            regression_values.append(temperature)


X_full = np.array(classification_data_full)
y_clf = np.array(classification_labels)
X_reg = np.array(regression_data)
y_reg = np.array(regression_values)


# ----------------------------------------------------
# 模型訓練 (使用 Assignment 4 模型)
# ----------------------------------------------------

# 1. 分類模型 C(x→): 訓練 RandomForestClassifier
clf = RandomForestClassifier(n_estimators=100, random_state=42, max_depth=10)
clf.fit(X_full, y_clf) # 在所有資料上訓練分類器

# 2. 回歸模型 R(x→): 訓練 RandomForestRegressor
reg = RandomForestRegressor(n_estimators=100, random_state=42, max_depth=10)
reg.fit(X_reg, y_reg) # 僅在有效值 (陸地) 上訓練回歸器


# ----------------------------------------------------
# 實作組合模型 h(x→)
# ----------------------------------------------------

def h(X_input, classifier, regressor):
    """
    實作分段模型 h(x→):
    h(x→)={R(x→),if C(x→)=1
         {-999,if C(x→)=0
    
    X_input: 要預測的特徵矩陣 (N x 2), [Longitude, Latitude]
    classifier: 訓練好的 C(x→) 分類模型
    regressor: 訓練好的 R(x→) 回歸模型
    """
    
    # 步驟 1: 使用分類器 C(x→) 預測每個點的類別
    # C_pred = 1 (有效值/陸地) 或 0 (無效值/-999.0/海洋)
    C_pred = classifier.predict(X_input)
    
    # 初始化 h(x→) 的預測結果，預設為 -999.0
    h_pred = np.full(X_input.shape[0], -999.0)
    
    # 找到預測為有效值 (C_pred = 1) 的索引
    valid_indices = C_pred == 1
    
    # 步驟 2: 對預測為有效值的點，使用回歸器 R(x→) 進行溫度預測
    X_valid = X_input[valid_indices]
    
    if len(X_valid) > 0:
        # 進行回歸預測 R(x→)
        R_pred = regressor.predict(X_valid)
        
        # 將回歸結果填入 h_pred 對應的索引位置
        h_pred[valid_indices] = R_pred
        
    return h_pred


# 應用模型到完整的資料集
H_full_prediction = h(X_full, clf, reg)
print("\n組合模型 h(x→) 預測完成。")

# 驗證模型行為

# 1. 檢查預測為 C(x→)=0 (海洋) 的點
# 找到分類器預測為 0 的點的索引
invalid_indices_pred = (clf.predict(X_full) == 0)
# 檢查這些點在 H_full_prediction 中是否為 -999.0
h_pred_invalid_check = H_full_prediction[invalid_indices_pred]
# 應該全部是 -999.0
print("--------------------------------------------------")
print(f"預測為 C(x→)=0 的點數量: {np.sum(invalid_indices_pred)}")
print(f"h(x→) 在這些點上是否全部為 -999.0: {np.all(h_pred_invalid_check == -999.0)}")
print("--------------------------------------------------")

# 2. 檢查預測為 C(x→)=1 (陸地) 的點
# 找到分類器預測為 1 的點的索引
valid_indices_pred = (clf.predict(X_full) == 1)
# 檢查這些點是否都不是 -999.0
h_pred_valid_check = H_full_prediction[valid_indices_pred]
# 應該沒有 -999.0
print(f"預測為 C(x→)=1 的點數量: {np.sum(valid_indices_pred)}")
print(f"h(x→) 在這些點上是否有 -999.0: {np.any(h_pred_valid_check == -999.0)}")
print(f"h(x→) 在這些點上的最小預測溫度: {np.min(h_pred_valid_check):.2f}°C")
print("--------------------------------------------------")

# ----------------------------------------------------
# 模型行為可視化
# ----------------------------------------------------

# 將預測結果 H_full_prediction reshape 回 120x67 網格
H_grid = H_full_prediction.reshape(row_num, col_num)

# 創建經度和緯度的網格座標
long_coords = np.linspace(baseLong, baseLong + (col_num - 1) * offset, col_num)
lat_coords = np.linspace(baseLat, baseLat + (row_num - 1) * offset, row_num)
xx, yy = np.meshgrid(long_coords, lat_coords)


plt.figure(figsize=(10, 8))

# 繪製 H_grid
# 為了清楚展示溫度值，我們需要遮罩 -999.0 的部分
masked_H_grid = np.ma.masked_where(H_grid == -999.0, H_grid)

# 繪製陸地 (有效值) 的預測溫度等高線圖
im = plt.contourf(xx, yy, masked_H_grid, levels=20, cmap='viridis')
plt.colorbar(im, label='Predicted Temperature (°C)')

# 繪製海洋 (無效值) 區域 (可選: 塗上單一顏色)
ocean_mask = (H_grid == -999.0)
plt.pcolormesh(xx, yy, ocean_mask, cmap=plt.cm.gray_r, vmin=0, vmax=1, alpha=0.3)


plt.xlabel('Longitude')
plt.ylabel('Latitude')
plt.title(f'Combined Model h(x→) Prediction: Temperature & Masking')
plt.axis('equal') 
plt.legend(['Masked Ocean Area (h(x)= -999.0)'], loc='upper left')
plt.show()