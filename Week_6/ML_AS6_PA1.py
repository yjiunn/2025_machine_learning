import xml.etree.ElementTree as ET
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns

# 讀取 XML 資料
tree = ET.parse("O-A0038-003.xml")
root = tree.getroot()
ns = {"cwa": "urn:cwa:gov:tw:cwacommon:0.1"}
content = root.find(".//cwa:Content", ns).text
values = [float(x) for x in content.replace("\n", ",").split(",") if x.strip()]

# 轉成 2D array (120x67)
grid = np.array(values).reshape(120, 67)
print("Grid shape:", grid.shape)

# 設定經緯度基準與偏移量
baseLong = 120
baseLat = 21.88
offset = 0.03

classification_data_simple = []
classification_labels = []
regression_data_simple = []
regression_values = []

row_num = grid.shape[0]
col_num = grid.shape[1]

long = []
lat = []

for i in range(row_num):
    for j in range(col_num):
        longitude = baseLong + j * offset
        latitude = baseLat + i * offset
        temperature = grid[i][j]
        
        # 分類資料
        label = 0 if temperature == -999.0 else 1
        classification_data_simple.append([longitude, latitude, longitude ** 2, latitude ** 2, latitude * longitude])
        classification_labels.append(label)
        
        # 回歸資料（僅有效值）
        if temperature != -999.0:
            long.append(longitude)
            lat.append(latitude)
            regression_data_simple.append([longitude, latitude])
            regression_values.append(temperature)


classification_data_simple = np.array(classification_data_simple)
classification_labels = np.array(classification_labels)
regression_data_simple = np.array(regression_data_simple)
regression_values = np.array(regression_values)


class GDA:
    def fit(self, X, y):
        m, n = X.shape
        self.phi = np.mean(y)

        X0 = X[y == 0]
        X1 = X[y == 1]
        self.mu0 = np.mean(X0, axis=0)
        self.mu1 = np.mean(X1, axis=0)

        self.Sigma = (
            np.dot((X0 - self.mu0).T, (X0 - self.mu0)) + 
            np.dot((X1 - self.mu1).T, (X1 - self.mu1))
        ) / m

        self.Sigma_inv = np.linalg.inv(self.Sigma)

    def predict(self, X):
        def log_likelihood(x, mu):
            return -0.5 * np.dot((x - mu).T, np.dot(self.Sigma_inv, (x - mu)))
        
        preds = []
        for x in X:
            logp0 = log_likelihood(x, self.mu0) + np.log(1 - self.phi)
            logp1 = log_likelihood(x, self.mu1) + np.log(self.phi)
            preds.append(1 if logp1 > logp0 else 0)
        return np.array(preds)
    

X_train, X_test, y_train, y_test = train_test_split(
    classification_data_simple, classification_labels, 
    test_size=0.2, random_state=42
)
model = GDA()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
accuracy = np.mean(y_pred == y_test)
print(f"準確率: {accuracy:.4f}")

print(classification_report(y_test, y_pred))

# ----------------------------------------------------
# 繪圖步驟 1: 建立完整的經緯度網格點
# ----------------------------------------------------

# 確保網格數據的經度和緯度範圍正確
long_min = baseLong
long_max = baseLong + (col_num - 1) * offset
lat_min = baseLat
lat_max = baseLat + (row_num - 1) * offset

# 創建經度和緯度的網格向量
long_coords = np.linspace(long_min, long_max, col_num)
lat_coords = np.linspace(lat_min, lat_max, row_num)

# 使用 np.meshgrid 創建網格點
xx, yy = np.meshgrid(long_coords, lat_coords)

# 將網格點攤平並創建原始 2D 特徵矩陣 X_grid_raw
X_grid_raw = np.c_[xx.ravel(), yy.ravel()]

# ----------------------------------------------------
# 繪圖步驟 2: 應用與訓練資料相同的多項式特徵轉換
# ----------------------------------------------------

X_grid_poly = np.hstack((
    X_grid_raw, 
    X_grid_raw[:, 0].reshape(-1, 1)**2,
    X_grid_raw[:, 1].reshape(-1, 1)**2,
    (X_grid_raw[:, 0] * X_grid_raw[:, 1]).reshape(-1, 1)
))

# ----------------------------------------------------
# 繪圖步驟 3: 預測網格標籤並繪製決策邊界
# ----------------------------------------------------

# 預測整個網格點的標籤
Z = model.predict(X_grid_poly)
# 將結果 reshape 回原始網格形狀 (120x67)
Z = Z.reshape(xx.shape)

plt.figure(figsize=(10, 8))

# 繪製決策邊界：使用顏色區域表示分類結果
# levels=1 表示在類別 0 和 1 之間劃分
# cmap='coolwarm' 或 'bwr' 用於區分兩個類別
plt.contourf(xx, yy, Z, levels=[0, 0.5, 1], cmap=plt.cm.RdYlBu, alpha=0.6)
# 也可以只繪製邊界線
# plt.contour(xx, yy, Z, levels=[0.5], colors='k', linestyles='-', linewidths=2)


# 繪製原始有效資料點 (標籤 1) 作為參考
# 使用原始分類資料集，只取標籤為 1 的點
valid_points = classification_data_simple[classification_labels == 1]
plt.scatter(valid_points[:, 0], valid_points[:, 1], c='black', marker='.', label='Valid Points (T != -999.0)')

plt.xlabel('Longtitude')
plt.ylabel('Latitude')
plt.title(f'GDA Decision Boundary - Accuracy: {accuracy:.4f}')
plt.legend()
plt.colorbar(ticks=[0.25, 0.75], format=plt.FuncFormatter(lambda x, pos: ['Invalid (0)', 'Valid (1)'][int(x > 0.5)]))
plt.axis('equal') # 保持經緯度比例一致
plt.show()

# ----------------------------------------------------
# 繪圖步驟 4: 繪製混淆矩陣
# ----------------------------------------------------

# 計算混淆矩陣
cm = confusion_matrix(y_test, y_pred)
labels = ['Invalid (0)', 'Valid(1)']

plt.figure(figsize=(6, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=labels, yticklabels=labels, cbar=False)
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.title(f'GDA Confusion Matrix (Accuracy: {accuracy:.4f})')
plt.show()


