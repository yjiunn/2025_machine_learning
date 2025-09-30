import xml.etree.ElementTree as ET
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error, confusion_matrix
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
        classification_data_simple.append([longitude, latitude])
        classification_labels.append(label)
        
        # 回歸資料（僅有效值）
        if temperature != -999.0:
            long.append(longitude)
            lat.append(latitude)
            regression_data_simple.append([longitude, latitude])
            regression_values.append(temperature)

#資料分類視覺化
plt.scatter(long, lat, c='black', marker='o')
plt.xlabel('Longtitude')
plt.ylabel('Latitude')
plt.title('Taiwan')
plt.show()

classification_data_simple = np.array(classification_data_simple)
classification_labels = np.array(classification_labels)
regression_data_simple = np.array(regression_data_simple)
regression_values = np.array(regression_values)

print(f"總格點數: {len(classification_data_simple)}")
print(f"有效值數量: {np.sum(classification_labels == 1)}")
print(f"無效值數量: {np.sum(classification_labels == 0)}")


# 分類模型 - Random Forest
X_train, X_test, y_train, y_test = train_test_split(
    classification_data_simple, classification_labels, 
    test_size=0.2, random_state=42
)

clf = RandomForestClassifier(n_estimators=100, random_state=42, max_depth=10)
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)

print("===================")
print(f"準確率: {np.mean(y_pred == y_test):.4f}")
print("===================")


#混淆矩陣 
cm = confusion_matrix(y_test, y_pred)
print("混淆矩陣:")
print(cm)
plt.figure(figsize=(10,6))
sns.heatmap(cm, square=True, annot=True, fmt='d', linecolor='white', cmap='RdBu', linewidths=1.5, cbar=False)
plt.xlabel('Pred')
plt.ylabel("True")
plt.show()

# 回歸模型 - Random Forest
if len(regression_data_simple) > 0:
    X_train_reg, X_test_reg, y_train_reg, y_test_reg = train_test_split(
        regression_data_simple, regression_values, 
        test_size=0.2, random_state=42
    )
    
    reg = RandomForestRegressor(n_estimators=100, random_state=42, max_depth=10)
    reg.fit(X_train_reg, y_train_reg)
    y_pred_reg = reg.predict(X_test_reg)
    
    print(f"\n[回歸] Random Forest Regressor")
    print(f"RMSE: {np.sqrt(mean_squared_error(y_test_reg, y_pred_reg)):.4f} °C")
    print(f"MAE: {mean_absolute_error(y_test_reg, y_pred_reg):.4f} °C")
    print(f"R²: {r2_score(y_test_reg, y_pred_reg):.4f}")
