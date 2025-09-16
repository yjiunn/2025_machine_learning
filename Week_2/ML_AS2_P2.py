import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

# 1. 定義 Runge 函數
def runge_function(x):
    """
    計算 Runge 函數的值：f(x) = 1 / (1 + 25x^2)
    """
    return 1 / (1 + 25 * x**2)

# 2. 準備資料
# 在 [-1, 1] 區間內生成均勻分佈的 x 值
num_samples = 1000
x = np.linspace(-1, 1, num_samples).reshape(-1, 1)

# 計算對應的 Runge 函數 y 值
y = runge_function(x)

# 將數據轉換為 PyTorch Tensor
X_tensor = torch.tensor(x, dtype=torch.float32)
y_tensor = torch.tensor(y, dtype=torch.float32)

# 將資料分割為訓練集和驗證集（80% 訓練, 20% 驗證）
X_train, X_val, y_train, y_val = train_test_split(X_tensor, y_tensor, test_size=0.2, random_state=42)

# 3. 定義神經網路模型
class RungeApproximator(nn.Module):
    """
    一個簡單的前饋神經網路模型，用來近似 Runge 函數。
    包含兩個隱藏層，使用 ReLU 激活函數。
    """
    def __init__(self):
        super(RungeApproximator, self).__init__()
        self.layer_1 = nn.Linear(1, 64)  # 輸入層：1個輸入 (x)，64個神經元
        self.layer_2 = nn.Linear(64, 64) # 隱藏層：64個神經元
        self.layer_3 = nn.Linear(64, 1)  # 輸出層：1個輸出 (y)
    
    def forward(self, x):
        x = torch.relu(self.layer_1(x))
        x = torch.relu(self.layer_2(x))
        x = self.layer_3(x)
        return x

# 4. 訓練模型
def train_model(model, X_train, y_train, X_val, y_val, num_epochs=1000, learning_rate=0.001):
    """
    訓練神經網路模型並記錄訓練與驗證損失。
    """
    criterion = nn.MSELoss()  # 使用均方誤差作為損失函數
    optimizer = optim.Adam(model.parameters(), lr=learning_rate) # 使用 Adam 優化器

    train_losses = []
    val_losses = []

    for epoch in range(num_epochs):
        # 訓練階段
        model.train() # 設定為訓練模式
        optimizer.zero_grad()
        y_pred = model(X_train)
        loss = criterion(y_pred, y_train)
        loss.backward()
        optimizer.step()

        train_losses.append(loss.item())

        # 驗證階段
        model.eval() # 設定為評估模式
        with torch.no_grad():
            y_val_pred = model(X_val)
            val_loss = criterion(y_val_pred, y_val)
            val_losses.append(val_loss.item())

        if (epoch + 1) % 100 == 0:
            print(f'Epoch [{epoch+1}/{num_epochs}], Training Loss: {loss.item():.6f}, Validation Loss: {val_loss.item():.6f}')
    
    return train_losses, val_losses

# 5. 執行訓練與評估
model = RungeApproximator()
train_losses, val_losses = train_model(model, X_train, y_train, X_val, y_val)

# 6. 繪製圖表與計算誤差
# 在完整區間 [-1, 1] 進行預測以獲得平滑曲線
model.eval()
with torch.no_grad():
    y_pred_all = model(X_tensor).numpy()

# 將 Tensor 轉換為 numpy 陣列以便計算誤差
y_true = y.flatten()
y_pred = y_pred_all.flatten()

# 計算均方誤差 (MSE)
mse = mean_squared_error(y_true, y_pred)
# 計算最大誤差 (Max Error)
max_error = np.max(np.abs(y_true - y_pred))

# 繪製真函數與神經網路預測曲線
### 真實函數與預測曲線比較

plt.figure(figsize=(10, 6))
plt.plot(x, y, label='True Runge Function', color='blue', linewidth=2)
plt.plot(x, y_pred_all, label='Neural Network Prediction', color='red', linestyle='--', linewidth=2)
plt.title('Runge Function Approximation by Neural Network')
plt.xlabel('x')
plt.ylabel('f(x)')
plt.legend()
plt.grid(True)
plt.show()

# 繪製訓練/驗證損失曲線
### 訓練與驗證損失曲線

plt.figure(figsize=(10, 6))
plt.plot(train_losses, label='Training Loss', color='green')
plt.plot(val_losses, label='Validation Loss', color='orange')
plt.title('Training and Validation Loss over Epochs')
plt.xlabel('Epochs')
plt.ylabel('Loss (MSE)')
plt.legend()
plt.grid(True)
plt.yscale('log') # 使用對數坐標可以更清楚地看到損失下降
plt.show()

# 繪製誤差分佈曲線
### 誤差分佈視覺化

# 計算絕對誤差
abs_errors = np.abs(y_true - y_pred)

plt.figure(figsize=(10, 6))
plt.plot(x, abs_errors, label='Absolute Error', color='purple')
plt.title('Absolute Error across the Domain [-1, 1]')
plt.xlabel('x')
plt.ylabel('Absolute Error |f(x) - f_pred(x)|')
plt.legend()
plt.grid(True)
plt.show()

# 報告最終誤差數值
### 誤差報告

print("--------------------------------------------------")
print("Reported Errors:")
print(f"Mean Squared Error (MSE): {mse:.6f}")
print(f"Max Error: {max_error:.6f}")
print("--------------------------------------------------")