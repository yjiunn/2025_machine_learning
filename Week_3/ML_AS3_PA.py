import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

# 1. 定義 Runge 函數及其導數
def runge_function(x):
    """
    計算 Runge 函數的值：f(x) = 1 / (1 + 25x^2)
    """
    return 1 / (1 + 25 * x**2)

def runge_derivative(x):
    """
    計算 Runge 函數的解析導數：f'(x) = -50x / (1 + 25x^2)^2
    """
    return -50 * x / (1 + 25 * x**2)**2

# 2. 準備數據
num_samples = 1000
x = np.linspace(-1, 1, num_samples).reshape(-1, 1)

y = runge_function(x)
y_prime = runge_derivative(x)

# 將數據轉換為 PyTorch Tensor，並設置 requires_grad=True 來追蹤梯度
X_tensor = torch.tensor(x, dtype=torch.float32, requires_grad=True)
y_tensor = torch.tensor(y, dtype=torch.float32)
y_prime_tensor = torch.tensor(y_prime, dtype=torch.float32)

# 將資料分割為訓練集和驗證集
X_train, X_val, y_train, y_val, y_prime_train, y_prime_val = train_test_split(
    X_tensor, y_tensor, y_prime_tensor, test_size=0.2, random_state=42
)

# 3. 定義神經網路模型
class RungeApproximator(nn.Module):
    def __init__(self):
        super(RungeApproximator, self).__init__()
        self.layer_1 = nn.Linear(1, 64)
        self.layer_2 = nn.Linear(64, 64)
        self.layer_3 = nn.Linear(64, 1)
    
    def forward(self, x):
        x = torch.relu(self.layer_1(x))
        x = torch.relu(self.layer_2(x))
        x = self.layer_3(x)
        return x

# 4. 訓練模型
model = RungeApproximator()
optimizer = optim.Adam(model.parameters(), lr=0.001)
num_epochs = 2000 # 增加訓練次數以達到更好的收斂

# 定義複合損失函數的權重
lambda_prime = 1.0 # 可以調整此值來平衡函數損失和導數損失

train_losses = []
val_losses = []
function_losses = []
derivative_losses = []

print("Starting model training with dual loss function...")
for epoch in range(num_epochs):
    model.train()
    optimizer.zero_grad()
    
    # 函數預測
    y_pred = model(X_train)
    
    # 函數損失 (Function loss)
    function_loss = nn.MSELoss()(y_pred, y_train)
    
    # 計算導數預測
    y_pred_prime = torch.autograd.grad(
        outputs=y_pred, 
        inputs=X_train, 
        grad_outputs=torch.ones_like(y_pred), 
        create_graph=True,
        retain_graph=True
    )[0]
    
    # 導數損失 (Derivative loss)
    derivative_loss = nn.MSELoss()(y_pred_prime, y_prime_train)
    
    # 總損失 (Total loss)
    total_loss = function_loss + lambda_prime * derivative_loss
    
    # 反向傳播
    total_loss.backward(retain_graph=True)
    optimizer.step()

    # 驗證階段
    model.eval()
    # 移除 with torch.no_grad(): 區塊，並使用 torch.enable_grad() 來確保梯度追蹤開啟
    with torch.enable_grad():
        y_val_pred = model(X_val)
        val_function_loss = nn.MSELoss()(y_val_pred, y_val)
        
        y_val_prime_pred = torch.autograd.grad(
            outputs=y_val_pred.sum(), 
            inputs=X_val, 
            create_graph=False
        )[0]
        val_derivative_loss = nn.MSELoss()(y_val_prime_pred, y_prime_val)

    # 記錄損失
    train_losses.append(total_loss.item())
    val_losses.append(val_function_loss.item() + lambda_prime * val_derivative_loss.item())
    function_losses.append(function_loss.item())
    derivative_losses.append(derivative_loss.item())

    if (epoch + 1) % 100 == 0:
        print(f'Epoch [{epoch+1}/{num_epochs}], Total Loss: {total_loss.item():.6f}, '
              f'Function Loss: {function_loss.item():.6f}, Derivative Loss: {derivative_loss.item():.6f}')
print("Training complete.")

# 5. 繪製圖表與報告
model.eval()
with torch.enable_grad():
    y_pred_all = model(X_tensor)
    
    y_prime_pred_all = torch.autograd.grad(
        outputs=y_pred_all.sum(), 
        inputs=X_tensor, 
        create_graph=False
    )[0].detach().numpy().flatten()
    
    y_pred_all = y_pred_all.detach().numpy().flatten()

true_y = y_tensor.detach().numpy().flatten()
true_y_prime = y_prime_tensor.detach().numpy().flatten()

# 計算最終誤差
mse_y = mean_squared_error(true_y, y_pred_all)
max_error_y = np.max(np.abs(true_y - y_pred_all))

mse_y_prime = mean_squared_error(true_y_prime, y_prime_pred_all)
max_error_y_prime = np.max(np.abs(true_y_prime - y_prime_pred_all))

### 1. 真實函數與神經網路預測曲線

plt.figure(figsize=(10, 6))
plt.plot(x, true_y, label='True Function', color='blue', linewidth=2)
plt.plot(x, y_pred_all, label='NN Prediction', color='red', linestyle='--', linewidth=2)
plt.title('Approximation of the Runge Function')
plt.xlabel('x')
plt.ylabel('f(x)')
plt.legend()
plt.grid(True)
plt.show()

### 2. 真實導數與神經網路導數預測曲線

plt.figure(figsize=(10, 6))
plt.plot(x, true_y_prime, label='True Derivative', color='blue', linewidth=2)
plt.plot(x, y_prime_pred_all, label='NN Derivative Prediction', color='red', linestyle='--', linewidth=2)
plt.title('Approximation of the Runge Function Derivative')
plt.xlabel('x')
plt.ylabel("f'(x)")
plt.legend()
plt.grid(True)
plt.show()

### 3. 訓練/驗證損失曲線

plt.figure(figsize=(10, 6))
plt.plot(train_losses, label='Total Training Loss', color='green')
plt.plot(val_losses, label='Total Validation Loss', color='orange')
plt.title('Total Training and Validation Loss over Epochs')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.grid(True)
plt.yscale('log')
plt.show()

### 4. 最終誤差報告

print("--------------------------------------------------")
print("Final Errors:")
print("\n--- Function Approximation ---")
print(f"Mean Squared Error (MSE): {mse_y:.6f}")
print(f"Max Error: {max_error_y:.6f}")
print("\n--- Derivative Approximation ---")
print(f"Mean Squared Error (MSE): {mse_y_prime:.6f}")
print(f"Max Error: {max_error_y_prime:.6f}")
print("--------------------------------------------------")