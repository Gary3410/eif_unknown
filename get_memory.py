import torch

# 检查是否有可用的GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 创建形状为 [800, 800, 1024] 的 tensor，类型为 float16
tensor = torch.randn((1000, 1000, 1024), dtype=torch.float16)

# 将 tensor 放到 GPU 上
tensor = tensor.to(device)

# 打印 tensor 的设备信息，确认已经在 GPU 上
print(tensor.device)

# 等待用户输入 'q' 后退出
while True:
    user_input = input("Enter 'q' to quit: ")
    if user_input.lower() == 'q':
        print("Exiting...")
        break