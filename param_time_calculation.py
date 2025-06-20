import torch
from funcs.model.DSCNet import DSCNet
from torchinfo import summary
import time

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

for rate in 8, 24, 72, 144, 216:
    model = DSCNet(layer_depth=3, rate=rate, ga="MiT").to(device)

    for input_shape in [(1,3,400,400)]: # (1,3,400,400)
        
        print("rate, input_shape:", rate, input_shape)
        input_sample = torch.randn(input_shape).to(device)

        summary(model, input_size=input_shape) 


        model.eval()
        # 单个样本
        with torch.no_grad():
            start = time.time()
            for _ in range(1000):
                output = model(input_sample)
            end = time.time()
        print(f"预测时间: {round((end - start), 2)} ms")

for depth in 2, 3, 4, 5:
    model = DSCNet(layer_depth=depth, rate=72, ga="MiT").to(device)

    for input_shape in [(1,3,304,304), (1,3,400,400)]:
        
        print("depth, input_shape:", depth, input_shape)
        input_sample = torch.randn(input_shape).to(device)

        summary(model, input_size=input_shape) 

        model.eval()
        # 单个样本
        with torch.no_grad():
            start = time.time()
            for _ in range(1000):
                output = model(input_sample)
            end = time.time()
        print(f"预测时间: {round((end - start), 2)} ms")