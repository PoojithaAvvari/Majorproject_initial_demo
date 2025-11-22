import torch
print("torch:", torch.__version__)
print("cuda available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("device name:", torch.cuda.get_device_name(0))
    x = torch.randn(1024, 1024).to('cuda')
    print("tensor on cuda:", x.device)
