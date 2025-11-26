import importlib, torch
print("torch:", torch.__version__)
print("cuda available:", torch.cuda.is_available())
print("gpu:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "No GPU")
spec = importlib.util.find_spec("bitsandbytes")
if spec:
    import bitsandbytes as bnb
    print("bitsandbytes:", bnb.__version__)
else:
    print("bitsandbytes: not installed")
