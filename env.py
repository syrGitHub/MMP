def set_device(dev):
    global _device
    _device = dev
def get_device():
    # return torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    return _device