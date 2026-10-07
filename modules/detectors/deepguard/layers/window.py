def window_partition(x, window_size):
    B,H,W,C = x.shape
    x = x.view(B, H//window_size, window_size, W//window_size, window_size, C)
    
    windows = x.permute(0,1,3,2,4,5).reshape(-1, window_size, window_size, C)
    return windows

def window_reverse(windows, window_size, H , W):
    
    C = windows.shape[-1]
    x = windows.view(-1, H//window_size, W//window_size, window_size, window_size, C)
    x = x.permute(0,1,3,2,4,5).reshape(-1,H,W,C)
    
    return x
