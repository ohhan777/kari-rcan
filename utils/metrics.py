def fitness(x):
    # Model fitness as a weighted combination of metrics
    w = [1.0, 0.0]  # weights for [psnr, ssim] 
    return (x * w).sum(0)