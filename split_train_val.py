import random
import glob
import shutil

lr_files = glob.glob('./data/wv3_to_k3a_pairs/train/fake_lr/*.tif')
random.shuffle(lr_files)
size = len(lr_files)
val_size = 0.1 * size
i = 0
for lr_file in lr_files:
    hr_file = lr_file.replace('/fake_lr/', '/hr/')
    dst_lr_file = lr_file.replace('/train/', '/val/')
    dst_hr_file = hr_file.replace('/train/', '/val/')
    shutil.move(lr_file, dst_lr_file)
    shutil.move(hr_file, dst_hr_file)
    i += 1
    if i >= val_size:
        break
    







