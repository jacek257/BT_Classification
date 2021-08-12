from data_tools.itk import *
from multiprocessing import Pool
from tqdm import tqdm 
import sys
import os
import random

if __name__ == "__main__":
    
    # interpret args
    if len(sys.argv) > 1:
        num_processes = int(sys.argv[1])
    else:
        num_processes = os.cpu_count() - 1
    
        
    print(f"Preprocess using: {num_processes} cores")
    
    # define constants 
    flair_mean_spacings = [0.7121478558063163, 0.8886607742498911, 2.4320288976061724]
    t1w_mean_spacings = [0.8630510661421857, 0.7675369660980376, 2.631836211367285]
    t1wce_mean_spacings = [0.8768679856749232, 0.884927486136288, 1.7342040534353473]
    t2w_mean_spacings = [0.6834752597955375, 0.7160193959823881, 2.5539125597552883]
    
    # generate dirs
    TRAIN_DIR = "./data/train/"
    OUTPUT_DIR = "./processed/train/"
    safe_make_dir(OUTPUT_DIR)
    
    # get images
    IMAGE_DIRS = get_dir_dict(TRAIN_DIR)
    
    processed_dir = "./processed/train/"
    safe_make_dir(processed_dir)
    
    # generate input lists
    flair = [(f, processed_dir, flair_mean_spacings)
              for f in IMAGE_DIRS["flair"]]
    t1w = [(f, processed_dir, t1w_mean_spacings)
           for f in IMAGE_DIRS["t1w"]]
    t1wce = [(f, processed_dir, t1wce_mean_spacings)
             for f in IMAGE_DIRS["t1wce"]]
    t2w = [(f, processed_dir, t2w_mean_spacings)
           for f in IMAGE_DIRS["t2w"]]
    
    # testing mods (REMOVE-START)
#     flair = random.sample(flair, 10)
#     gobble = [f[0] for f in flair]
#     for g in gobble:
#         print(g)
    # REMOVE-END

    # process flair
    with Pool(num_processes) as p:
        list(tqdm(p.imap(map_safe_process, flair), total=len(flair),
            desc="flair"))
    # process t1w    
    with Pool(num_processes) as p: 
        list(tqdm(p.imap(map_safe_process, t1w), total=len(flair),
            desc="t1w  "))
    # process t1wce    
    with Pool(num_processes) as p:
        list(tqdm(p.imap(map_safe_process, t1wce), total=len(flair),
            desc="t1wce"))
    # process t2w
    with Pool(num_processes) as p:
        list(tqdm(p.imap(map_safe_process, t2w), total=len(flair),
            desc="t2w. "))
 