from data_tools.itk import *
from multiprocessing import Pool
from tqdm import tqdm 

if __name__ == "__main__":
    # generate dirs
    TRAIN_DIR = "./data/train/"
    OUTPUT_DIR = "./processed/train/"
    safe_make_dir(OUTPUT_DIR)
    
    # get images
    IMAGE_DIRS = get_dir_dict(TRAIN_DIR)
    
    processed_flair_dir = "./processed/train/flair/"
    safe_make_dir(processed_flair_dir)
    
    # generate input lists
    flairs = IMAGE_DIRS["flair"]
    ilist = [(f, OUTPUT_DIR, [0.7121478558063163, 0.8886607742498911, 2.4320288976061724])
                for f in flairs]
    
    with Pool(10) as p:
        list(tqdm(p.imap(map_safe_process, ilist), total=len(flairs)))