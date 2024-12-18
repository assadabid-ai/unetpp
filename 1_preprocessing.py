import os
from concurrent.futures import ProcessPoolExecutor
from itertools import repeat
from tqdm import tqdm
import warnings
import multiprocessing
from glob import glob
from utils import reclassify_month, dilate_image, preprocess_s1, preprocess_s2, stratify_data, calcuate_mean_std

warnings.simplefilter("ignore")

root_dir = os.getcwd() # Change to the root folder where you downloaded raw data

train_img_dir = f"{root_dir}/train_features"
test_img_dir = f"{root_dir}/test_features"

CORES = multiprocessing.cpu_count() // 2

S1_S2_TRAIN_TIFs = glob(f"{train_img_dir}/*.tif")
uIDs_train = sorted(set([os.path.basename(name).split('_')[0] for name in S1_S2_TRAIN_TIFs]))

S1_S2_TEST_TIFs = glob(f"{test_img_dir}/*.tif")
uIDs_test = sorted(set([os.path.basename(name).split('_')[0] for name in S1_S2_TEST_TIFs]))


if __name__ == '__main__':

    train_img_dir_s1 = f"{root_dir}/train_features_s1_6S"
    if not os.path.exists(train_img_dir_s1):
        os.mkdir(train_img_dir_s1)
    with ProcessPoolExecutor(CORES) as pool:
        print(f'Pre-processing {train_img_dir_s1} data')
        result = list(tqdm(pool.map(preprocess_s1, uIDs_train, repeat(train_img_dir), repeat(train_img_dir_s1), 
                                    repeat('6S')), total=len(uIDs_train)))
    try:
        result
    except Exception:
        raise Exception(f"preprocess_s1() on {train_img_dir_s1} failed")


    train_img_dir_s2 = f"{root_dir}/train_features_s2_6S"
    if not os.path.exists(train_img_dir_s2):
        os.mkdir(train_img_dir_s2)
    with ProcessPoolExecutor(CORES) as pool:
        print(f'Pre-processing {train_img_dir_s2} data')
        result = list(tqdm(pool.map(preprocess_s2, uIDs_train, repeat(train_img_dir), repeat(train_img_dir_s2), 
                                    repeat('6S')), total=len(uIDs_train)))
    try:
        result
    except Exception:
        raise Exception(f"preprocess_s2() on {train_img_dir_s2} failed")


    test_img_dir_s1 = f"{root_dir}/test_features_s1_6S"
    if not os.path.exists(test_img_dir_s1):
        os.mkdir(test_img_dir_s1)
    with ProcessPoolExecutor(CORES) as pool:
        print(f'Pre-processing {test_img_dir_s1} data')
        result = list(tqdm(pool.map(preprocess_s1, uIDs_test, repeat(test_img_dir), repeat(test_img_dir_s1), 
                                    repeat('6S')), total=len(uIDs_test)))
    try:
        result
    except Exception:
        raise Exception(f"preprocess_s1() on {test_img_dir_s1} failed")


    test_img_dir_s2 = f"{root_dir}/test_features_s2_6S"
    if not os.path.exists(test_img_dir_s2):
        os.mkdir(test_img_dir_s2)
    with ProcessPoolExecutor(CORES) as pool:
        print(f'Pre-processing {test_img_dir_s2} data')
        result = list(tqdm(pool.map(preprocess_s2, uIDs_test, repeat(test_img_dir), repeat(test_img_dir_s2), 
                                    repeat('6S')), total=len(uIDs_test)))
    try:
        result
    except Exception:
        raise Exception(f"preprocess_s2() on {test_img_dir_s2} failed")
