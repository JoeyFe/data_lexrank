import os

import argparse
import time
from datetime import timedelta
from LexRank.LexRank_sdk.LexRank_sdk import model_sdk

def get_args():
    parser = argparse.ArgumentParser(
        """Implementation of LexRank""")
    parser.add_argument("--test", type=str, required=False,default='../dataset/test_text')
    args = parser.parse_args()
    return args


def get_time_dif(start_time):
    """获取已使用时间"""
    end_time = time.time()
    time_dif = end_time - start_time
    return timedelta(seconds=int(round(time_dif)))


if __name__ == "__main__":
    opt = get_args()
    start_time = time.time()
    test_path = os.path.join(opt.test, 'test.txt')
    print(test_path)
    if opt.test != None:
        print("start testing")
        model_sdk(test_path)

    time_dif = get_time_dif(start_time)
    print("Time usage", time_dif)
