import BMMSegTest
import HMMSegTest
import HMMPOSTest
import time


if __name__ == "__main__":
    t = time.strftime('%Y%m%d%H%M%S')
    result_path = "./"+t+"_result.txt"
    bmm_seg = BMMSegTest.BMMSegTest("./data/dict.txt", "./data/人民日报语料（UTF8）.utf8", result_path)
    hmm_seg = HMMSegTest.HMMSegTest("./data/人民日报语料（UTF8）.utf8", "./data/人民日报语料（UTF8）.utf8", result_path, 10000)
    hmm_pos = HMMPOSTest.HMMPOSTest("./data/人民日报词性标注语料.txt", "./data/人民日报词性标注语料.txt", result_path, 10000)
