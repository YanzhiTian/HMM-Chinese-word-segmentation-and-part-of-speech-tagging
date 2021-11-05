import BMMSeg
import numpy as np
import time


class BMMSegTest:
    """
    用于测试BMM分词，构造函数中参数为词典路径和测试文档路径
    由于该类与HMMSegTest类实现方法基本一致，因此不再进行注释
    """
    def __init__(self, dict_file_path, test_file_path, result_path):
        seg = BMMSeg.BMMSeg(dict_file_path)
        print("------BMM词典载入成功------")
        start_time = time.time()
        with open(test_file_path, 'r', encoding='utf-8') as f:
            text = f.readline()
            ans = []
            test = []
            cnt = 0
            while True:
                cnt += 1
                if text == '':
                    break
                if text == '\n':
                    text = f.readline()
                    continue
                for each_part in text.split():
                    if len(each_part) == 1:
                        ans.append('S')
                    elif len(each_part) == 2:
                        ans.append('B')
                        ans.append('E')
                    elif len(each_part) >= 3:
                        ans.append('B')
                        for i in range(len(each_part) - 2):
                            ans.append('M')
                        ans.append('E')
                text = "".join(text.split())
                seg.cut(text)
                test.extend(seg.BMES_result)
                text = f.readline()
                if cnt % 1000 == 0:
                    print(cnt)
            f.close()
        label_index = {'B': 0, 'M': 1, 'E': 2, 'S': 3}
        index_label = {0: 'B', 1: 'M', 2: 'E', 3: 'S'}
        test_label_cnt = np.zeros(4)
        ans_label_cnt = np.zeros(4)
        for i in range(len(test)):
            test_label_cnt[label_index[test[i]]] += 1
            ans_label_cnt[label_index[ans[i]]] += 1
        correct = []
        for i in range(len(ans)):
            if test[i] == ans[i]:
                correct.append(test[i])
        correct_label_cnt = np.zeros(4)
        for i in range(len(correct)):
            correct_label_cnt[label_index[correct[i]]] += 1
        print("------BMM分词测试-----")
        self.save_to_file("------BMM分词测试-----\n", result_path)
        for i in range(4):
            print(index_label[i], ':')
            writestr = str(index_label[i])+':\n'
            self.save_to_file(writestr, result_path)
            R = correct_label_cnt[i] / ans_label_cnt[i]
            P = correct_label_cnt[i] / test_label_cnt[i]
            F = 2 * R * P / (R + P)
            print('R:', R, 'P:', P, 'F:', F)
            writestr = 'R:'+str(R)+'P:'+str(P)+'F:'+str(F)+'\n'
            self.save_to_file(writestr, result_path)

        print("共计", len(ans), "字")
        writestr = "共计"+str(len(ans))+"字\n"
        self.save_to_file(writestr, result_path)
        end_time = time.time()
        print("用时：", end_time - start_time, "s")
        print("------BMM分词测试完成------")
        writestr = "用时："+str(end_time - start_time)+"s\n------BMM分词测试完成------\n"
        self.save_to_file(writestr, result_path)

    def save_to_file(self, writestr, result_path):
        with open(result_path, 'a') as f:
            f.write(writestr)
        f.close()

if __name__ == "__main__":
    hmm_test = BMMSegTest("./data/dict.txt", "./data/人民日报语料（UTF8）.utf8", './result.txt')


