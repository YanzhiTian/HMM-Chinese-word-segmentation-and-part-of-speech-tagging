import HMMSeg
import numpy as np
import time


class HMMSegTest:
    """
    用于测试HMM分词，构造函数中参数为训练文档路径和测试文档路径
    """
    def __init__(self, train_file_path, test_file_path, result_path, train_corpus_num=10000):
        # 实例化并训练HMM词性标注
        train_num = train_corpus_num
        seg = HMMSeg.HMMSeg(train_file_path, train_num)
        seg.train()
        print(f"------HMM分词训练完成，使用语料{train_num}行------")
        start_time = time.time()
        with open(test_file_path, 'r', encoding='utf-8') as f:
            text = f.readline()
            ans = []  # 用于记录测试文档中正确的分词结果
            test = []  # 用于记录HMM模型预测的分词结果
            cnt = 0  # 显示测试进度
            while True:
                cnt += 1
                if cnt <= train_num:
                    # 根据划分的语料库，从train_corpus_num行后座位测试语料库
                    f.readline()
                    continue
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
                seg.cut(text)  # 调用分词（cut）方法
                test.extend(seg.BMES_result)
                text = f.readline()
                if cnt % 1000 == 0:
                    print(cnt)
            f.close()
        label_index = {'B': 0, 'M': 1, 'E': 2, 'S': 3}
        index_label = {0: 'B', 1: 'M', 2: 'E', 3: 'S'}
        test_label_cnt = np.zeros(4)
        ans_label_cnt = np.zeros(4)
        for i in range(len(test)):  # 分别统计BMES分词标注的个数
            test_label_cnt[label_index[test[i]]] += 1
            ans_label_cnt[label_index[ans[i]]] += 1
        correct = []
        for i in range(len(ans)):  # 记录正确分词的BMES标注
            if test[i] == ans[i]:
                correct.append(test[i])
        correct_label_cnt = np.zeros(4)
        for i in range(len(correct)):  # 统计正确分词的BMES标注个数
            correct_label_cnt[label_index[correct[i]]] += 1
        print("------HMM分词测试-----")
        self.save_to_file("------HMM分词测试-----\n", result_path)
        for i in range(4):  # 分别对BMES四种标注计算其R,P和F值
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
        print("------HMM分词测试完成------")
        writestr = "用时："+str(end_time - start_time)+"s\n------HMM分词测试完成------\n"
        self.save_to_file(writestr, result_path)

    def save_to_file(self, writestr, result_path):
        with open(result_path, 'a') as f:
            f.write(writestr)
        f.close()

if __name__ == "__main__":
    hmm_test = HMMSegTest("./data/人民日报语料（UTF8）.utf8", "./data/人民日报语料（UTF8）.utf8", './result.txt', 10000)


