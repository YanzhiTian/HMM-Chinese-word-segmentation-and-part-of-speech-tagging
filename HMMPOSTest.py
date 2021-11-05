import HMMPOS
import numpy as np
import time


class HMMPOSTest:
    """
    用于测试HMM词性标注，构造函数中参数为训练文档路径和测试文档路径
    该类的实现与HMMSegTest较为类似，因此仅对不同之处进行注释
    """
    def __init__(self, train_file_path, test_file_path, result_path, train_corpus_num=10000):
        train_num = train_corpus_num
        pos = HMMPOS.HMMPOS(train_file_path, train_num)
        pos.train()
        print(f"------HMM词性标注训练完成，使用语料{train_num}行------")
        start_time = time.time()
        with open(test_file_path, 'r', encoding='utf-8') as f:
            text = f.readline()
            ans = []  # 记录测试文档中正确的词性标注
            test = []  # 记录HMM模型预测的词性标注
            cnt = 0
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
                text_list = []
                for each_part in text.split():
                    word_label = each_part.split('/')
                    ans.append(word_label[1])
                    text_list.append(word_label[0])
                pos.label(text_list)
                test.extend(pos.label_result)
                text = f.readline()
                if cnt % 1000 == 0:
                    print(cnt)
            f.close()
        label_index = pos.label_index  # 调用HMMPOS类中建立的词性标注与索引之间的字典
        index_label = pos.index_label  # 调用HMMPOS类中建立的索引与词性标注之间的字典
        test_label_cnt = np.zeros(pos.label_number)
        ans_label_cnt = np.zeros(pos.label_number)
        for i in range(len(test)):  # 分别统计各词性标注的个数
            test_label_cnt[label_index[test[i]]] += 1
            ans_label_cnt[label_index[ans[i]]] += 1
        correct = []
        for i in range(len(ans)):  # 记录正确的词性标注
            if test[i] == ans[i]:
                correct.append(test[i])
        correct_label_cnt = np.zeros(pos.label_number)
        for i in range(len(correct)):  # 统计正确的词性标注个数
            correct_label_cnt[label_index[correct[i]]] += 1
        print("------HMM词性标注测试-----")
        self.save_to_file("------HMM词性标注测试-----\n", result_path)
        for i in range(pos.label_number):  # 分别对各词性标注计算其R,P和F值
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
        print("------HMM词性标注测试完成------")
        writestr = "用时："+str(end_time - start_time)+"s\n------HMM词性标注测试完成------\n"
        self.save_to_file(writestr, result_path)

    def save_to_file(self, writestr, result_path):
        with open(result_path, 'a') as f:
            f.write(writestr)
        f.close()

if __name__ == "__main__":
    hmm_test = HMMPOSTest("./data/人民日报词性标注语料.txt", "./data/人民日报词性标注语料.txt", './result.txt', 10000)


