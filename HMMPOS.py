import numpy as np
import BMMSeg
import HMMSeg


class HMMPOS:
    """
    由于使用隐马尔科夫模型进行词性标注和分词原理及算法基本一致，只有部分细节需要修改，
    因此类HMMPOS在类HMMSeg上进行了修改，相同部分不再进行注释，只有不同的部分进行注释
    """
    def __init__(self, file_path, train_corpus_num=10000):
        with open(file_path, 'r', encoding="utf-8") as f:
            # 使用前train_corpus_num行训练
            self.text = ''
            train_num = train_corpus_num
            for i in range(train_num):
                self.text += f.readline()
        f.close()
        self.part = self.text.split()
        self.label_index = {}  # 词性和索引之间建立字典，可以使用词性直接查询其对应的索引
        self.index_label = {}  # 索引和词性之间建立字典，可以使用词性直接查询其对应的索引
        self.label_number = 0  # 计算该语料库中一共有多少种词性，提高模型适用范围
        self.pi = None
        self.A = None
        self.B = {}
        self.last_word_label = None
        self.label_text = []
        self.label_result = []
        self.label_delta = None
        self.label_path = None

    def train(self):
        for each_parts in self.part:
            word_label = each_parts.split("/")
            if self.label_index.get(word_label[1]) is None:
                self.label_index[word_label[1]] = self.label_number
                self.index_label[self.label_number] = word_label[1]
                self.label_number += 1
            self.last_word_label = word_label[1]
        self.pi = np.zeros(self.label_number)
        self.A = np.zeros((self.label_number, self.label_number))

        for each_parts in self.part:
            word_label = each_parts.split("/")
            self.pi[self.label_index[word_label[1]]] += 1
            self.A[self.label_index[self.last_word_label], self.label_index[word_label[1]]] += 1
            if self.B.get(word_label[0]) is None:
                self.B[word_label[0]] = np.zeros(self.label_number)
            self.B[word_label[0]][self.label_index[word_label[1]]] += 1
            self.last_word_label = word_label[1]

        for x in self.B:
            for i in range(self.label_number):
                self.B[x][i] /= self.pi[i]
        self.pi /= self.pi.sum()

        for i in range(self.label_number):
            self.A[i] /= self.A[i].sum()

    def label(self, text_list) -> list:
        """
        对文本text进行词性标注，首先调用viterbi算法，
        根据返回的return_list将词性标注的结果进行打印和存储。
        """
        self.label_text = text_list
        self.label_result = []
        result_list = []
        if len(text_list) == 0:
            print("文本不能为空！")
        else:
            result_list = self.viterbi()
            for i in range(len(result_list)):
                self.label_result.append(self.index_label[result_list[i]])
        # 以下注释内容为打印词性标注结果
        # for i in range(len(result_list)):
        #     if i != len(result_list) - 1:
        #         print(self.label_text[i], self.index_label[result_list[i]], end='|')
        #     else:
        #         print(self.label_text[i], self.index_label[result_list[i]])
        # print("词性标注已完成！")
        return self.label_result

    def viterbi(self) -> list:
        # 根据语料中词性标注类别的多少初始化delta和path矩阵
        self.label_delta = np.zeros((self.label_number, len(self.label_text)))
        self.label_path = np.zeros((self.label_number, len(self.label_text)))
        for i in range(self.label_number):
            if self.label_text[0] not in self.B:
                self.add_word(self.label_text[0])
            self.label_delta[i, 0] = self.pi[i] * self.B[self.label_text[0]][i]
        for j in range(1, len(self.label_text)):
            for i in range(self.label_number):
                max_value = 0
                max_index = 0
                for k in range(self.label_number):
                    if self.label_text[j] not in self.B:
                        self.add_word(self.label_text[j])
                    if self.label_delta[k, j - 1] * self.A[k, i] * self.B[self.label_text[j]][i] > max_value:
                        max_value = self.label_delta[k, j - 1] * self.A[k, i] * self.B[self.label_text[j]][i]
                        max_index = k
                self.label_delta[i, j] = max_value
                self.label_path[i, j] = max_index
        result_list = []
        max_value = 0
        max_index = 0
        for i in range(self.label_number):
            if self.label_delta[i, len(self.label_text) - 1] > max_value:
                max_value = self.label_delta[i, len(self.label_text) - 1]
                max_index = i
        result_list.append(max_index)

        if len(self.label_text) != 1:
            max_pre_index = int(self.label_path[max_index][len(self.label_text) - 1])
            for i in range(len(self.label_text) - 2, 0, -1):
                result_list.append(max_pre_index)
                max_pre_index = int(self.label_path[max_pre_index][i])
            result_list.append(max_pre_index)
            result_list.reverse()
        return result_list

    def add_word(self, text):
        # 如果B词典中没有这个分词，把它加上。之后该怎么处理？概率怎么填？
        self.B[text] = np.ones(self.label_number)


if __name__ == "__main__":

    seg = HMMSeg.HMMSeg("./data/人民日报语料（UTF8）.utf8", 10000)
    seg.train()
    seg_result = seg.cut('迈向充满希望的新世纪——一九九八年新年讲话（附图片１张）')

    pos = HMMPOS("./data/人民日报词性标注语料.txt", 10000)
    pos.train()
    pos.label(seg_result)
    print(seg_result)
    print(pos.label_result)
