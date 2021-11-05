import numpy as np
import pandas
class HMMNER:
    def __init__(self, file_path):
        with open(file_path, 'r', encoding='utf-8') as f:
            self.text = f.read()
        f.close()
        self.word_label = []

        self.labels_index = {}  # 词性和索引之间建立字典，可以使用词性直接查询其对应的索引
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
        temp_list = []
        temp_str = ''
        for i in self.text:
            if len(temp_list) == 0:
                if i != '\t' and i != '\n':
                    temp_list.append(i)
            else:
                if i != '\t' and i != '\n':
                    temp_str += i
                if i == '\n':
                    temp_list.append(temp_str)
                    self.word_label.append(temp_list)
                    temp_list = []
                    temp_str = ''

        for each_pair in self.word_label:
            if self.labels_index.get(each_pair[1]) is None:
                self.labels_index[each_pair[1]] = self.label_number
                self.index_label[self.label_number] = each_pair[1]
                self.label_number += 1
            self.last_word_label = each_pair[1]

        self.pi = np.zeros(self.label_number)
        self.A = np.zeros((self.label_number, self.label_number))

        for each_pair in self.word_label:
            self.pi[self.labels_index[each_pair[1]]] += 1
            self.A[self.labels_index[self.last_word_label], self.labels_index[each_pair[1]]] += 1
            if self.B.get(each_pair[0]) is None:
                self.B[each_pair[0]] = np.zeros(self.label_number)
            self.B[each_pair[0]][self.labels_index[each_pair[1]]] += 1
            self.last_word_label = each_pair[1]

        for x in self.B:
            for i in range(self.label_number):
                self.B[x][i] /= self.pi[i]
        self.pi /= self.pi.sum()

        for i in range(self.label_number):
            self.A[i] /= self.A[i].sum()

    def label(self, text_list):
        """
        对文本text进行词性标注，首先调用viterbi算法，
        根据返回的return_list将词性标注的结果进行打印和存储。
        """
        self.label_text = text_list
        result_list = self.viterbi()
        for i in range(len(result_list)):
            if i != len(result_list) - 1:
                print(self.label_text[i], self.index_label[result_list[i]], end='|')
            else:
                print(self.label_text[i], self.index_label[result_list[i]])
        print("实体标注已完成！")

    def viterbi(self) -> list:
        # 根据语料中词性标注类别的多少初始化delta和path矩阵
        self.label_delta = np.zeros((self.label_number, len(self.label_text)))
        self.label_path = np.zeros((self.label_number, len(self.label_text)))
        for i in range(self.label_number):
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
    ner = HMMNER("./data/msra_train_bio")
    ner.train()
    ner.label('１２月３１日，中共中央总书记、国家主席江泽民发表１９９８年新年讲话《迈向充满希望的新世纪》。（新华社记者兰红光摄）')
