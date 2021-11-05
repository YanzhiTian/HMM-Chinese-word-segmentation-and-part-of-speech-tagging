import numpy as np


class HMMSeg:
    # 在该程序中，四种词标BMES的索引值分别为0123
    def __init__(self, file_path, train_corpus_num=10000):
        with open(file_path, 'r', encoding="utf-8") as f:
            # 使用前train_corpus_num行训练
            self.text = ''
            train_num = train_corpus_num
            for i in range(train_num):
                self.text += f.readline()
        f.close()
        self.pi = np.zeros(4)
        self.A = np.zeros((4, 4))
        self.B = {}
        self.words = self.text.split()
        if len(self.text[-1]) >= 3:
            self.last_word_type = 2
        else:
            self.last_word_type = 3
        self.cut_text = ''  # 存储需要分词的文本
        self.cut_result = []  # 以列表形式储存分词结果
        self.BMES_result = []  # 以BMES形式返回分词结果
        self.cut_delta = None  # viterbi算法中所需的delta矩阵
        self.cut_path = None  # viterbi算法中寻找最优序列的矩阵

    def train(self):
        """
        根据语料库的分词结果计算出HMM所需的三个矩阵，详细的数据结构及说明如下：
        pi:初始状态概率矩阵，计算了BMES四种词标（状态）出现的概率。
        使用的数据结构为一维numpy向量。
        | B | P(B) |
        | M | P(M) |
        | E | P(E) |
        | S | P(S) |

        A:状态转移概率矩阵，计算了BMES四种词标之间转移的概率。
        使用的数据结构为二维numpy数组。
            |   B   |   M   |   E   |   S   |
        | B |P(B->B)|P(B->M)|P(B->E)|P(B->S)|
        | M |P(M->B)|P(M->M)|P(M->E)|P(M->S)|
        | E |P(E->B)|P(E->M)|P(E->E)|P(E->S)|
        | S |P(S->B)|P(S->M)|P(S->E)|P(S->S)|

        B:观测概率矩阵，计算了MNES四种词标（状态）下观察值为某汉字的概率。
        使用的数据结构为字典，其中key为某汉字，value为该汉字的[B, M, E, S]概率分布。
            |   汉字1  |   汉字2  |   汉字3  |...
        | B |P(汉字1|B)|P(汉字2|B)|P(汉字3|B)|...
        | M |P(汉字1|M)|P(汉字2|M)|P(汉字3|M)|...
        | E |P(汉字1|E)|P(汉字2|E)|P(汉字3|E)|...
        | S |P(汉字1|S)|P(汉字2|S)|P(汉字3|S)|...
        """
        for each_word in self.words:
            if len(each_word) == 0:
                continue
            elif len(each_word) == 1:  # 词中有一个汉字，词标为S
                self.A[self.last_word_type, 3] += 1
                self.pi[3] += 1
                self.last_word_type = 3
                if self.B.get(each_word) is None:
                    self.B[each_word] = [0, 0, 0, 0]
                else:
                    self.B[each_word][3] += 1
            elif len(each_word) == 2:  # 词中有两个汉字，词标为BE
                self.A[self.last_word_type, 0] += 1
                self.A[0, 2] += 1  # B->E
                self.pi[0] += 1
                self.pi[2] += 1
                self.last_word_type = 2
                if self.B.get(each_word[0]) is None:
                    self.B[each_word[0]] = [0, 0, 0, 0]
                self.B[each_word[0]][0] += 1
                if self.B.get((each_word[1])) is None:
                    self.B[each_word[1]] = [0, 0, 0, 0]
                self.B[each_word[1]][2] += 1
            else:  # 词中有三个及以上汉字，词标为BM...ME
                self.A[self.last_word_type, 0] += 1
                self.A[0, 1] += 1  # B->M
                self.A[1, 1] += (len(each_word) - 3)  # M->M
                self.A[1, 2] += 1  # M->E
                self.pi[0] += 1
                self.pi[1] += (len(each_word) - 2)
                self.pi[2] += 1
                self.last_word_type = 2
                for i in range(len(each_word)):
                    if(self.B.get(each_word[i])) is None:
                        self.B[each_word[i]] = [0, 0, 0, 0]
                    if i == 0:
                        self.B[each_word[i]][0] += 1
                    elif i == (len(each_word) - 1):
                        self.B[each_word[i]][2] += 1
                    else:
                        self.B[each_word[i]][1] += 1
        #  B矩阵归一化，每个位置所有词之和=1
        for x in self.B:
            for i in range(4):
                self.B[x][i] /= self.pi[i]
        #  pi矩阵归一化
        self.pi /= self.pi.sum()
        #  A矩阵归一化，每个位置转移到所有位置之和=1
        for i in range(4):
            self.A[i] /= self.A[i].sum()

    def cut(self, text) -> list:
        """
        对文本text进行分词，首先调用viterbi算法，
        根据返回的return_list将分词结果进行打印和存储。
        """
        self.cut_text = text
        self.cut_result = []
        self.BMES_result = []
        if len(text) == 0:
            print("文本不能为空！")
        else:
            result_list = self.viterbi()
            # 以下注释内容为打印分词结果
            # for i in range(len(result_list)):
            #     if result_list[i] == 2 or result_list[i] == 3:
            #         if i != len(result_list) - 1:
            #             print(self.cut_text[i], end=' |')
            #         else:
            #             print(self.cut_text[i])
            #     else:
            #         print(self.cut_text[i], end='')
            # print("分词已完成！")
            temp_str = ''
            for i in range(len(result_list)):
                if result_list[i] == 0:
                    self.BMES_result.append('B')
                elif result_list[i] == 1:
                    self.BMES_result.append('M')
                elif result_list[i] == 2:
                    self.BMES_result.append('E')
                elif result_list[i] == 3:
                    self.BMES_result.append('S')

                if result_list[i] == 2 or result_list[i] == 3:
                    temp_str += self.cut_text[i]
                    self.cut_result.append(temp_str)
                    temp_str = ''
                else:
                    temp_str += self.cut_text[i]
        return self.cut_result

    def viterbi(self) -> list:
        """
        使用viterbi算法求解概率最大的分词序列。
        算法的状态转移方程为delta[i,j]=max{delta[k,j-1]*A[k,i]*B[第j个汉字][i]}
        其中i为前一隐状态（词标的索引值），k为当前隐状态（词标的索引值），j为文本中当前汉字的序数。
        cut_delta矩阵用于viterbi算法中动态规划的状态转移和记录。
        cut_path矩阵用于回溯viterbi算法计算的最优序列。
        """
        self.cut_delta = np.zeros((4, len(self.cut_text)))
        self.cut_path = np.zeros((4, len(self.cut_text)))
        # 初始化delta矩阵
        for i in range(4):
            if self.cut_text[0] not in self.B:
                self.add_word(self.cut_text[0])
            self.cut_delta[i, 0] = self.pi[i] * self.B[self.cut_text[0]][i]
        # 对分词文本中每个汉字计算其为BMES四种词标的概率
        for j in range(1, len(self.cut_text)):
            for i in range(4):
                max_value = 0
                max_index = 0
                # 遍历前一隐状态，寻找当前汉字最大概率的隐状态
                for k in range(4):
                    if self.cut_text[j] not in self.B:
                        self.add_word(self.cut_text[j])
                    if self.cut_delta[k, j - 1] * self.A[k, i] * self.B[self.cut_text[j]][i] > max_value:
                        max_value = self.cut_delta[k, j - 1] * self.A[k, i] * self.B[self.cut_text[j]][i]
                        max_index = k
                self.cut_delta[i, j] = max_value
                self.cut_path[i, j] = max_index
        result_list = []
        max_value = 0
        max_index = 0
        # 计算文本中最后一个汉字哪种词标的概率最大
        for i in range(4):
            if self.cut_delta[i, len(self.cut_text) - 1] > max_value:
                max_value = self.cut_delta[i, len(self.cut_text) - 1]
                max_index = i
        result_list.append(max_index)

        if len(self.cut_text) != 1:
            # 根据刚计算出最后一个汉字概率最大词标对应的索引，向前回溯最优序列。
            max_pre_index = int(self.cut_path[max_index][len(self.cut_text) - 1])
            for i in range(len(self.cut_text) - 2, 0, -1):
                result_list.append(max_pre_index)
                max_pre_index = int(self.cut_path[max_pre_index][i])
            result_list.append(max_pre_index)
            result_list.reverse()
        return result_list

    def add_word(self, text):
        # 如果B词典中没有这个分词，把它加上。之后该怎么处理？概率怎么填？
        self.B[text] = np.ones(4)


if __name__ == "__main__":
    seg = HMMSeg("./data/人民日报语料（UTF8）.utf8", 10000)
    seg.train()
    seg.cut('迈向充满希望的新世纪——一九九八年新年讲话（附图片１张）')
    print(seg.cut_result)
    print(seg.BMES_result)
