class Trie:
    def __init__(self):
        self.root = {}
        self.end_token = '[END]'  # 表示一个词语的结束

    def insert(self, word):
        node = self.root
        for char in word:
            node = node.setdefault(char, {})
        node[self.end_token] = 'END'  # 一个词插入完成

    def search(self, word):
        node = self.root
        for char in word:
            if char not in node:
                return False
            node = node[char]
        if self.end_token in node:
            return True
        else:
            return False


class BMMSeg(Trie):
    def __init__(self, file_path):
        self.trie = Trie()
        self.text = ''
        self.BMES_result = []
        self.cut_result = []
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.split()
                self.trie.insert(line[0])
        f.close()

    def cut(self, sentence):
        self.cut_result = []
        self.BMES_result = []
        if len(sentence) == 0:
            print("文本不能为空！")
        else:
            self.text = sentence
            end = len(sentence)
            while end > 0:
                word = ''
                for start in range(0, end):  # 从后向前检查文本
                    word = sentence[start:end]
                    if self.trie.search(word):
                        end = start + 1
                        break
                end -= 1
                self.cut_result.append(word)

            self.cut_result.reverse()  # 由于是从后向前匹配，所以需要对列表进行翻转
            for each_part in self.cut_result:
                if len(each_part) == 1:
                    self.BMES_result.append('S')
                elif len(each_part) == 2:
                    self.BMES_result.append('B')
                    self.BMES_result.append('E')
                elif len(each_part) >= 3:
                    self.BMES_result.append('B')
                    for i in range(len(each_part) - 2):
                        self.BMES_result.append('M')
                    self.BMES_result.append('E')
            # 以下注释内容为打印分词结果
            # for i in range(len(result)):
            #     if i != len(result) - 1:
            #         print(result[i], end=' |')
            #     else:
            #         print(result[i])
            # print("分词已完成！")
        return self.cut_result


if __name__ == "__main__":
    seg = BMMSeg('./data/dict.txt')
    seg.cut('迈向充满希望的新世纪——一九九八年新年讲话（附图片１张）')
    print(seg.cut_result)
    print(seg.BMES_result)
