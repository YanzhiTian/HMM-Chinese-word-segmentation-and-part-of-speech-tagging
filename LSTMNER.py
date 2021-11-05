import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import os


class MSRADataset(Dataset):
    def __init__(self, file_path, train_corpus_num=10000):
        self.word_index = {}
        self.file_path = file_path
        self.word_num = 0
        self.corpus_num = train_corpus_num
        self.train_loader = None
        with open(file_path, 'r', encoding='utf-8') as f:
            text = f.read()
            word = text.split("\n")
            cnt = 0
            for each_word in word:
                if cnt == self.corpus_num:
                    break
                if len(each_word) != 0:
                    each_word = each_word.split('\t')
                    if self.word_index.get(each_word[0]) is None:
                        self.word_index[each_word[0]] = self.word_num
                        self.word_num += 1
                if len(each_word) == 0:
                    cnt += 1
        f.close()
        self.label_index = {'O': 0, 'B-LOC': 1, 'I-LOC': 2, 'B-ORG': 3, 'I-ORG': 4,
                            'B-PER': 5, 'I-PER': 6}
        self.x = []
        self.y = []
        cnt = 0
        tempx = []
        tempy = []
        for each_word in word:
            # 设置一个含有tensor的list，其中一个tensor对应一个完整的句子。
            if len(each_word) == 0:
                tempx = torch.tensor(tempx)
                tempy = torch.tensor(tempy)
                self.x.append(tempx)
                self.y.append(tempy)
                tempx = []
                tempy = []
                cnt += 1
            if len(each_word) != 0:
                each_word = each_word.split('\t')
                tempx.append(self.word_index[each_word[0]])
                tempy.append(self.label_index[each_word[1]])

            if cnt == self.corpus_num:
                break

    def __getitem__(self, index):
        return self.x[index], self.y[index]

    def __len__(self):
        return self.corpus_num


class LSTM(nn.Module):
    def __init__(self, n_vocab, n_label, embedding_dim=100, hidden_dim=150):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.embedding = nn.Embedding(n_vocab, embedding_dim)  # 设置词嵌入embedding层
        self.lstm = nn.LSTM(input_size=embedding_dim, hidden_size=hidden_dim//2,
                            num_layers=1, bidirectional=True)  # 设置双向LSTM层
        self.linear = nn.Linear(hidden_dim, n_label)  # 在LSTM层后增加全连接层

        # 使用正态分布初始化h0，c0
        self.h0 = torch.randn(2, 1, self.hidden_dim//2)
        self.c0 = torch.randn(2, 1, self.hidden_dim//2)

    def forward(self, x):
        embedding_out = self.embedding(x).view(len(x), 1, -1)
        lstm_out, _ = self.lstm(embedding_out, (self.h0, self.c0))
        lstm_out = lstm_out.view(len(x), self.hidden_dim)
        fc = self.linear(lstm_out)
        return fc


if __name__ == "__main__":
    config = {
        'data_path': './data/msra_train_bio',
        'model_path': './model/LSTM_model',
        'train_corpus_num': 10000,  # 训练的语句数
        'epochs': 10,
        'embedding_dim': 100,  # 词嵌入层的维度
        'hidden_dim': 150
    }
    dataset = MSRADataset(config['data_path'],config['train_corpus_num'])
    n_vocab = len(dataset.word_index)
    n_label = len(dataset.label_index)
    embedding_dim = config['embedding_dim']
    hidden_dim = config['hidden_dim']
    model = LSTM(n_vocab, n_label, embedding_dim, hidden_dim)
    if os.path.exists(config['model_path']):  # 加载已训练好的模型
        model.load_state_dict(torch.load(config['model_path']))
    epochs = 10
    optimizer = optim.Adam(model.parameters(), lr=0.01)  # 使用Adam优化器
    loss_function = nn.CrossEntropyLoss()  # 使用交叉熵损失函数
    step = 0
    for epoch in range(config['epochs']):
        running_loss = 0  # 一个epoch的总loss
        for x, y in dataset:
            step += 1
            optimizer.zero_grad()
            pred = model(x)
            loss = loss_function(pred, y)
            running_loss += float(loss)
            loss.backward()
            optimizer.step()
            if step % 100 == 0:
                print("loss=", float(loss))
        print('epoch:', epoch, ' loss:', running_loss)
        torch.save(model.state_dict(), config['model_path'])
