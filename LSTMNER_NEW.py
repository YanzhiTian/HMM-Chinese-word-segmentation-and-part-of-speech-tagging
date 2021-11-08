import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import os


class MSRADataset(Dataset):
    def __init__(self, file_path, train_corpus_num=1000000):
        self.word_index = {}
        self.file_path = file_path
        self.word_num = 1
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
        self.label_index = {'<pad>': 0, 'B-LOC': 1, 'I-LOC': 2, 'B-ORG': 3, 'I-ORG': 4,
                            'B-PER': 5, 'I-PER': 6, 'O': 7}
        self.x = []
        self.y = []
        cnt = 0
        tempx = []
        tempy = []
        flag = 0
        for each_word in word:
            # 设置一个含有tensor的list，其中一个tensor对应一个完整的句子。
            if len(each_word) == 0 and flag == 1:
                tempx = torch.tensor(tempx)
                tempy = torch.tensor(tempy)
                self.x.append(tempx)
                self.y.append(tempy)
                tempx = []
                tempy = []
                cnt += 1
                flag = 0
            if len(each_word) != 0:
                each_word = each_word.split('\t')
                tempx.append(self.word_index[each_word[0]])
                tempy.append(self.label_index[each_word[1]])
                flag = 1

            if cnt == self.corpus_num:
                break

    def __getitem__(self, index):
        return self.x[index], self.y[index]

    def __len__(self):
        return self.corpus_num


class LSTM(nn.Module):
    def __init__(self, n_vocab, n_label, embedding_dim=100, hidden_dim=512, batch_size=64):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.embedding = nn.Embedding(n_vocab, embedding_dim)  # 设置词嵌入embedding层
        self.lstm = nn.LSTM(input_size=embedding_dim, hidden_size=hidden_dim,
                            num_layers=1, bidirectional=True, batch_first=True)  # 设置双向LSTM层
        self.linear = nn.Linear(hidden_dim*2, n_label)  # 在LSTM层后增加全连接层

        # 使用正态分布初始化h0，c0
        self.h0 = torch.randn(2, batch_size, self.hidden_dim)#batchsize=64
        self.c0 = torch.randn(2, batch_size, self.hidden_dim)

    def forward(self, x):
        embedding_out = self.embedding(x)
        lstm_out, _ = self.lstm(embedding_out, (self.h0, self.c0))
        fc = self.linear(lstm_out)
        return fc


class MyCollate:
    # 对一个batch中的句子做padding
    def __init__(self, pad_value):
        self.padding = pad_value

    def __call__(self, batch_data):
        x_ = []
        y_ = []
        for x, y in batch_data:
            x_.append(x)
            y_.append(y)
        x_ = nn.utils.rnn.pad_sequence(x_, batch_first=True, padding_value=self.padding)
        y_ = nn.utils.rnn.pad_sequence(y_, batch_first=True, padding_value=self.padding)
        return x_, y_


if __name__ == "__main__":
    config = {
        'data_path': './data/msra_train_bio',
        'model_path': './model/batch_BiLSTM_model',
        'train_corpus_num': 100000,  # 训练的语句数
        'epochs': 10,
        'embedding_dim': 100,  # 词嵌入层的维度
        'hidden_dim': 512,
        'batch_size': 64
    }
    embedding_dim = config['embedding_dim']
    hidden_dim = config['hidden_dim']
    batch_size = config['batch_size']
    dataset = MSRADataset(config['data_path'], config['train_corpus_num'])
    dl = DataLoader(dataset=dataset, batch_size=batch_size, collate_fn=MyCollate(0))
    n_vocab = len(dataset.word_index)
    n_label = len(dataset.label_index)
    model = LSTM(n_vocab, n_label, embedding_dim, hidden_dim, batch_size)
    if os.path.exists(config['model_path']):  # 加载已训练好的模型
        model.load_state_dict(torch.load(config['model_path']))
    epochs = 10
    optimizer = optim.Adam(model.parameters(), lr=0.01)  # 使用Adam优化器
    loss_function = nn.CrossEntropyLoss()  # 使用交叉熵损失函数
    for epoch in range(config['epochs']):
        running_loss = 0  # 一个epoch的总loss
        for batch, (x, y) in enumerate(dl):
            optimizer.zero_grad()
            pred = model(x)
            pred = pred.view(-1, n_label)
            y = y.view(-1)
            loss = loss_function(pred, y)
            running_loss += float(loss)
            loss.backward()
            optimizer.step()
            if batch % 100 == 0:
                print("loss=", float(loss))
        print('epoch:', epoch, ' loss:', running_loss)
        torch.save(model.state_dict(), config['model_path'])
