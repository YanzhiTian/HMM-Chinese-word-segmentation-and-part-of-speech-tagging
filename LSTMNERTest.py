import LSTMNER
import torch
import os


if __name__ == "__main__":
    load_index = LSTMNER.MSRADataset("./data/msra_train_bio", 10000)
    model = LSTMNER.LSTM(len(load_index.word_index), len(load_index.label_index))
    if os.path.exists('./model/LSTM_model'):
        model.load_state_dict(torch.load('./model/LSTM_model'))
    test_sentence = '中国和美国。'
    index_list = []
    for each_word in test_sentence:
        index_list.append(load_index.word_index[each_word])
    inp = torch.tensor(index_list)
    predict = model(inp)
    index_label = {0: 'O', 1: 'B-LOC', 2: 'I-LOC', 3: 'B-ORG', 4: 'I-ORG',
                   5: 'B-PER', 6: 'I-PER'}
    for each_index in predict:
        print(index_label[int(torch.argmax(each_index))])
