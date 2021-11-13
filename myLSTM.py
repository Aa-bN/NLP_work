import os
import math
import torch
import torch.nn as nn
import torch.optim as optim
import give_valid_test
import _pickle as cpickle

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


#device = torch.device("cpu")

def make_batch(train_path, word2number_dict, batch_size, n_step):
    all_input_batch = []
    all_target_batch = []

    text = open(train_path, 'r', encoding='utf-8')  # open the file

    input_batch = []
    target_batch = []
    for sen in text:
        word = sen.strip().split(" ")  # space tokenizer
        word = ["<sos>"] + word
        word = word + ["<eos>"]

        if len(word) <= n_step:  # pad the sentence
            word = ["<pad>"] * (n_step + 1 - len(word)) + word

        for word_index in range(len(word) - n_step):
            input = [word2number_dict[n] for n in word[word_index:word_index + n_step]]  # create (1~n-1) as input
            target = word2number_dict[
                word[word_index + n_step]]  # create (n) as target, We usually call this 'casual language model'
            input_batch.append(input)
            target_batch.append(target)

            if len(input_batch) == batch_size:
                all_input_batch.append(input_batch)
                all_target_batch.append(target_batch)
                input_batch = []
                target_batch = []

    return all_input_batch, all_target_batch  # (batch num, batch size, n_step) (batch num, batch size)


def make_dict(train_path):
    text = open(train_path, 'r', encoding='utf-8')  # open the train file
    word_list = set()  # a set for making dict

    for line in text:
        line = line.strip().split(" ")
        word_list = word_list.union(set(line))

    word_list = list(sorted(word_list))  # set to list

    word2number_dict = {w: i + 2 for i, w in enumerate(word_list)}
    number2word_dict = {i + 2: w for i, w in enumerate(word_list)}

    # add the <pad> and <unk_word>
    word2number_dict["<pad>"] = 0
    number2word_dict[0] = "<pad>"
    word2number_dict["<unk_word>"] = 1
    number2word_dict[1] = "<unk_word>"
    word2number_dict["<sos>"] = 2
    number2word_dict[2] = "<sos>"
    word2number_dict["<eos>"] = 3
    number2word_dict[3] = "<eos>"

    return word2number_dict, number2word_dict


class TextLSTM(nn.Module):
    def __init__(self):
        super(TextLSTM, self).__init__()
        self.C = nn.Embedding(n_class, embedding_dim=emb_size)
        # self.LSTM = nn.LSTM(input_size=emb_size, hidden_size=n_hidden)
        self.W = nn.Linear(n_hidden, n_class, bias=False)
        self.b = nn.Parameter(torch.ones([n_class]))

        # 定义LSTM中的需要的参数, 参考pytorch官方文档中的实现方式
        # 同一个表达式中的两个bias合并为一个
        # 遗忘门 ft
        self.Wif = nn.Parameter(torch.Tensor(emb_size, n_hidden))
        self.Whf = nn.Parameter(torch.Tensor(n_hidden, n_hidden))
        self.bf = nn.Parameter(torch.Tensor(batch_size, n_hidden))
        # 输入门 it
        self.Wii = nn.Parameter(torch.Tensor(emb_size, n_hidden))
        self.Whi = nn.Parameter(torch.Tensor(n_hidden, n_hidden))
        self.bi = nn.Parameter(torch.Tensor(batch_size, n_hidden))
        # 输入门 gt
        self.Wig = nn.Parameter(torch.Tensor(emb_size, n_hidden))
        self.Whg = nn.Parameter(torch.Tensor(n_hidden, n_hidden))
        self.bg = nn.Parameter(torch.Tensor(batch_size, n_hidden))
        # 输出门 Ot
        self.Wio = nn.Parameter(torch.Tensor(emb_size, n_hidden))
        self.Who = nn.Parameter(torch.Tensor(n_hidden, n_hidden))
        self.bo = nn.Parameter(torch.Tensor(batch_size, n_hidden))
        # 激活函数
        self.sig = nn.Sigmoid()
        self.tah = nn.Tanh()

        # 第二层LSTM及以后层，需要使用的矩阵变量
        self.Wif2 = nn.Parameter(torch.Tensor(batch_size, n_hidden))    # 遗忘门
        self.Wii2 = nn.Parameter(torch.Tensor(batch_size, n_hidden))    # 输入门
        self.Wig2 = nn.Parameter(torch.Tensor(batch_size, n_hidden))    # 输入门
        self.Wio2 = nn.Parameter(torch.Tensor(batch_size, n_hidden))    # 输出门

    def forward(self, X):
        X = self.C(X)       # [128, 5, 256]
        # 确定参数
        num_layers = 2
        hidden_state = torch.rand(num_layers, batch_size, n_hidden).to(device)
        cell_state = torch.rand(num_layers, batch_size, n_hidden).to(device)

        outputs, ht, ct = self.myLSTM(X=X, hidden_size=n_hidden, num_layers=num_layers, hidden_state=hidden_state,
                                      cell_state=cell_state, batch_first=True)
        # outputs, ht, ct = self.myLSTM(X=X, hidden_size=n_hidden, num_layers=num_layers, batch_first=True)
        outputs = outputs[-1]
        model = self.W(outputs) + self.b  # model : [batch_size, n_class]
        return model

    def myLSTM(self, X, hidden_size, num_layers=1, hidden_state=None, cell_state=None, batch_first=False):
        # input_size = emb_size
        # hidden_size = n_hidden
        # input_size没有作为函数参数，因为可由数据X得到：input_size = X.shape[2]

        # 初始化 hidden_state 和 cell_state
        # 首先，通过 batch_first 参数，默认False。
        # False 确定 X 的 shape 为 (L, N, Hin)  即 (sequence_length, batch_size, input_size)
        # True 确定 X 的 shape 为 (N, L, Hin)
        # 获取X的 batch_size, 最终将X转化为True的形式 (batch_size = 128, sequence_length = 5, input_size = 256)
        # 每次传入128个句子，每个句子的长度/单词数量为5，每个单词向量的长度为256
        if batch_first is True:     # X [128, 5, 256]
            lenX = len(X)
        if batch_first is False:    # X [5, 128, 256]
            X = X.transpose(0, 1)   # X [128, 5, 256]
            lenX = len(X)           # batch_size 128

        # 确定h和c
        if hidden_state is None:
            hidden_state = torch.zeros(num_layers, lenX, hidden_size).to(device)    # 只考虑单向LSTM，故num_directions = 1
        if cell_state is None:
            cell_state = torch.zeros(num_layers, lenX, hidden_size).to(device)      # [1, 128, 128]

        # 后续操作的总体思路，LSTM cell -> RNN -> LSTM, 循环 LSTM 实现多层的LSTM

        # 生成xt
        # 首先获取数据的 sequence_length, 由于已经把数据转换成batch_first，即(N, L, Hin)的形式
        # 所以第二个参数 L 即为sequence_length，也就是我们主程序中的n_step
        sequence_length = X.shape[1]    # 5

        # 存储每层LSTM最后一个cell生成的 hidden_state 和 cell_state, 即每层的hn，ctn
        hidden_state_layer_final = []
        cell_state_layer_final = []

        # 存储每层LSTM生成的 hidden_state，即[h1， h2, ... , hn]， 用来向下一层传递
        hidden_state_one_layer = []

        # LSTM

        # 第一层 LSTM  (index=0)
        # 取出第一个值，作为第一层LSTM中，h0和c0的初始值
        h0 = hidden_state[0]    # [128, 128]    [batch_size, hidden_size]
        c0 = cell_state[0]      # [128, 128]
        for i in range(sequence_length):
            # batch_size个句子，每个句子的第i个单词的词向量
            xt = X[:, i, :]             # [128, 256]    [batch_size, input_size]

            # LSTM cell
            # 一下变量的格式 [batch_size, hidden_size] [128, 128]
            # 遗忘门
            ft = self.sig(xt @ self.Wif + h0 @ self.Whf + self.bf)
            # ft = torch.sigmoid(xt @ self.Wif + h0 @ self.Whf + self.bf)
            # ft = nn.functional.sigmoid(xt @ self.Wif + h0 @ self.Whf + self.bf)
            # 输入门
            it = self.sig(xt @ self.Wii + h0 @ self.Whi + self.bi)
            gt = self.tah(xt @ self.Wig + h0 @ self.Whg + self.bg)
            # 输出门
            ot = self.sig(xt @ self.Wio + h0 @ self.Who + self.bo)
            # 记忆更新  这里的乘法是 Hadamard Product
            ct = ft * c0 + it * gt
            # 输出门   这里的乘法是 Hadamard Product
            ht = ot * self.tah(ct)

            # 保存本层LSTM产生的hidden_state，用以传入下一层
            hidden_state_one_layer.append(ht)       # [128, 128]， 列表中最终会有sequence_length=5个这样的tensor

        # 到这里，第一层LSTM结束了，保存该层最终的hn和cn
        hidden_state_layer_final.append(ht.unsqueeze(0))    # [1, 128, 128]     [num_layers, batch_size, hidden_size]
        cell_state_layer_final.append(ct.unsqueeze(0))      # [1, 128, 128]

        # 下面对num_layers进行判断，决定是否执行下面num_layers-1层的LSTM
        # 第 num_layers 层 LSTM   (index = 1, 2, num_layers-1)
        if num_layers >= 2:
            for j in range(1, num_layers):
                # 取出第j个值，作为第j层的LSTM中，h0和c0的初始值
                h0 = hidden_state[j]
                c0 = cell_state[j]
                # 单层LSTM
                for k in range(sequence_length):
                    # 从上层hidden_state中传入参数      [128, 128]
                    xt = hidden_state_one_layer[k]
                    # 注意第一个矩阵变为Wif2，Wii2等       [128, 128]
                    ft = self.sig(xt @ self.Wif2 + h0 @ self.Whf + self.bf)
                    it = self.sig(xt @ self.Wii2 + h0 @ self.Whi + self.bi)
                    gt = self.tah(xt @ self.Wig2 + h0 @ self.Whg + self.bg)
                    ot = self.sig(xt @ self.Wio2 + h0 @ self.Who + self.bo)
                    ct = ft * c0 + it * gt
                    ht = ot * self.tah(ct)
                    # 将原来的hidden_state替换为新的hidden_state以供下层使用
                    hidden_state_one_layer[k] = ht      # [128, 128]

                # 到这里，第j层LSTM结束了，保存该层最终的hn和cn
                hidden_state_layer_final.append(ht.unsqueeze(0))   # [1, 128, 128], 算第一层，最终列表中有num_layer个这样的元素
                cell_state_layer_final.append(ct.unsqueeze(0))

        # 到这里，所有的层的LSTM均执行完毕, 此时hidden_state_one_layer为最后一层的输出，len=5
        outputs_temp = []
        for i in range(len(hidden_state_one_layer)):
            last_layer_output_each_cell = hidden_state_one_layer[i].unsqueeze(0)    # [128, 128] -> [1, 128, 128]
            outputs_temp.append(last_layer_output_each_cell)                        # 最终中有5个tensor，形状为[1,128,128]

        outputs_return = torch.cat(outputs_temp, dim=0)                     # [sequence_length, batch_size, hidden_size]
        hidden_state_return = torch.cat(hidden_state_layer_final, dim=0)    # [num_layers, batch_size, hidden_size]
        cell_state_return = torch.cat(cell_state_layer_final, dim=0)        # [num_layers, batch_size, hidden_size]

        return outputs_return, hidden_state_return, cell_state_return


def train_LSTMlm():
    model = TextLSTM()
    model.to(device)
    print(model)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learn_rate)

    # Training
    batch_number = len(all_input_batch)
    for epoch in range(all_epoch):
        count_batch = 0
        for input_batch, target_batch in zip(all_input_batch, all_target_batch):
            optimizer.zero_grad()

            # input_batch : [batch_size, n_step, n_class]
            output = model(input_batch)

            # output : [batch_size, n_class], target_batch : [batch_size] (LongTensor, not one-hot)
            loss = criterion(output, target_batch)
            ppl = math.exp(loss.item())
            if (count_batch + 1) % 100 == 0:
                print('Epoch:', '%04d' % (epoch + 1), 'Batch:', '%02d' % (count_batch + 1), f'/{batch_number}',
                      'loss =', '{:.6f}'.format(loss), 'ppl =', '{:.6}'.format(ppl))

            loss.backward()
            optimizer.step()

            count_batch += 1
        print('Epoch:', '%04d' % (epoch + 1), 'Batch:', '%02d' % (count_batch + 1), f'/{batch_number}',
              'loss =', '{:.6f}'.format(loss), 'ppl =', '{:.6}'.format(ppl))

        # valid after training one epoch
        all_valid_batch, all_valid_target = give_valid_test.give_valid(data_root, word2number_dict, n_step)
        all_valid_batch = torch.LongTensor(all_valid_batch).to(device)  # list to tensor
        all_valid_target = torch.LongTensor(all_valid_target).to(device)

        total_valid = len(all_valid_target) * 128  # valid and test batch size is 128
        with torch.no_grad():
            total_loss = 0
            count_loss = 0
            for valid_batch, valid_target in zip(all_valid_batch, all_valid_target):
                valid_output = model(valid_batch)
                valid_loss = criterion(valid_output, valid_target)
                total_loss += valid_loss.item()
                count_loss += 1

            print(f'Valid {total_valid} samples after epoch:', '%04d' % (epoch + 1), 'loss =',
                  '{:.6f}'.format(total_loss / count_loss),
                  'ppl =', '{:.6}'.format(math.exp(total_loss / count_loss)))

        if (epoch + 1) % save_checkpoint_epoch == 0:
            torch.save(model, f'models/20182338ws_model_layer2_nondefault_state_epoch{epoch + 1}.ckpt')


def test_LSTMlm(select_model_path):
    model = torch.load(select_model_path, map_location="cpu")  # load the selected model
    model.to(device)

    # load the test data
    all_test_batch, all_test_target = give_valid_test.give_test(data_root, word2number_dict, n_step)
    all_test_batch = torch.LongTensor(all_test_batch).to(device)  # list to tensor
    all_test_target = torch.LongTensor(all_test_target).to(device)
    total_test = len(all_test_target) * 128  # valid and test batch size is 128
    model.eval()
    criterion = nn.CrossEntropyLoss()
    total_loss = 0
    count_loss = 0
    for test_batch, test_target in zip(all_test_batch, all_test_target):
        test_output = model(test_batch)
        test_loss = criterion(test_output, test_target)
        total_loss += test_loss.item()
        count_loss += 1

    print(f"Test {total_test} samples with {select_model_path}……………………")
    print('loss =', '{:.6f}'.format(total_loss / count_loss),
          'ppl =', '{:.6}'.format(math.exp(total_loss / count_loss)))


if __name__ == '__main__':
    n_step = 5  # number of cells(= number of Step)
    n_hidden = 128  # number of hidden units in one cell
    batch_size = 128  # batch size
    learn_rate = 0.0005
    all_epoch = 5  # the all epoch for training
    emb_size = 256  # embeding size
    save_checkpoint_epoch = 5  # save a checkpoint per save_checkpoint_epoch epochs !!! Note the save path !!!
    data_root = 'penn_small'
    train_path = os.path.join(data_root, 'train.txt')  # the path of train dataset

    print("print parameter ......")
    print("n_step:", n_step)
    print("n_hidden:", n_hidden)
    print("batch_size:", batch_size)
    print("learn_rate:", learn_rate)
    print("all_epoch:", all_epoch)
    print("emb_size:", emb_size)
    print("save_checkpoint_epoch:", save_checkpoint_epoch)
    print("train_data:", data_root)

    word2number_dict, number2word_dict = make_dict(train_path)
    # print(word2number_dict)

    print("The size of the dictionary is:", len(word2number_dict))
    n_class = len(word2number_dict)  # n_class (= dict size)

    print("generating train_batch ......")
    all_input_batch, all_target_batch = make_batch(train_path, word2number_dict, batch_size, n_step)  # make the batch
    train_batch_list = [all_input_batch, all_target_batch]

    print("The number of the train batch is:", len(all_input_batch))
    all_input_batch = torch.LongTensor(all_input_batch).to(device)  # list to tensor
    all_target_batch = torch.LongTensor(all_target_batch).to(device)
    # print(all_input_batch.shape)
    # print(all_target_batch.shape)
    all_input_batch = all_input_batch.reshape(-1, batch_size, n_step)
    all_target_batch = all_target_batch.reshape(-1, batch_size)

    print("\nTrain the LSTMLM……………………")
    train_LSTMlm()

    print("\nTest the LSTMLM……………………")
    select_model_path = "models/20182338ws_model_layer2_nondefault_state_epoch5.ckpt"
    test_LSTMlm(select_model_path)

