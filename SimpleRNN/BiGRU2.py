import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
from matplotlib import pyplot as plt
# 检查是否有可用的GPU设备
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 定义BiGRU模型
class BiGRUModel(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, output_dim):
        super(BiGRUModel, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.bi_gru = nn.GRU(embedding_dim, hidden_dim, batch_first=True, bidirectional=True)
        self.fc = nn.Linear(hidden_dim * 2, output_dim)

    def forward(self, x):
        embedded = self.embedding(x)
        output, _ = self.bi_gru(embedded)
        output = torch.cat((output[:, -1, :hidden_dim], output[:, 0, hidden_dim:]), dim=1)
        return self.fc(output)

# 数据预处理和加载
def preprocess_data(filename):
    data = pd.read_csv(filename)
    sentences = data['sentences'].tolist()
    labels = data['labels'].tolist()

    # 构建词汇表
    vocab = set(" ".join(sentences).split())
    word2idx = {word: idx + 2 for idx, word in enumerate(vocab)}
    word2idx['<unknown>'] = 0
    word2idx['<pad>'] = 1

    # 将句子转换为索引序列
    sentences_idx = [[word2idx.get(word, 0) for word in sentence.split()] for sentence in sentences]

    # 对句子进行填充，使其长度一致
    max_length = max(len(sentence) for sentence in sentences_idx)
    sentences_padded = [sentence + [1] * (max_length - len(sentence)) for sentence in sentences_idx]

    return sentences_padded, labels, word2idx

# 加载数据并划分训练集和测试集
sentences, labels, word2idx = preprocess_data('D:\PyCharmWorkSpeace\BERT-Classifation\dataset\datasets.csv')
X_train, X_test, y_train, y_test = train_test_split(sentences, labels, test_size=0.2, random_state=42)

# 将数据转换为PyTorch张量并移动到GPU
X_train = torch.LongTensor(X_train).to(device)
X_test = torch.LongTensor(X_test).to(device)
y_train = torch.LongTensor(y_train).to(device)
y_test = torch.LongTensor(y_test).to(device)

# 定义模型参数
vocab_size = len(word2idx)
embedding_dim = 100
hidden_dim = 128
output_dim = 1

# 初始化模型并移动到GPU
model = BiGRUModel(vocab_size, embedding_dim, hidden_dim, output_dim).to(device)
optimizer = optim.Adam(model.parameters())
criterion = nn.BCEWithLogitsLoss()

# 训练模型
def train(model, optimizer, criterion, X, y):
    model.train()
    optimizer.zero_grad()
    outputs = model(X)
    loss = criterion(outputs.squeeze(), y.float())
    loss.backward()
    optimizer.step()
    return loss.item()

# 预测函数
def predict(model, X):
    model.eval()
    with torch.no_grad():
        outputs = model(X)
        predictions = torch.round(torch.sigmoid(outputs)).squeeze()
    return predictions

# 计算准确率和F1值
def calculate_metrics(predictions, targets):
    accuracy = torch.mean((predictions == targets).float())
    f1 = f1_score(targets.cpu().numpy(), predictions.cpu().numpy())
    return accuracy.item(), f1

# 设置训练参数
num_epochs = 30
batch_size = 32
Train_Acc=[]
Train_Loss=[]
Train_F1=[]
Test_Acc=[]
l_epo=[]
# 训练模型
for epoch in range(num_epochs):
    print(f"Epoch {epoch + 1}/{num_epochs}")
    total_loss = 0
    total_predictions = []
    total_targets = []

    for i in range(0, len(X_train), batch_size):
        batch_X = X_train[i:i + batch_size]
        batch_y = y_train[i:i + batch_size]

        loss = train(model, optimizer, criterion, batch_X, batch_y)
        total_loss += loss

        predictions = predict(model, batch_X)
        total_predictions.extend(predictions)
        total_targets.extend(batch_y)

    train_loss = total_loss / (len(X_train) // batch_size)
    train_accuracy, train_f1 = calculate_metrics(torch.Tensor(total_predictions).to(device),
                                                 torch.Tensor(total_targets).to(device))
    print(f"Train Loss: {train_loss:.4f} | Train Accuracy: {train_accuracy:.4f} | Train F1: {train_f1:.4f}")

    # 在测试集上进行评估
    test_predictions = predict(model, X_test)
    test_accuracy, test_f1 = calculate_metrics(test_predictions, y_test)
    print(f"Test Accuracy: {test_accuracy:.4f} | Test F1: {test_f1:.4f}")
    Train_Acc.append(train_accuracy)
    Train_Loss.append(train_loss)
    Train_F1.append(train_f1)
    Test_Acc.append(test_accuracy)
    l_epo.append(epoch)
plt.plot(l_epo, Train_Acc, label='Train_Acc')
plt.plot(l_epo, Train_Loss, label='Train_Loss')
plt.plot(l_epo, Train_F1, label='train F1')
plt.plot(l_epo, Test_Acc, label='Test_Acc')
print(f'Train_Acc:{Train_Acc}')
print(f'Train_Loss:{Train_Loss}')
print(f'Train_F1:{Train_F1}')
print(f'Test_Acc:{Test_Acc}')
plt.ylabel('Metric value')
plt.xlabel('Epoch')
plt.legend()
plt.show()
# plt.savefig('pic/BiGRU.png')