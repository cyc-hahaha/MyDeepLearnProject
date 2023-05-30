import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import pandas as pd
from torch.nn.utils.rnn import pad_sequence
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split
from matplotlib import pyplot as plt
# Step 1: 准备数据集
data = pd.read_csv('D:\PyCharmWorkSpeace\BERT-Classifation\dataset\datasets.csv')
labels = data['labels'].tolist()
sentences = data['sentences'].tolist()

# Step 2: 文本预处理
class TextDataset(Dataset):
    def __init__(self, sentences, labels):
        self.sentences = sentences
        self.labels = labels
        self.vocab, self.word2index = self.build_vocab()

    def __len__(self):
        return len(self.sentences)

    def __getitem__(self, index):
        sentence = self.sentences[index]
        label = self.labels[index]
        sentence_indices = self.sentence_to_indices(sentence)
        return sentence_indices, label

    def build_vocab(self):
        all_words = ' '.join(self.sentences).split()
        vocab = sorted(set(all_words))
        vocab.append('<UNK>')  # Add the <UNK> token to the vocabulary
        word2index = {word: idx for idx, word in enumerate(vocab)}
        return vocab, word2index

    def sentence_to_indices(self, sentence):
        indices = [self.word2index.get(word, self.word2index['<UNK>']) for word in sentence.split()]
        return torch.tensor(indices)

dataset = TextDataset(sentences, labels)

# Step 3: 划分训练集和测试集
train_sentences, test_sentences, train_labels, test_labels = train_test_split(sentences, labels, test_size=0.2, random_state=42)

train_dataset = TextDataset(train_sentences, train_labels)
test_dataset = TextDataset(test_sentences, test_labels)

# Step 4: 创建数据加载器
batch_size = 32
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def collate_fn(batch):
    inputs = [item[0] for item in batch]
    labels = [item[1] for item in batch]
    padded_inputs = pad_sequence(inputs, batch_first=True)
    return padded_inputs, torch.tensor(labels)

train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)

# Step 5: 定义SimpleRNN模型
# Step 5: 定义SimpleRNN模型
class RNNModel(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, output_dim):
        super(RNNModel, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.rnn = nn.RNN(embedding_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        embedded = self.embedding(x)
        output, hidden = self.rnn(embedded)
        hidden_last = hidden[-1]
        logits = self.fc(hidden_last)
        return logits

vocab_size = len(dataset.vocab)
embedding_dim = 50
hidden_dim = 64
output_dim = 2

model = RNNModel(vocab_size, embedding_dim, hidden_dim, output_dim)
model.to(device)


# Step 6: 定义损失函数和优化器
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters())

# Step 7: 训练模型
num_epochs = 50
Train_Acc=[]
Train_Loss=[]
Train_F1=[]
Test_Acc=[]
l_epo=[]
for epoch in range(num_epochs):
    model.train()
    train_loss = 0.0
    train_predictions = []
    train_labels = []

    for inputs, labels in train_dataloader:
        inputs = inputs.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        train_loss += loss.item() * inputs.size(0)

        _, predicted = torch.max(outputs.data, 1)
        train_predictions.extend(predicted.cpu().tolist())
        train_labels.extend(labels.cpu().tolist())

    train_loss /= len(train_dataset)
    train_accuracy = accuracy_score(train_labels, train_predictions)
    train_f1 = f1_score(train_labels, train_predictions)

    model.eval()
    test_loss = 0.0
    test_predictions = []
    test_labels = []

    with torch.no_grad():
        for inputs, labels in test_dataloader:
            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = model(inputs)
            loss = criterion(outputs, labels)
            test_loss += loss.item() * inputs.size(0)

            _, predicted = torch.max(outputs.data, 1)
            test_predictions.extend(predicted.cpu().tolist())
            test_labels.extend(labels.cpu().tolist())

    test_loss /= len(test_dataset)
    test_accuracy = accuracy_score(test_labels, test_predictions)
    test_f1 = f1_score(test_labels, test_predictions)
    Train_Acc.append(train_accuracy)
    Train_Loss.append(train_loss)
    Train_F1.append(train_f1)
    Test_Acc.append(test_accuracy)
    l_epo.append(epoch)
    print(f'Epoch [{epoch+1}/{num_epochs}], Train Loss: {train_loss:.4f}, Train Acc: {train_accuracy:.4f}, Train F1: {train_f1:.4f}, Test Loss: {test_loss:.4f}, Test Acc: {test_accuracy:.4f}, Test F1: {test_f1:.4f}')

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
plt.savefig('BiLSTM2.png')