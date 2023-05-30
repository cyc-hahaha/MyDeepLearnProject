import torch
import torch.nn as nn
from matplotlib import pyplot as plt
from tqdm import tqdm
from transformers import logging, AutoTokenizer, AutoModel
import os

from config import get_config
from data import load_dataset
from model import Transformer, Gru_Model, BiLstm_Model, Lstm_Model, Rnn_Model, TextCNN_Model, Transformer_CNN_RNN, \
    BiGru_Model, Transformer_CNN_BiLSTM


class Niubility:
    def __init__(self, args, logger):
        self.args = args
        self.logger = logger
        self.logger.info('> creating model {}'.format(args.model_name))
        # Create model
        if args.model_name == 'bert':
            self.tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
            self.input_size = 768
            base_model = AutoModel.from_pretrained('bert-base-uncased')
        elif args.model_name == 'roberta':
            self.tokenizer = AutoTokenizer.from_pretrained('roberta-base', add_prefix_space=True)
            self.input_size = 768
            base_model = AutoModel.from_pretrained('roberta-base')
        elif args.model_name == 'codebert':
            self.tokenizer = AutoTokenizer.from_pretrained("microsoft/codebert-base")
            self.input_size = 768
            base_model = AutoModel.from_pretrained("microsoft/codebert-base")
        else:
            raise ValueError('unknown model')
        # Operate the method
        if args.method_name == 'fnn':
            self.Mymodel = Transformer(base_model, args.num_classes, self.input_size)
        elif args.method_name == 'gru':
            self.Mymodel = Gru_Model(base_model, args.num_classes, self.input_size)
        elif args.method_name == 'bigru':
            self.Mymodel = BiGru_Model(base_model, args.num_classes, self.input_size)
        elif args.method_name == 'lstm':
            self.Mymodel = Lstm_Model(base_model, args.num_classes, self.input_size)
        elif args.method_name == 'bilstm':
            self.Mymodel = BiLstm_Model(base_model, args.num_classes, self.input_size)
        elif args.method_name == 'rnn':
            self.Mymodel = Rnn_Model(base_model, args.num_classes, self.input_size)
        elif args.method_name == 'textcnn':
            self.Mymodel = TextCNN_Model(base_model, args.num_classes)
        elif args.method_name == 'lstm+textcnn':
            self.Mymodel = Transformer_CNN_RNN(base_model, args.num_classes)
        elif args.method_name == 'bilstm+textcnn':
            self.Mymodel = Transformer_CNN_BiLSTM(base_model, args.num_classes)
        else:
            raise ValueError('unknown method')

        self.Mymodel.to(args.device)
        if args.device.type == 'cuda':
            self.logger.info('> cuda memory allocated: {}'.format(torch.cuda.memory_allocated(args.device.index)))
        self._print_args()

    def _print_args(self):
        self.logger.info('> training arguments:')
        for arg in vars(self.args):
            self.logger.info(f">>> {arg}: {getattr(self.args, arg)}")

    def _train(self, dataloader, criterion, optimizer):
        train_loss, n_correct, n_train = 0, 0, 0
        tp, fp, fn, tn = 0, 0, 0, 0  # initialize TP, FP, FN and TN to 0
        # Turn on the train mode
        self.Mymodel.train()
        for inputs, targets in tqdm(dataloader, disable=self.args.backend, ascii='>='):
            inputs = {k: v.to(self.args.device) for k, v in inputs.items()}
            targets = targets.to(self.args.device)
            predicts = self.Mymodel(inputs)
            loss = criterion(predicts, targets)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * targets.size(0)
            n_correct += (torch.argmax(predicts, dim=1) == targets).sum().item()
            n_train += targets.size(0)

            # calculate TP, FP, and FN
            tp += ((torch.argmax(predicts, dim=1) == 1) & (targets == 1)).sum().item()
            fp += ((torch.argmax(predicts, dim=1) == 1) & (targets == 0)).sum().item()
            fn += ((torch.argmax(predicts, dim=1) == 0) & (targets == 1)).sum().item()
            tn += ((torch.argmax(predicts, dim=1) == 0) & (targets == 0)).sum().item()

        # calculate metrics
        self.logger.info(
            '[train] TP: {}, fp: {}, fn: {}, tn: {}'.format(tp, fp, fn, tn))
        accuracy = n_correct / n_train
        precision = tp / (tp + fp + 1e-8)
        recall = tp / (tp + fn + 1e-8)
        f1_score = 2 * precision * recall / (precision + recall + 1e-8)

        return train_loss / n_train, accuracy, recall, f1_score

    def _test(self, dataloader, criterion):
        test_loss, n_correct, n_test = 0, 0, 0
        tp, fp, fn = 0, 0, 0  # initialize TP, FP, and FN to 0
        # Turn on the eval mode
        self.Mymodel.eval()

        with torch.no_grad():
            for inputs, targets in tqdm(dataloader, disable=self.args.backend, ascii=' >='):
                inputs = {k: v.to(self.args.device) for k, v in inputs.items()}
                targets = targets.to(self.args.device)
                predicts = self.Mymodel(inputs)
                loss = criterion(predicts, targets)

                test_loss += loss.item() * targets.size(0)
                n_correct += (torch.argmax(predicts, dim=1) == targets).sum().item()
                n_test += targets.size(0)

                # calculate TP, FP, and FN
                tp += ((torch.argmax(predicts, dim=1) == 1) & (targets == 1)).sum().item()
                fp += ((torch.argmax(predicts, dim=1) == 1) & (targets == 0)).sum().item()
                fn += ((torch.argmax(predicts, dim=1) == 0) & (targets == 1)).sum().item()

        # calculate metrics

        accuracy = n_correct / n_test
        precision = tp / (tp + fp + 1e-8)
        recall = tp / (tp + fn + 1e-8)
        f1_score = 2 * precision * recall / (precision + recall + 1e-8)

        return test_loss / n_test, accuracy, recall, f1_score

    def run(self):
        train_dataloader, test_dataloader = load_dataset(tokenizer=self.tokenizer,
                                                         train_batch_size=self.args.train_batch_size,
                                                         test_batch_size=self.args.test_batch_size,
                                                         model_name=self.args.model_name,
                                                         method_name=self.args.method_name,
                                                         workers=self.args.workers)
        _params = filter(lambda x: x.requires_grad, self.Mymodel.parameters())
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.AdamW(_params, lr=self.args.lr, weight_decay=self.args.weight_decay)

        l_acc, l_trloss, l_teloss, l_epo = [], [], [], []
        l_train_rec, l_train_f1, l_test_rec, l_test_f1 = [], [], [], []
        l_train_acc, l_test_acc= [], []
        # Get the best_loss and the best_acc
        best_loss, best_acc = 0, 0

        name = '{}_{}_{}.model'.format(self.args.model_name, self.args.method_name,
                                          self.args.train_batch_size)
            #self.args.model_name + "_" + self.args.method_name + "_" + self.args.train_batch_size + "_" + self.args.num_epoch
        path = "model/" + name
        # Load saved model if exists
        if os.path.isfile(path):
            self.Mymodel.load_state_dict(torch.load(path))
            self.logger.info('Loaded saved model: {}'.format(name))

        for epoch in range(self.args.num_epoch):
            train_loss, train_acc, train_rec, train_f1 = self._train(train_dataloader, criterion, optimizer)
            test_loss, test_acc, test_rec, test_f1 = self._test(test_dataloader, criterion)
            l_epo.append(epoch), l_acc.append(test_acc), l_trloss.append(train_loss), l_teloss.append(test_loss)
            l_train_rec.append(train_rec), l_train_f1.append(train_f1), l_test_rec.append(test_rec), l_test_f1.append(
                test_f1)
            l_train_acc.append(train_acc)
            l_test_acc.append(test_acc)
            #保存模型
            # if test_acc > best_acc or (test_acc == best_acc and test_loss < best_loss):
            #     best_acc, best_loss = test_acc, test_loss
            #     # Save best model
            #     torch.save(self.Mymodel.state_dict(), path)
            #     self.logger.info("Saved best model:"+ name)
            self.logger.info(
                '{}/{} - {:.2f}%'.format(epoch + 1, self.args.num_epoch, 100 * (epoch + 1) / self.args.num_epoch))
            self.logger.info(
                '[train] loss: {:.4f}, acc: {:.2f}, rec: {:.2f}, f1: {:.2f}'.format(train_loss, train_acc * 100,
                                                                                    train_rec * 100, train_f1 * 100))
            self.logger.info(
                '[test] loss: {:.4f}, acc: {:.2f}, rec: {:.2f}, f1: {:.2f}'.format(test_loss, test_acc * 100,
                                                                                   test_rec * 100, test_f1 * 100))
        self.logger.info('best loss: {:.4f}, best acc: {:.2f}'.format(best_loss, best_acc * 100))
        self.logger.info('log saved: {}'.format(self.args.log_name))
        # Draw the training process
        self.logger.info(l_epo)
        self.logger.info(l_train_acc)
        self.logger.info(l_trloss)
        self.logger.info(l_train_f1)
        self.logger.info(l_test_acc)
        self.logger.info(l_teloss)
        self.logger.info(l_test_f1)
        plt.plot(l_epo, l_train_acc, label='train acc')
        plt.plot(l_epo, l_trloss, label='train loss')
        plt.plot(l_epo, l_train_f1, label='train F1')
        plt.plot(l_epo, l_test_acc, label='test acc')
        plt.plot(l_epo, l_teloss, label='test loss')
        plt.plot(l_epo, l_test_f1, label='test F1')
        plt.ylabel('Metric value')
        plt.xlabel('Epoch')
        plt.legend()
        plt.savefig('pic/BERT+bigru.png')




if __name__ == '__main__':
    logging.set_verbosity_error()
    args, logger = get_config()
    print(args.device.type)
    nb = Niubility(args, logger)
    nb.run()
