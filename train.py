import os
import sys
from data.protein_dataset import *
from torch.utils.data import DataLoader
from torch.autograd import Variable
from model import transPPI
from sklearn.model_selection import KFold
from evaluation import analysis
import pandas as pd
import time
from utils import *

LOSS2_ALPHA = 2E-1
LOSS3_beta = 5E-1
BATCH_SIZE = 1
LEARNING_RATE = 2E-3

NUMBER_EPOCHS = 100
loss2fn = SupConLoss()

loss1fn = nn.CrossEntropyLoss()  # automatically do softmax to the predicted value and one-hot to the label


def train_one_epoch(model, optimizer , data_loader):
    epoch_loss_train = 0.0
    n = 0
    for data in data_loader:
        sequence_name, _, labels, str_G_batch, seq_G_batch, seqid = data

        if torch.cuda.is_available():
            str_G_batch = str_G_batch.to(torch.device('cuda:0'))
            seq_G_batch = seq_G_batch.to(torch.device('cuda:0'))

            # adj_matrix = Variable(adj_matrix.cuda())
            labels = torch.cat([torch.tensor(arr) for arr in labels])
            y_true = Variable(labels.cuda())
            # print(node_features.shape)
        else:
            # adj_matrix = Variable(adj_matrix)
            y_true = Variable(labels)

        # adj_matrix = torch.squeeze(adj_matrix)
        y_true = torch.squeeze(y_true)
        y_true = y_true.long()
        y_pred, struct_attention, seq_attention, x = model(str_G_batch, seq_G_batch, seqid)
        # calculate loss

        loss1 = loss1fn(y_pred, y_true)
        loss2 = loss2fn(x, y_true) 
        loss3 = mse_attention_loss(struct_attention, seq_attention)
        # backward gradient
        SumLoss = loss1 + LOSS2_ALPHA * loss2 + loss3

        optimizer.zero_grad()
        SumLoss.backward()
        optimizer.step()

        epoch_loss_train += SumLoss.item()
        n += 1

    epoch_loss_train_avg = epoch_loss_train / n
    return epoch_loss_train_avg


def evaluate(model, data_loader):
    model.eval()
    epoch_loss = 0.0
    n = 0
    valid_pred = []
    valid_true = []
    pred_dict = {}

    for data in data_loader:
        with torch.no_grad():
            sequence_name, _, labels, str_G_batch, seq_G_batch, seqid = data

            if torch.cuda.is_available():
                str_G_batch = str_G_batch.to(torch.device('cuda:0'))
                seq_G_batch = seq_G_batch.to(torch.device('cuda:0'))
                # adj_matrix = Variable(adj_matrix.cuda())
                labels = torch.cat([torch.tensor(arr) for arr in labels])
                y_true = Variable(labels.cuda())
                # print(node_features.shape)
            else:
                y_true = Variable(labels)

            y_true = torch.squeeze(y_true)
            y_true = y_true.long()
            y_pred, struct_attention, seq_attention, x = model(str_G_batch, seq_G_batch, seqid)

            loss1 = loss1fn(y_pred, y_true)
            loss2 = loss2fn(x, y_true)
            loss3 = mse_attention_loss(struct_attention, seq_attention)
            # backward gradient
            SumLoss = loss1 + LOSS2_ALPHA * loss2 + loss3
            softmax = torch.nn.Softmax(dim=1)
            y_pred = softmax(y_pred)
            y_pred = y_pred.cpu().detach().numpy()
            y_true = y_true.cpu().detach().numpy()
            valid_pred += [pred[1] for pred in y_pred]
            valid_true += list(y_true)
            pred_dict[sequence_name[0]] = [pred[1] for pred in y_pred]
            epoch_loss += SumLoss.item()
            n += 1

    epoch_loss_avg = epoch_loss / n

    return epoch_loss_avg, valid_true, valid_pred, pred_dict


# train_dataframe = {"ID": IDs, "sequence": sequences, "seq_array":seq_array, "label": labels}
def train(model, optimizer, scheduler, train_dataframe, valid_dataframe, fold=0):
    train_loader = DataLoader(dataset=ProDataset(train_dataframe), batch_size=BATCH_SIZE, shuffle=True, num_workers=2,
                              collate_fn=graph_collate)
    valid_loader = DataLoader(dataset=ProDataset(valid_dataframe), batch_size=BATCH_SIZE, shuffle=True, num_workers=2,
                              collate_fn=graph_collate)

    best_epoch = 0
    best_val_auc = 0
    best_val_aupr = 0

    for epoch in range(NUMBER_EPOCHS):
        print("\n========== Train epoch " + str(epoch + 1) + " ==========")
        model.train()
        epoch_loss_train_avg = train_one_epoch(model, optimizer,train_loader)
        print("========== Evaluate Train set ==========")
        _, train_true, train_pred, _ = evaluate(model, train_loader)
        result_train = analysis(train_true, train_pred)
        print("Train loss: ", epoch_loss_train_avg)
        print("Train binary acc: ", result_train['binary_acc'])
        print("Train AUC: ", result_train['AUC'])
        print("Train AUPRC: ", result_train['AUPRC'])

        print("========== Evaluate Valid set ==========")
        epoch_loss_valid_avg, valid_true, valid_pred, _ = evaluate(model, valid_loader)
        result_valid = analysis(valid_true, valid_pred)
        print("Valid loss: ", epoch_loss_valid_avg)
        print("Valid binary acc: ", result_valid['binary_acc'])
        print("Valid precision: ", result_valid['precision'])
        print("Valid recall: ", result_valid['recall'])
        print("Valid f1: ", result_valid['f1'])
        print("Valid AUC: ", result_valid['AUC'])
        print("Valid AUPRC: ", result_valid['AUPRC'])
        print("Valid mcc: ", result_valid['mcc'])

        if best_val_aupr < result_valid['AUPRC']:
            best_epoch = epoch + 1
            best_val_auc = result_valid['AUC']
            best_val_aupr = result_valid['AUPRC']
            torch.save(model.state_dict(), os.path.join(Model_Path, 'Fold' + str(fold) + '_best_model.pkl'))

        scheduler.step(epoch_loss_valid_avg)
        print("epoch:, now learning rate: {}".format(scheduler.optimizer.param_groups[0]['lr']))

    return best_epoch, best_val_auc, best_val_aupr


def cross_validation(all_dataframe, fold_number=5):
    sequence_names = all_dataframe['ID'].values
    sequence_labels = all_dataframe['label'].values
    kfold = KFold(n_splits=fold_number, shuffle=True)
    fold = 0
    best_epochs = []
    valid_aucs = []
    valid_auprs = []

    for train_index, valid_index in kfold.split(sequence_names, sequence_labels):
        print("\n\n========== Fold " + str(fold + 1) + " ==========")
        train_dataframe = all_dataframe.iloc[train_index, :]
        valid_dataframe = all_dataframe.iloc[valid_index, :]
        print("Train on", str(train_dataframe.shape[0]), "samples, validate on", str(valid_dataframe.shape[0]),
              "samples")

        model = transPPI()

        optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=5e-4)

        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=10,
                                                               verbose=True)
        if torch.cuda.is_available():
            model.cuda()

        best_epoch, valid_auc, valid_aupr = train(model, optimizer, scheduler, train_dataframe, valid_dataframe,
                                                  fold + 1)
        best_epochs.append(str(best_epoch))
        valid_aucs.append(valid_auc)
        valid_auprs.append(valid_aupr)
        fold += 1

    print("\n\nBest epoch: " + " ".join(best_epochs))
    print("Average AUC of {} fold: {:.4f}".format(fold_number, sum(valid_aucs) / fold_number))
    print("Average AUPR of {} fold: {:.4f}".format(fold_number, sum(valid_auprs) / fold_number))
    return round(sum([int(epoch) for epoch in best_epochs]) / fold_number)


def train_full_model(all_dataframe, aver_epoch):
    print("\n\nTraining a full model using all training data...\n")
    model = transPPI()
    if torch.cuda.is_available():
        model.cuda()

    train_loader = DataLoader(dataset=ProDataset(all_dataframe), batch_size=BATCH_SIZE, shuffle=True, num_workers=4,
                              collate_fn=graph_collate)

    for epoch in range(NUMBER_EPOCHS):
        print("\n========== Train epoch " + str(epoch + 1) + " ==========")
        model.train()
        optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=5e-4)

        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=10,
                                                               verbose=True)
        epoch_loss_train_avg = train_one_epoch(model, optimizer, train_loader)
        scheduler.step(epoch_loss_train_avg)
        print("========== Evaluate Train set ==========")
        _, train_true, train_pred, _ = evaluate(model, train_loader)
        result_train = analysis(train_true, train_pred)
        print("Train loss: ", epoch_loss_train_avg)
        print("Train binary acc: ", result_train['binary_acc'])
        print("Train AUC: ", result_train['AUC'])
        print("Train AUPRC: ", result_train['AUPRC'])

        if epoch + 1 in [aver_epoch, 45]:
            torch.save(model.state_dict(), os.path.join(Model_Path, 'Full_model_{}.pkl'.format(epoch + 1)))


class Logger(object):
    def __init__(self, filename="Default.log"):
        self.terminal = sys.stdout
        self.log = open(filename, 'ab', buffering=0)

    def write(self, message):
        self.terminal.write(message)
        try:
            self.log.write(message.encode('utf-8'))
        except ValueError:
            pass

    def close(self):
        self.log.close()
        sys.stdout = self.terminal

    def flush(self):
        pass


def main():
    if not os.path.exists(Log_path): os.makedirs(Log_path)
    Dataset_Path = './data/'
    with open(Dataset_Path + "Train_335_df_esm.pkl", "rb") as f:
        df_esm = pickle.load(f)
    df_esm = df_esm[df_esm["protein_name"] != "2j3ra"]

    with open(Dataset_Path + "Train_335.pkl", "rb") as f:
        Train_335 = pickle.load(f)
        Train_335.pop('2j3rA')  # remove the protein with error sequence in the train dataset

    IDs, sequences, labels, res_array, esm_feature = [], [], [], [], []

    for ID in Train_335:
        IDs.append(ID)
        item = Train_335[ID]
        sequences.append(item[0])
        labels.append(item[1])
        res_array.append(item[2])
        esm_feature.append(df_esm[df_esm["protein_name"] == ID]["esm_feature"].values)

    train_dic = {"ID": IDs, "sequence": sequences, "label": labels, "res_array": res_array, "esm_feature": esm_feature}
    train_dataframe = pd.DataFrame(train_dic)

    aver_epoch = cross_validation(train_dataframe, fold_number=5)
    train_full_model(train_dataframe, aver_epoch)


if __name__ == "__main__":

    Log_path = './LOG'
    localtime = time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime())
    checkpoint_path = os.path.normpath(Log_path + "/" + localtime)
    os.makedirs(checkpoint_path)
    Model_Path = os.path.normpath(checkpoint_path + '/model')
    if not os.path.exists(Model_Path): os.makedirs(Model_Path)
    sys.stdout = Logger(os.path.normpath(checkpoint_path + '/training.log'))
    main()
    sys.stdout.log.close()
