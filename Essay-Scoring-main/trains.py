import os
import sys
import argparse
import random
import time
import numpy as np
from utils import *
from metrics import *
from utils import rescale_tointscore
from utils import domain_specific_rescale
import data_prepare
from hierarchical_att_model import HierAttNet
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torch.utils.data as Data
from reader import *

logger = get_logger("Train sentence sequences Recurrent Convolutional model (LSTM stack over CNN)")

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

def get_augmentation_methods():
    methods = []
    for file in os.listdir('./datas'):
        if file.startswith('train_') and file.endswith('_augmented.tsv'):
            method = file[6:-14]  # Extract method name from filename
            methods.append(method)
    return methods

def main():
    parser = argparse.ArgumentParser(description="sentence Hi_CNN model")
    parser.add_argument('--embedding', type=str, default='word2vec', help='Word embedding type, word2vec, senna or glove')
    parser.add_argument('--embedding_dict', type=str, default=None, help='Pretrained embedding path')
    parser.add_argument('--embedding_dim', type=int, default=64, help='Only useful when embedding is randomly initialised')

    parser.add_argument('--num_epochs', type=int, default=50, help='number of epochs for training')
    parser.add_argument('--batch_size', type=int, default=10, help='Number of texts in each batch')
    parser.add_argument("-v", "--vocab-size", dest="vocab_size", type=int, metavar='<int>', default=4000, help="Vocab size (default=4000)")

    parser.add_argument('--oov', choices=['random', 'embedding'], help="Embedding for oov word", required=True)

    parser.add_argument('--optimizer', choices=['sgd', 'momentum', 'nesterov', 'adagrad', 'rmsprop'], help='updating algorithm', default='sgd')
    parser.add_argument('--learning_rate', type=float, default=0.001, help='Initial learning rate')
    parser.add_argument('--dropout', type=float, default=0.5, help='Dropout rate for layers')

    parser.add_argument('--datapath', type=str, default='./datas/')
    parser.add_argument('--prompt_id', type=int, default=1, help='prompt id of essay set')

    args = parser.parse_args()
    if torch.cuda.is_available():
        torch.cuda.manual_seed(123)
    else:
        torch.manual_seed(123)

    augmentation_methods = get_augmentation_methods()
    results = {method: [] for method in augmentation_methods}
    dev_path = os.path.join(args.datapath, 'dev.tsv')
    test_path = os.path.join(args.datapath, 'test.tsv')
    embedding_path = args.embedding_dict
    oov = args.oov
    embedding = args.embedding
    embedd_dim = args.embedding_dim
    prompt_id = args.prompt_id

    # 遍历每个增强方法
    for method in augmentation_methods:
        print(f"Training with augmentation method: {method}")    

        train_path = os.path.join(args.datapath, f'train_{method}_augmented.tsv')
        datapaths = [train_path, dev_path, test_path]
        vocab = create_vocab(datapaths[0],prompt_id,0,True,True)
        (X_train, Y_train, mask_train,train_pmt), (X_dev, Y_dev, mask_dev,dev_pmt), (X_test, Y_test, mask_test,test_pmt), \
                     embed_table, overal_maxlen, overal_maxnum, init_mean_value = prepare_sentence_data(datapaths, vocab,\
                    embedding_path, embedding, embedd_dim, prompt_id, tokenize_text=True, \
                    to_lower=True, sort_by_len=False,  score_index=6)        
        max_sentnum = overal_maxnum
        max_sentlen = overal_maxlen
        Y_train= torch.tensor(Y_train)
        Y_dev = torch.tensor(Y_dev)
        Y_test= torch.tensor(Y_test)
        X_train= torch.LongTensor(X_train)
        X_dev = torch.LongTensor(X_dev)
        X_test= torch.LongTensor(X_test)
        train_data = Data.TensorDataset(X_train, Y_train)
        dev_data = Data.TensorDataset(X_dev, Y_dev)
        test_data = Data.TensorDataset(X_test,Y_test)
        train_loader = Data.DataLoader(dataset=train_data, batch_size=args.batch_size, shuffle=True, num_workers=3)
        dev_loader = Data.DataLoader(dataset=dev_data, batch_size=args.batch_size, shuffle=True, num_workers=3)
        test_loader = Data.DataLoader(dataset=test_data, batch_size=args.batch_size, shuffle=True, num_workers=3)

        # 初始化模型
        model = HierAttNet(100, 100, 10, embed_table, max_sentnum, max_sentlen)
        if torch.cuda.is_available():
            model.cuda()
        # 初始化损失函数和优化器
        criterion = nn.MSELoss()
        optimizer = torch.optim.RMSprop(filter(lambda p: p.requires_grad, model.parameters()), lr=args.learning_rate, alpha=0.9)
        
        model.train()
        p = 0
        num_iter_per_epoch = len(train_loader)

        for epoch in range(args.num_epochs):
            print(f"\repoch:{epoch+1}/{args.num_epochs} begin train...", end="")
            # 训练模型
            for iter, (feature, label) in enumerate(train_loader):
                # print(len_train,label)
                if torch.cuda.is_available():
                    feature = feature.cuda()
                    label = label.cuda()
                optimizer.zero_grad()
                model._init_hidden_state()
                predictions = model(feature)
                loss = criterion(predictions, label)
                loss.backward()
                optimizer.step()
            #print("loss:", loss)
            print(f"\repoch:{epoch+1}/{args.num_epochs} end train->begin evaluate...", end="")

            # 计算dev集的指标
            if epoch >= 0:
                model.eval()
                loss_ls = []
                te_label_ls = []
                te_pred_ls = []
                for te_feature, te_label in dev_loader:
                    num_sample = len(te_label)
                    if torch.cuda.is_available():
                        te_feature = te_feature.cuda()
                        te_label = te_label.cuda()
                    with torch.no_grad():
                        model._init_hidden_state(num_sample)
                        te_predictions = model(te_feature)
                    te_loss = criterion(te_predictions, te_label)
                    loss_ls.append(te_loss * num_sample)
                    te_label_ls.extend(te_label.clone().cpu())
                    te_pred_ls.extend(te_predictions.clone().cpu())

            te_label = np.array(te_label_ls)
            predictions = convert_to_dataset_friendly_scores(np.array(te_pred_ls), prompt_id)
            q1 = quadratic_weighted_kappa(predictions, te_label)
            p1 = pearson(predictions, te_label)[0]
            s1 = spearman(predictions, te_label)

            results[method].append((q1, p1, s1,sum(loss_ls)))

            #print(
                #"dev  Epoch: {}/{}, Iteration: {}/{}, loss : {},quadratic_weighted_kappa: {}, pearson: {}, spearman: {}".format(
                    #epoch + 1, args.num_epochs,
                    #iter + 1,num_iter_per_epoch, 
                    #sum(loss_ls),
                    #q1, p1, s1))
            print(f"\repoch:{epoch+1}/{args.num_epochs} end evaluate->begin test...", end="")

            # 计算test集的指标
            loss_ls = []
            te_label_ls = []
            te_pred_ls = []            
            for te_feature, te_label in test_loader:
                num_sample = len(te_label)
                if torch.cuda.is_available():
                    te_feature = te_feature.cuda()
                    te_label = te_label.cuda()
                with torch.no_grad():
                    model._init_hidden_state(num_sample)
                    te_predictions = model(te_feature)
                te_loss = criterion(te_predictions, te_label)
                loss_ls.append(te_loss * num_sample)
                te_label_ls.extend(te_label.clone().cpu())
                te_pred_ls.extend(te_predictions.clone().cpu())

            te_label = np.array(te_label_ls)
            predictions = convert_to_dataset_friendly_scores(np.array(te_pred_ls), prompt_id)
            q2 = quadratic_weighted_kappa(predictions, te_label)
            p2 = pearson(predictions, te_label)[0]
            s2 = spearman(predictions, te_label)

            #print(
                #"test  Epoch: {}/{}, Iteration: {}/{}, loss: {},quadratic_weighted_kappa: {}, pearson: {}, spearman: {}".format(
                    #epoch + 1, args.num_epochs, iter + 1, num_iter_per_epoch, sum(loss_ls), q2, s2, p2))
            print(f"\repoch:{epoch+1}/{args.num_epochs} end test...QWK:{q2}, Pearson:{p2}, Spearman:{s2}", end="\n")

            # 保存最好的模型
            if q1 > p:
                p = q1
                q3 = q2
                p3 = p2
                s3 = s2
                torch.save(model, f'{method}.pkl')
                #print("best result Epoch : {},quadratic_weighted_kappa: {}, pearson: {}, spearman: {}".format(epoch + 1,q2, p2,s2))
            model.train()
        #print("best result Epoch : {},quadratic_weighted_kappa: {}, pearson: {}, spearman: {}".format(epoch + 1, q3, p3,s3))
    # 打印结果表格
    print_results_table(results)

def print_results_table(results):
    # 定义表头
    header1 = ["Method", "Best", "", "","", "Avg", "", ""]
    header2 = ["", "QWK", "Pearson", "Spearman","Loss", "QWK", "Pearson", "Spearman"]
    
    # 计算每列的最大宽度
    col_widths = [max(max(len(str(x)),10) for x in col) for col in zip(header1, header2)]
    col_widths[0] = max(col_widths[0], max(len(method) for method in results.keys()))
    
    # 打印表头
    print_separator(col_widths)
    print_row(header1, col_widths)
    print_row(header2, col_widths)
    print_separator(col_widths)
    
    # 打印数据行
    for method, scores in results.items():
        if len(scores) == 0:
            continue
        best_score = max(scores, key=lambda s: s[0])
        avg_qwk = np.mean([s[0] for s in scores])
        avg_pearson = np.mean([s[1] for s in scores])
        avg_spearman = np.mean([s[2] for s in scores])
        row = [
            method,
            f"{best_score[0]:.8f}",
            f"{best_score[1]:.8f}",
            f"{best_score[2]:.8f}",
            f"{best_score[3]:.8f}",
            f"{avg_qwk:.8f}",
            f"{avg_pearson:.8f}",
            f"{avg_spearman:.8f}"
        ]
        print_row(row, col_widths)
    
    print_separator(col_widths)

def print_row(row, widths):
    print("|" + "|".join(f" {str(x).ljust(w)} " for x, w in zip(row, widths)) + "|")

def print_separator(widths):
    print("+" + "+".join("-" * (w + 2) for w in widths) + "+")

if __name__ == '__main__':
    main()