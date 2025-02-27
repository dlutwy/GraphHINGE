import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import sys
import gc
sys.path.append("..")
from utils import data_loader,utils
import argparse
import random
import pickle as pkl
from tqdm import tqdm
import datetime
from time import time
from algo import GraphHINGE_FFT,GraphHINGE_Conv,GraphHINGE_Cross,GraphHINGE_CrossConv,GraphHINGE_ALL, GraphHINGE_ConvALL, GraphHINGE_CNN
from tensorboardX import SummaryWriter
import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score


class dummy_context_mgr():
    def __enter__(self):
        return None
    def __exit__(self, exc_type, exc_value, traceback):
        return False

def train(model, device, optimizer, train_loader, eval_loader, save_dir, epochs, writer, model_name, patience):
    best_val_acc = 0.0
    cnt = 0
    inference = eval_loader is None
    for epoch in range(epochs):
        #train
        model.train()
        train_loss = []
        pbar = tqdm(train_loader)
        train_acc = []
        for i, data in enumerate(pbar):
            UI, IU, UIUI, IUIU, UIAI1, IAIU1, UIAI2, IAIU2, UIAI3, IAIU3, labels = data
            optimizer.zero_grad()
            pred = model(UI.to(device), IU.to(device),\
            UIUI.to(device), IUIU.to(device), \
            UIAI1.to(device), IAIU1.to(device), \
            UIAI2.to(device), IAIU2.to(device), \
            UIAI3.to(device), IAIU3.to(device))

            loss = criterion(pred, labels.to(device))
            acc = utils.evaluate_acc(pred.detach().cpu().numpy(), labels.numpy())
            train_acc.append(acc)
            loss.backward()
            optimizer.step()
            
            train_loss.append(loss.item())
            pbar.set_description('Epoch {:d} | Loss {:.4f}'.format(epoch + 1, loss))
            train_iter= epoch * len(train_loader) + i
            if train_iter%10 ==0:
                writer.add_scalar('Loss/train', loss.item(), train_iter)
                
            '''
            if i%500==0:
                print('Epoch {:d} | Batch {:d} | Train Loss {:.4f} | '.format(epoch + 1, i+1,loss))
                

            del UI, IU, UIUI, IUIU, UIAI1, IAIU1, UIAI2, IAIU2, UIAI3, IAIU3, labels, pred
            gc.collect()
            with torch.cuda.device(device):
                torch.cuda.empty_cache()
            '''
        train_acc = np.mean(train_acc)
        writer.add_scalar('Acc/Train', train_acc, train_iter)
        #eval
        if inference:
            continue
        model.eval()
        eval_loss = []
        eval_acc = []
        eval_auc = []
        eval_logloss = []
        eval_f1 = []
        with torch.no_grad():
            pbar = tqdm(eval_loader)
            for i, data in enumerate(pbar):
                UI, IU, UIUI, IUIU, UIAI1, IAIU1, UIAI2, IAIU2, UIAI3, IAIU3, labels = data
                
                pred = model(UI.to(device), IU.to(device),\
                UIUI.to(device), IUIU.to(device), \
                UIAI1.to(device), IAIU1.to(device), \
                UIAI2.to(device), IAIU2.to(device), \
                UIAI3.to(device), IAIU3.to(device))

                loss = criterion(pred, labels.to(device))
                acc = utils.evaluate_acc(pred.detach().cpu().numpy(), labels.numpy())
                eval_loss.append(loss.item())
                eval_acc.append(acc)
                pbar.set_description('Epoch {:d} | Loss {:.4f}'.format(epoch + 1, loss))

                '''
                del UI,IU, UIUI, IUIU, UIAI1, IAIU1, UIAI2, IAIU2, UIAI3, IAIU3, labels, pred
                gc.collect()
                with torch.cuda.device(device):
                    torch.cuda.empty_cache()
                '''

            eval_loss = np.mean(eval_loss)
            eval_acc = np.mean(eval_acc)


            print('Epoch {:d} | Eval Acc {:.4f} | '.format(epoch + 1, eval_acc))
            writer.add_scalar('Loss/Eval', eval_loss, train_iter)
            writer.add_scalar('Acc/Eval', eval_acc, train_iter)
            if eval_acc > best_val_acc:
                print(f"Prev best Acc {best_val_acc:.4f}, new best {eval_acc:.4f}. Saving best weights...")
                best_val_acc = eval_acc
                best_epoch = epoch
                torch.save(model.state_dict(), os.path.join(save_dir,model_name))
                cnt = 0
            else: 
                cnt = cnt + 1
                if cnt >= patience:
                    print("Early stopping")
                    print("best epoch:{}".format(best_epoch))
                    break
    if inference:
        torch.save(model.state_dict(), os.path.join(save_dir,model_name))


def test(model, device, test_loader, writer, inference_output = None):
    test_loss = []
    test_logloss = []
    pred_list = None
    label_list = None
    model.eval()
    with torch.no_grad():
        for i, data in enumerate(tqdm(test_loader, desc = 'Infer')):
            UI, IU, UIUI, IUIU, UIAI1, IAIU1, UIAI2, IAIU2, UIAI3, IAIU3, labels = data
            pred = model(UI.to(device), IU.to(device),\
            UIUI.to(device), IUIU.to(device), \
            UIAI1.to(device), IAIU1.to(device), \
            UIAI2.to(device), IAIU2.to(device), \
            UIAI3.to(device), IAIU3.to(device))
        
            loss = criterion(pred, labels.to(device))

            pred = pred.detach().cpu().numpy()
            labels = labels.numpy()
            users = UI.detach().cpu().numpy()[:, 0, 0]
            items = IU.detach().cpu().numpy()[:, 0, 0]
            if pred_list is None:
                pred_list = pred
                label_list = labels
            else:
                pred_list = np.vstack([pred_list, pred])
                label_list = np.vstack([label_list, labels])
            if inference_output is not None:
                for user, item, logit, label in zip(users, items, pred, labels):
                    print(user, item, logit[0].item(), sep = '\t', file = inference_output)
            logloss = utils.evaluate_logloss(pred, labels)
            test_loss.append(loss.item())
            test_logloss.append(logloss)
            writer.add_scalar('Loss/Test', loss.item(), i)
            writer.add_scalar('Logloss/Test', logloss, i)
            '''
            
            del UI, IU, UIUI, IUIU, UIAI1, IAIU1, UIAI2, IAIU2, UIAI3, IAIU3, labels, pred
            gc.collect()
            
            with torch.cuda.device(device):
                torch.cuda.empty_cache()
            '''
        test_auc = utils.evaluate_auc(pred_list, label_list)
        test_acc = utils.evaluate_acc(pred_list, label_list)
        test_aupr = average_precision_score(label_list, pred_list)

        test_f1 = utils.evaluate_f1_score(pred_list, label_list)
        
        test_loss = np.mean(test_loss)
        test_logloss = np.mean(test_logloss)
        writer.add_scalar('AUC/Test', test_auc, i)
        writer.add_scalar('Acc/Test', test_acc, i)
        writer.add_scalar('F1/Test', test_f1, i)
        print('Test Loss {:.4f} | Test AUC {:.4f} | Test ACC {:.4f} | Test F1 {:.4f} | AUPR {:.4f} | Test Logloss {:.4f} |'.format(test_loss, test_auc, test_acc, test_f1, test_aupr, test_logloss))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Parameters.')
    parser.add_argument('-cuda', type=str, default='cpu' ,help='Cuda number.')
    parser.add_argument('-hidden', type=int, default=128, help='Hidden Units.')
    parser.add_argument('-heads', type=int, default=3, help='Attention heads.')
    parser.add_argument('-lr', type=float, default=0.001, help='Learning rate.')
    parser.add_argument('-wd', type=float, default=0.0009, help='Weight decay.')
    parser.add_argument('-epochs', type=int, default=500,help='Maximum Epoch.')

    parser.add_argument('-model', type=str, default='GraphHINGE_FFT',help='Model.')
    parser.add_argument('-save_dir', type=str, default='../out',help='Trained models to be saved.')
    parser.add_argument('-pretrained', type=str, default='pretrained.pth',help='Load a model from breakpoint.')

    parser.add_argument('-d', type=str,help='Dataset.')
    parser.add_argument('-p', type=str,help='Dataset path.')
    parser.add_argument('-o', type=str,default='../data/data_out',help='Output path.')
    parser.add_argument('-b', type=int, default=128, help='Batch size.')
    parser.add_argument('-s', type=int,default=0, help='Sample path to be saved.')
    parser.add_argument('-n', type=int, default=16,help='Num of walks per node.')
    parser.add_argument('-w', type=int, default=1,help='Scale of walk length.')
    parser.add_argument('-temp1', type=float, default=0.2, help='Temperature factor for node att.')
    parser.add_argument('-temp2', type=float, default=0.2, help='Temperature factor for path att.')
    parser.add_argument('-ratio', type=float, default=1, help='Sample ratio.')
    parser.add_argument('-train', action= "store_true", default= False,help='If false, model will be loaded from the latest saved one.')
    parser.add_argument('-usersid', type=int, action="append", default=None, help='Specify users id to inference. Default all users.')
    parser.add_argument('-itemsid', type=int, action="append", default=None, help='Specify items id to inference. Default all items.')
    parser.add_argument('-inference', action= "store_true", default=False,help='Inference stage. (Train All data and inference new links between users and items)')
    parser.add_argument('-model_state_file', type=str,default = None, help='Load model state file to infer')
    parser.add_argument('-use_tensorboard', action= "store_true",default = False)
    parser.add_argument('-kfold', action= "store_true",default = False)
    args = parser.parse_args() 
    device = torch.device(args.cuda)

    fold_idx = -1 if args.kfold else None
    args.o += "_fold" if args.kfold else ""
    args.o += "_inference" if args.inference else ""

    for user_metas, item_metas, train_loader, eval_loader, test_loader, user_num, item_num, attr1_num, attr2_num, attr3_num in \
        data_loader.Dataloader(args.p, args.d, args.o, args.s, args.n, args.w, args.b, args.inference,ratio=args.ratio, usersid = args.usersid, itemsid = args.itemsid, kfold = args.kfold).load_data():
        print('Data Loaded!')
        if args.kfold:
            fold_idx += 1
        if args.model == 'GraphHINGE_FFT':
            model = GraphHINGE_FFT.GraphHINGE(
                user_num, item_num, attr1_num, attr2_num, attr3_num, args.hidden, args.hidden, args.hidden, args.heads, args.temp1, args.temp2, output_attentions = False
                ).to(device)
        elif args.model == 'GraphHINGE_Conv':
            model = GraphHINGE_Conv.GraphHINGE(
                user_num, item_num, attr1_num, attr2_num, attr3_num, args.hidden, args.hidden, args.hidden, args.heads, args.temp1, args.temp2
                ).to(device)
        elif args.model == 'GraphHINGE_Cross':
            model = GraphHINGE_Cross.GraphHINGE(
                user_num, item_num, attr1_num, attr2_num, attr3_num, args.hidden, args.hidden, args.hidden, args.heads, args.temp1, args.temp2
                ).to(device)
        elif args.model == 'GraphHINGE_CrossConv':
            model = GraphHINGE_CrossConv.GraphHINGE(
                user_num, item_num, attr1_num, attr2_num, attr3_num, args.hidden, args.hidden, args.hidden, args.heads, args.temp1, args.temp2
                ).to(device)
        elif args.model == 'GraphHINGE_ALL':
            model = GraphHINGE_ALL.GraphHINGE(
                user_num, item_num, attr1_num, attr2_num, attr3_num, args.hidden, args.hidden, args.hidden, args.heads, args.temp1, args.temp2
                ).to(device)
        elif args.model == 'GraphHINGE_ConvALL':
            model = GraphHINGE_ConvALL.GraphHINGE(
                user_num, item_num, attr1_num, attr2_num, attr3_num, args.hidden, args.hidden, args.hidden, args.heads, args.temp1, args.temp2
                ).to(device)
        elif args.model == 'GraphHINGE_CNN':
            model = GraphHINGE_CNN.GraphHINGE_CNN(
                user_num, item_num, attr1_num, attr2_num, attr3_num, args.hidden, args.hidden, args.hidden, args.heads, args.temp1, args.temp2
                ).to(device)
        else:
            raise NotImplementedError
        print('Model Created!')
        
        if (os.path.exists(os.path.join(args.save_dir, args.pretrained))):
            model.load_state_dict(torch.load(os.path.join(args.save_dir, args.pretrained)))
            print("Model loaded!")
        
        criterion = nn.BCELoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.wd)
        if args.model_state_file is None:
            model_name = "NotFFT" + '_'.join([args.model, 
                                        args.d, ("inference" if args.inference else "train"),
                                        f"Fold{fold_idx}" if args.kfold else "Nofold",
                                        "lr_"+str(args.lr), "bs_"+str(args.b), "wd_"+str(args.wd)]) \
                                    + '.pth' 
        else:
            model_name = args.model_state_file

        save_dir = args.save_dir
        startDateTime = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        log_dir = save_dir+'/Logs/'+ startDateTime + model_name
        print("Args ", args)
        if not os.path.exists(log_dir):
            os.mkdir(log_dir)
        with  SummaryWriter(log_dir = log_dir) as writer:
            if args.train:
                train(model, device, optimizer, train_loader, eval_loader, args.save_dir, args.epochs, writer, model_name,patience=50)
                pass
            model.load_state_dict(torch.load(os.path.join(args.save_dir, model_name)))
            if args.inference:
                if not os.path.exists(os.path.join(save_dir, "inference")):
                    os.mkdir(os.path.join(save_dir, "inference"))
                with open(os.path.join(save_dir, "inference", f"{startDateTime}.csv"), 'w') as infer_out:
                    test(model, device, test_loader, writer, infer_out)
            else:
                test(model, device, test_loader, writer)