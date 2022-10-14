import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from My_Model import get_model
# from My_Model_Residual import get_model
from My_Pytorch_Pretreatment import My_DataSet
from My_Opt import opt
from My_Tools import evalute
from tqdm import *


def main():
    train_opt = opt
    loss_result = 0
    # train_opt['load_weights'] = 0
    # train_opt['dropout'] = 0.1
    train_opt['load_weights'] = 1
    best_acc, best_acc_epoch = 0, 0

    print('Loading DataSet...')
    train_data = My_DataSet(train_opt['train_data_path'], train_opt['device'], train_opt['n_class'])
    val_data = My_DataSet(train_opt['val_data_path'], train_opt['device'], train_opt['n_class'])
    subtrain_data = My_DataSet(train_opt['subtrain_data_path'], train_opt['device'], train_opt['n_class'])
    # ------------训练测试集换---------------------
    train_dataloader = DataLoader(train_data, batch_size=train_opt['batch_size'], drop_last=False, shuffle=True)
    val_dataloader = DataLoader(val_data, batch_size=train_opt['batch_size'], drop_last=False, shuffle=True)
    subtrain_dataloader = DataLoader(subtrain_data, batch_size=train_opt['batch_size'], drop_last=True, shuffle=True)
    # -------------------------------------------
    print('Finished!')
    print('Loading model...')
    model = get_model(train_opt)
    # if train_opt['device'] == 'cuda':
    model = model.to(device=train_opt['device'], dtype=torch.float64)
    print('Finished!')
    criterion = nn.CrossEntropyLoss()
    # if train_opt['device'] == 'cuda':
    criterion = criterion.to(train_opt['device'])
    optimizer = optim.Adam(model.parameters(), lr=train_opt['lr'])
    print('Training Begin...')
    with open(train_opt['weight_logs'], 'w') as logs:
        logs.write('num,train_acc,val_acc,loss\n')
    for epoch in range(train_opt['epochs']):
        for seg_cut_in_8, seg_cut_in_6, seg_cut_in_4, global_data, label in tqdm(train_dataloader):
            output = model(seg_cut_in_8, seg_cut_in_6, seg_cut_in_4, global_data)
            loss = criterion(output, label)
            loss_result = loss
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
        print('Epoch:', '%04d' % (epoch + 1), 'loss =', '{:.6f}'.format(loss_result))  # 'accuracy=%f' % loss_result)#
        # if epoch % 2 == 0:
        #     val_acc = evalute(model, val_dataloader)
        #     train_acc = evalute(model, train_dataloader)
        #     print('val_acc:', ' {:.4%}'.format(val_acc), 'train_acc:', '{:.4%}'.format(train_acc))
        #     if val_acc > best_acc:
        #         best_acc = val_acc
        #         best_acc_epoch = epoch
        #         torch.save(model.state_dict(), os.path.join(train_opt['load_state_dict'], '{}'.format(best_acc_epoch) + ',' + '{:.3%}'.format(best_acc) + ',' + '{:.6f}'.format(loss_result) + '.mdl'))
        val_acc = evalute(model, val_dataloader)
        train_acc = evalute(model, subtrain_dataloader)
        print('val_acc:', ' {:.4%}'.format(val_acc), 'train_acc:', '{:.4%}'.format(train_acc))
        with open(train_opt['weight_logs'], 'a') as logs:
            logs.write('{},{:.4%},{:.4%},{}\n'.format(epoch, train_acc, val_acc, loss_result))
        if val_acc > best_acc:
            best_acc = val_acc
            best_acc_epoch = epoch
            torch.save(model.state_dict(), os.path.join(train_opt['save_state_dict'],'{}'.format(best_acc_epoch) + ',' + '{:.3%}'.format(best_acc) + ',' + '{:.6f}'.format(loss_result) + '.mdl'))
            # torch.save(model, os.path.join(train_opt['load_state_dict'],
            #                                             '{}'.format(best_acc_epoch) + ',' + '{:.3%}'.format(
            #                                                 best_acc) + ',' + '{:.6f}'.format(loss_result) + '.mdl'))

    print('best acc:', best_acc, 'best acc epoch:', best_acc_epoch)
    print('Finished!')


if __name__ == "__main__":
    main()