import torch
import torch.utils.data as Data
from torch.utils.data.distributed import DistributedSampler
import torch.distributed
import numpy as np
import argparse
import time
from util import *
from MTGNN.trainer import Trainer
from MTGNN.net import gtnet
from SandwichGNN.sandwich_model import SandwichGNN
from SandwichGNN.dataset_meta_la import testDataset_metr_la, valDataset_metr_la, trainDataset_metr_la, get_scaler

def str_to_bool(value):
    if isinstance(value, bool):
        return value
    if value.lower() in {'false', 'f', '0', 'no', 'n'}:
        return False
    elif value.lower() in {'true', 't', '1', 'yes', 'y'}:
        return True
    raise ValueError(f'{value} is not a valid boolean value')


parser = argparse.ArgumentParser()

parser.add_argument('--device',type=str,default='cuda:0',help='')
parser.add_argument('--data',type=str,default='/home/hjl/deep_learning_workspace/data_metr_la/1212',help='data path')

parser.add_argument('--adj_data', type=str,default='data/adj_mx.pkl',help='adj data path')
parser.add_argument('--gcn_true', type=str_to_bool, default=True, help='whether to add graph convolution layer')
parser.add_argument('--buildA_true', type=str_to_bool, default=True,help='whether to construct adaptive adjacency matrix')
parser.add_argument('--load_static_feature', type=str_to_bool, default=False,help='whether to load static feature')
parser.add_argument('--cl', type=str_to_bool, default=True,help='whether to do curriculum learning')

parser.add_argument('--gcn_depth',type=int,default=2,help='graph convolution depth')
parser.add_argument('--num_nodes',type=int,default=207,help='number of nodes/variables')
parser.add_argument('--dropout',type=float,default=0.3,help='dropout rate')
parser.add_argument('--subgraph_size',type=int,default=20,help='k')
parser.add_argument('--node_dim',type=int,default=40,help='dim of nodes')
parser.add_argument('--dilation_exponential',type=int,default=1,help='dilation exponential')

parser.add_argument('--conv_channels',type=int,default=32,help='convolution channels')
parser.add_argument('--residual_channels',type=int,default=32,help='residual channels')
parser.add_argument('--skip_channels',type=int,default=64,help='skip channels')
parser.add_argument('--end_channels',type=int,default=128,help='end channels')


parser.add_argument('--in_dim',type=int,default=2,help='inputs dimension')
parser.add_argument('--seq_in_len',type=int,default=12,help='input sequence length')
parser.add_argument('--seq_out_len',type=int,default=12,help='output sequence length')

parser.add_argument('--layers',type=int,default=2,help='number of layers')
parser.add_argument('--batch_size',type=int,default=32,help='batch size')
parser.add_argument('--learning_rate',type=float,default=0.001,help='learning rate')
parser.add_argument('--weight_decay',type=float,default=0.0001,help='weight decay rate')
parser.add_argument('--clip',type=int,default=5,help='clip')
parser.add_argument('--step_size1',type=int,default=2500,help='step_size')
parser.add_argument('--step_size2',type=int,default=100,help='step_size')

parser.add_argument('--epochs',type=int,default=1, help='')
parser.add_argument('--print_every',type=int,default=50,help='')
parser.add_argument('--seed',type=int,default=101,help='random seed')
parser.add_argument('--save',type=str,default='./save/',help='save path')
parser.add_argument('--expid',type=int,default=1,help='experiment id')

parser.add_argument('--propalpha',type=float,default=0.05,help='prop alpha')
parser.add_argument('--tanhalpha',type=float,default=3,help='adj alpha')

parser.add_argument('--num_split',type=int,default=1,help='number of splits for graphs')

parser.add_argument('--runs',type=int,default=1,help='number of runs')
parser.add_argument('--local-rank', default=0, type=int)

args = parser.parse_args()
torch.set_num_threads(3)

local_rank = args.local_rank

torch.cuda.set_device(local_rank) # 设定cuda的默认GPU，每个rank不同
device = torch.device('cuda', args.local_rank)
torch.distributed.init_process_group(backend='nccl')

# dataset
train_dataset = trainDataset_metr_la()
val_dataset = valDataset_metr_la()
test_dataset = testDataset_metr_la()
print(len(train_dataset) + len(val_dataset) + len(test_dataset))

train_sampler = DistributedSampler(train_dataset)
train_loader = torch.utils.data.DataLoader(train_dataset, sampler=train_sampler,
                                           batch_size=int(args.batch_size/4))

# val_sampler = DistributedSampler(val_dataset)
# val_loader = torch.utils.data.DataLoader(val_dataset, sampler=val_sampler,
#                                          batch_size=int(args.batch_size/4))
#
# test_sampler = DistributedSampler(test_dataset)
# test_loader = torch.utils.data.DataLoader(test_dataset, sampler=test_sampler,
#                                          batch_size=int(args.batch_size/4))

# train_loader = Data.DataLoader(train_dataset, batch_size=args.batch_size,
#                                shuffle=True, num_workers=0, pin_memory=True)
val_loader = Data.DataLoader(val_dataset, batch_size=args.batch_size,
                             shuffle=False, num_workers=0, pin_memory=True)
test_loader = Data.DataLoader(test_dataset, batch_size=args.batch_size,
                              shuffle=False, num_workers=0, pin_memory=True)

scaler = get_scaler()

def main(runid):
    # torch.manual_seed(args.seed)
    # torch.backends.cudnn.deterministic = True
    # torch.backends.cudnn.benchmark = False
    # np.random.seed(args.seed)
    #load data
    # dataloader = load_dataset(args.data, args.batch_size, args.batch_size, args.batch_size)
    # scaler = dataloader['scaler']

    predefined_A = load_adj(args.adj_data)
    predefined_A = torch.tensor(predefined_A)-torch.eye(args.num_nodes)
    predefined_A = predefined_A.to(device)

    # if args.load_static_feature:
    #     static_feat = load_node_feature('data/sensor_graph/location.csv')
    # else:
    #     static_feat = None


    model = SandwichGNN(predefined_A=predefined_A
                        ).to(device)
    model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.local_rank],
                                                      output_device=args.local_rank, find_unused_parameters=True)

    print(args)
    # print('The recpetive field size is', model.receptive_field)
    nParams = sum([p.nelement() for p in model.parameters()])
    print('Number of model parameters is', nParams)

    engine = Trainer(model, args.learning_rate, args.weight_decay, args.clip, args.step_size1, args.seq_out_len, scaler, device, args.cl)

    print("start training...",flush=True)
    his_loss =[]
    val_time = []
    train_time = []
    minl = 1e5
    for i in range(1,args.epochs+1):
        train_loss = []
        train_mape = []
        train_rmse = []
        t1 = time.time()
        # dataloader['train_loader'].shuffle()
        train_loader.sampler.set_epoch(i)

        for iter, data in enumerate(train_loader):
            data = [item.to(device, non_blocking=True) for item in data]
            trainx, trainy = data
            trainx = trainx.transpose(1, 3)
            trainy = trainy.transpose(1, 3)
            if iter % args.step_size2 == 0:
                perm = np.random.permutation(range(args.num_nodes))
            num_sub = int(args.num_nodes/args.num_split)
            for j in range(args.num_split):
                if j != args.num_split-1:
                    id = perm[j * num_sub:(j + 1) * num_sub]
                else:
                    id = perm[j * num_sub:]
                id = torch.tensor(id, dtype=torch.long).to(device)
                tx = trainx[:, :, id, :]
                ty = trainy[:, :, id, :]
                metrics = engine.train(tx, ty[:,0,:,:],id)
                train_loss.append(metrics[0])
                train_mape.append(metrics[1])
                train_rmse.append(metrics[2])
            if iter % args.print_every == 0 :
                log = 'Iter: {:03d}, Train Loss: {:.4f}, Train MAPE: {:.4f}, Train RMSE: {:.4f}'
                print(log.format(iter, train_loss[-1], train_mape[-1], train_rmse[-1]),flush=True)
        t2 = time.time()
        train_time.append(t2-t1)
        #validation
        valid_loss = []
        valid_mape = []
        valid_rmse = []

        s1 = time.time()
        for iter, data_val in enumerate(val_loader):
            data_val = [item.to(device, non_blocking=True) for item in data_val]
            x_val, y_val = data_val
            testx = x_val.transpose(1, 3)
            testy = y_val.transpose(1, 3)
            metrics = engine.eval(testx, testy[:,0,:,:])
            valid_loss.append(metrics[0])
            valid_mape.append(metrics[1])
            valid_rmse.append(metrics[2])
        s2 = time.time()
        log = 'Epoch: {:03d}, Inference Time: {:.4f} secs'
        print(log.format(i,(s2-s1)))
        val_time.append(s2-s1)
        mtrain_loss = np.mean(train_loss)
        mtrain_mape = np.mean(train_mape)
        mtrain_rmse = np.mean(train_rmse)

        mvalid_loss = np.mean(valid_loss)
        mvalid_mape = np.mean(valid_mape)
        mvalid_rmse = np.mean(valid_rmse)
        his_loss.append(mvalid_loss)

        log = 'Epoch: {:03d}, Train Loss: {:.4f}, Train MAPE: {:.4f}, Train RMSE: {:.4f}, Valid Loss: {:.4f}, Valid MAPE: {:.4f}, Valid RMSE: {:.4f}, Training Time: {:.4f}/epoch'
        print(log.format(i, mtrain_loss, mtrain_mape, mtrain_rmse, mvalid_loss, mvalid_mape, mvalid_rmse, (t2 - t1)),flush=True)

        if mvalid_loss<minl:
            minl = mvalid_loss
            if local_rank == 0:
                torch.save(engine.model.state_dict(), args.save + "exp" + str(args.expid) + "_" + str(runid) +".pth")


    print("Average Training Time: {:.4f} secs/epoch".format(np.mean(train_time)))
    print("Average Inference Time: {:.4f} secs".format(np.mean(val_time)))


    bestid = np.argmin(his_loss)

    map_location = {'cuda:%d' % 0: 'cuda:%d' % local_rank}
    engine.model.load_state_dict(torch.load(args.save + "exp" + str(args.expid) + "_" + str(runid) +".pth",
                                            map_location=map_location))

    print("Training finished")
    print("The valid loss on best model is", str(round(his_loss[bestid],4)))

    #valid data
    outputs = []
    realys = []
    # realy = torch.Tensor(dataloader['y_val']).to(device)
    # realy = realy.transpose(1,3)[:,0,:,:]

    for iter, data_val in enumerate(val_loader):
        data_val = [item.to(device, non_blocking=True) for item in data_val]
        x_val, y_val = data_val
        # testx = torch.Tensor(x).to(device)
        testx = testx.transpose(1,3)
        realy = y_val.transpose(1,3)[:,0,:,:]
        with torch.no_grad():
            preds = engine.model(testx)
            preds = preds.transpose(1,3)
        outputs.append(preds.squeeze())
        realys.append(realy)

    realys = torch.cat(realys, dim=0)
    yhat = torch.cat(outputs,dim=0)
    yhat = yhat[:realys.size(0),...]


    pred = scaler.inverse_transform(yhat)
    vmae, vmape, vrmse = metric(pred,realys)

    #test data
    outputs = []
    realys = []
    # realy = torch.Tensor(dataloader['y_test']).to(device)
    # realy = realy.transpose(1, 3)[:, 0, :, :]

    for iter, data in enumerate(test_loader):
        data = [item.to(device, non_blocking=True) for item in data]
        testx, testy = data
        testx = testx.transpose(1, 3)
        realy = testy.transpose(1, 3)[:, 0, :, :]
        with torch.no_grad():
            preds = engine.model(testx)
            preds = preds.transpose(1, 3)
        outputs.append(preds.squeeze())
        realys.append(realy)

    realys = torch.cat(realys, dim=0)
    yhat = torch.cat(outputs, dim=0)
    yhat = yhat[:realys.size(0), ...]

    mae = []
    mape = []
    rmse = []
    for i in range(args.seq_out_len):
        pred = scaler.inverse_transform(yhat[:, :, i])
        real = realys[:, :, i]
        metrics = metric(pred, real)
        log = 'Evaluate best model on test data for horizon {:d}, Test MAE: {:.4f}, Test MAPE: {:.4f}, Test RMSE: {:.4f}'
        print(log.format(i + 1, metrics[0], metrics[1], metrics[2]))
        mae.append(metrics[0])
        mape.append(metrics[1])
        rmse.append(metrics[2])
    return vmae, vmape, vrmse, mae, mape, rmse

if __name__ == "__main__":

    vmae = []
    vmape = []
    vrmse = []
    mae = []
    mape = []
    rmse = []
    for i in range(args.runs):
        vm1, vm2, vm3, m1, m2, m3 = main(i)
        vmae.append(vm1)
        vmape.append(vm2)
        vrmse.append(vm3)
        mae.append(m1)
        mape.append(m2)
        rmse.append(m3)

    mae = np.array(mae)
    mape = np.array(mape)
    rmse = np.array(rmse)

    amae = np.mean(mae,0)
    amape = np.mean(mape,0)
    armse = np.mean(rmse,0)

    smae = np.std(mae,0)
    smape = np.std(mape,0)
    srmse = np.std(rmse,0)

    print('\n\nResults for 10 runs\n\n')
    #valid data
    print('valid\tMAE\tRMSE\tMAPE')
    log = 'mean:\t{:.4f}\t{:.4f}\t{:.4f}'
    print(log.format(np.mean(vmae),np.mean(vrmse),np.mean(vmape)))
    log = 'std:\t{:.4f}\t{:.4f}\t{:.4f}'
    print(log.format(np.std(vmae),np.std(vrmse),np.std(vmape)))
    print('\n\n')
    #test data
    print('test|horizon\tMAE-mean\tRMSE-mean\tMAPE-mean\tMAE-std\tRMSE-std\tMAPE-std')
    for i in [2,5,11]:
        log = '{:d}\t{:.4f}\t{:.4f}\t{:.4f}\t{:.4f}\t{:.4f}\t{:.4f}'
        print(log.format(i+1, amae[i], armse[i], amape[i], smae[i], srmse[i], smape[i]))





