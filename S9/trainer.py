from tqdm import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
def train(model, optimizer, train_loader,device,epoch):
    model.train()
    pbar = tqdm(train_loader)
    losses=[]
    accs=[]
    for idx, data in enumerate(pbar):
        input = data[0].to(device)
        label = data[1].to(device)
        model.to(device)
        optimizer.zero_grad()
        output = model(input)
        loss = F.nll_loss(output, label)
        loss.backward()
        optimizer.step()
        losses.append(loss)
        pred = output.argmax(dim=1)
        acc = pred.eq(label).sum()/input.shape[0]
        accs.append(acc)
    train_avg_loss = sum(losses)/len(losses)
    train_avg_acc = sum(accs)/len(accs)
    print (f'at epoch:{epoch+1} avg_train_loss:{train_avg_loss:.3f},avg_train_acc:{train_avg_acc:.3f}')

def test(model, test_loader,device,epoch):
    model.eval()
    pbar = tqdm(test_loader)
    losses=[]
    accs=[]
    with torch.no_grad():
        for idx, data in enumerate(pbar):
            input = data[0].to(device)
            label = data[1].to(device)
            model.to(device)
            output = model(input)
            loss = F.nll_loss(output, label)
            losses.append(loss)
            pred = output.argmax(dim=1)
            acc = pred.eq(label).sum()/input.shape[0]
            accs.append(acc)
        test_avg_loss = sum(losses)/len(losses)
        test_avg_acc = sum(accs)/len(accs)
        print (f'at epoch:{epoch+1} avg_test_loss:{test_avg_loss:.3f},avg_test_acc:{test_avg_acc:.3f}')