from tqdm import tqdm
import torch
import torch.nn.functional as F
def train(model, optimizer, train_dataloader,device,epoch,scheduler,criterion):
    model.train()
    pbar = tqdm(train_dataloader)
    losses=[]
    accs=[]
    for idx, (data,label) in enumerate(pbar):
        data=data.to(device)
        label=label.to(device)
        model=model.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, label)
        loss.backward()
        optimizer.step()
        scheduler.step()
        losses.append(loss)
        pred = output.argmax(dim=1)
        acc = pred.eq(label).sum()/data.shape[0]
        accs.append(acc)
    train_avg_loss = sum(losses)/len(losses)
    train_avg_acc = sum(accs)/len(accs)
    print (f'at epoch:{epoch}')
    print(f'avg_train_loss:{train_avg_loss},avg_train_acc:{train_avg_acc}')

def test(model, test_dataloader,device,epoch,criterion):
    model.eval()
    # pbar = tqdm(test_dataloader)
    losses=[]
    accs=[]
    with torch.no_grad():
        for idx, (data,label) in enumerate(test_dataloader):
            data=data.to(device)
            label=label.to(device)
            model=model.to(device)
            output = model(data)
            loss = criterion(output, label)
            losses.append(loss)
            pred = output.argmax(dim=1)
            acc = pred.eq(label).sum()/data.shape[0]
            accs.append(acc)
        test_avg_loss = sum(losses)/len(losses)
        test_avg_acc = sum(accs)/len(accs)
        print (f'avg_test_loss:{test_avg_loss},avg_test_acc:{test_avg_acc}')
