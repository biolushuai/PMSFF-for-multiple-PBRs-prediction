import os
import torch
from losses import WeightedCrossEntropy


def train(model, train_proteins, val_proteins, configs, num=1):
    epochs = configs.epochs
    lr = configs.learning_rate
    weight_decay = configs.weight_decay
    neg_wt = configs.neg_wt

    model_save_path = configs.save_path
    print(model_save_path)
    if not os.path.exists(model_save_path):
        os.makedirs(model_save_path)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)

    if torch.cuda.is_available():
        model = model.cuda()
    optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.8, weight_decay=weight_decay, nesterov=True)
    loss_fn = WeightedCrossEntropy(neg_wt=neg_wt, device=device)

    model.train()
    train_losses = []
    val_losses = []
    best_loss = 999
    count = 0
    for e in range(epochs):
        print("Runing {} epoch".format(e + 1))
        e_loss = 0.
        for train_p in train_proteins:
            if torch.cuda.is_available():
                # train_pbs_vertex = torch.FloatTensor(train_p['protein_feature']).cuda()
                train_pbs_vertex = torch.FloatTensor(train_p['protein_feature_prottranst5xluf50']).cuda()
                train_pbs_label = torch.LongTensor(train_p['protein_label']).cuda()
            else:
                # train_pbs_vertex = torch.FloatTensor(train_p['protein_feature'])
                train_pbs_vertex = torch.FloatTensor(train_p['protein_feature_prottranst5xluf50']).cuda()
                train_pbs_label = torch.LongTensor(train_p['protein_label'])

            optimizer.zero_grad()
            batch_pred = model(train_pbs_vertex)
            batch_loss = loss_fn.computer_loss(batch_pred, train_pbs_label)

            b_loss = batch_loss.item()
            e_loss += b_loss
            batch_loss.backward()
            optimizer.step()

        e_loss /= len(train_proteins)
        train_losses.append(e_loss)
        with open(os.path.join(model_save_path, 'train_losses{}.txt'.format(num)), 'a+') as f:
            f.write(str(e_loss) + '\n')

        e_loss = 0.
        for val_p in val_proteins:
            if torch.cuda.is_available():
                # val_pbs_vertex = torch.FloatTensor(val_p['protein_feature']).cuda()
                val_pbs_vertex = torch.FloatTensor(val_p['protein_feature_prottranst5xluf50']).cuda()
                val_pbs_label = torch.LongTensor(val_p['protein_label']).cuda()
            else:
                # val_pbs_vertex = torch.FloatTensor(val_p['protein_feature'])
                val_pbs_vertex = torch.FloatTensor(val_p['protein_feature_prottranst5xluf50'])
                val_pbs_label = torch.LongTensor(val_p['protein_label'])

            optimizer.zero_grad()
            batch_pred = model(val_pbs_vertex)
            batch_loss = loss_fn.computer_loss(batch_pred, val_pbs_label)
            b_loss = batch_loss.item()
            e_loss += b_loss

        e_loss /= len(val_proteins)
        val_losses.append(e_loss)
        with open(os.path.join(model_save_path, 'val_losses{}.txt'.format(num)), 'a+') as f:
            f.write(str(e_loss) + '\n')

        if best_loss > val_losses[-1]:
            count = 0
            torch.save(model.state_dict(), os.path.join(os.path.join(model_save_path, "model{}.tar".format(num))))
            best_loss = val_losses[-1]
            print("UPDATE\tEpoch {}: train loss {}\tval loss {}".format(e + 1, train_losses[-1], val_losses[-1]))
        else:
            count += 1
            if configs.early_stop and count >= configs.early_stop:
                return None
