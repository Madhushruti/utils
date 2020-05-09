import torch.nn.functional as F
from tqdm import tqdm
from helper import HelperModel


class Train(object):
    def __init__(self):
        self.train_losses = []
        self.train_acc = []

    def train(self, model, device, train_loader, optimizer, l1_factor=None):
        model.train()
        pbar = tqdm(train_loader)
        correct = 0
        processed = 0
        for batch_idx, (data, target) in enumerate(pbar):
            # get samples
            data, target = data.to(device), target.to(device)

            # Init
            optimizer.zero_grad()
            # In PyTorch, we need to set the gradients to zero before starting to do backpropragation because PyTorch accumulates the gradients on subsequent backward passes.
            # Because of this, when you start your training loop, ideally you should zero out the gradients so that you do the parameter update correctly.

            # Predict
            y_pred = model(data)
            # pdb.set_trace()
            # Calculate loss
            loss = F.nll_loss(y_pred, target)
            # update l1 regularizer if requested
            if l1_factor:
                loss = HelperModel.apply_l1_regularizer(model, loss, l1_factor)

            self.train_losses.append(loss.item())

            # Backpropagation
            loss.backward()
            optimizer.step()

            # Update pbar-tqdm

            pred = y_pred.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()
            processed += len(data)

            pbar.set_description(desc=f'Train Set: Train Loss={loss.item()} Batch_id={batch_idx} Accuracy={100*correct/processed:0.2f}')
            acc = float("{:.2f}".format(100 * correct / processed))
            # self.train_acc.append(100*correct/processed)
            self.train_acc.append(acc)