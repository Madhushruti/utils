import torch
import torch.nn.functional as F
from tqdm import tqdm
from helper import HelperModel


class Test(object):

    def __init__(self):
        self.test_losses = []
        self.test_acc = []
        self.misclassified_images = []

    def update_misclassified_images(self, data, target, pred):
        target_change = target.view_as(pred)
        for i in range(len(pred)):
          if pred[i].item()!= target_change[i].item():
            self.misclassified_images.append([data[i], pred[i], target_change[i]])

    def test(self, model, device, test_loader, misclassfied_required=False):
        model.eval()
        test_loss = 0
        correct = 0
        with torch.no_grad():
            for data, target in test_loader:
                data, target = data.to(device), target.to(device)
                output = model(data)
                test_loss += F.nll_loss(output, target, reduction='sum').item()  # sum up batch loss
                pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
                # update misclassified images if requested
                if misclassfied_required:
                   self.update_misclassified_images(data, target, pred)
                correct += pred.eq(target.view_as(pred)).sum().item()

        test_loss /= len(test_loader.dataset)
        self.test_losses.append(test_loss)

        print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)\n'.format(
            test_loss, correct, len(test_loader.dataset),
            100. * correct / len(test_loader.dataset)))

        self.test_acc.append(100. * correct / len(test_loader.dataset))
