import matplotlib.pyplot as plt
import numpy as np

class Plot(object):

    def __init__(self, train_acc, train_losses, test_acc, test_losses):
        self.train_acc = train_acc
        self.train_losses = train_losses
        self.test_acc = test_acc
        self.test_losses = test_losses

    def display_all_plot(self):
        """Plots graph for train, validation accuracy and losses"""
        try:
            fig, axs = plt.subplots(2, 2, figsize=(15, 10))
            axs[0, 0].plot(self.train_losses)
            axs[0, 0].set_title("Training Loss")
            axs[1, 0].plot(self.train_acc)
            axs[1, 0].set_title("Training Accuracy")
            axs[0, 1].plot(self.test_losses)
            axs[0, 1].set_title("Validation Loss")
            axs[1, 1].plot(self.test_acc)
            axs[1, 1].set_title("Validation Accuracy")
        except Exception as err:
            raise err

    def plot_train_graph(self, plot_case="Accuracy"):
        try:
            fig = plt.figure(figsize=(9, 9))
            if plot_case == "Accuracy":
                train_data = self.train_acc
            else:
                train_data = self.train_losses
            plt.title("Training {0}".format(plot_case))
            plt.xlabel("Epochs")
            plt.ylabel(plot_case)
            plt.plot(train_data)
            plt.show()
            fig.savefig('train_{0}_graph.png'.format(plot_case.lower()))
        except Exception as err:
            raise err

    def plot_validation_graph(self, plot_case="Accuracy"):
        """Plots single graph for validation losses/accuracy"""
        try:
            fig = plt.figure(figsize=(9, 9))
            if plot_case == "Accuracy":
                test_data = self.test_acc
            else:
                test_data = self.test_losses
            plt.title("Validation {0}".format(plot_case))
            plt.xlabel("Epochs")
            plt.ylabel(plot_case)
            plt.plot(test_data)
            plt.show()
            fig.savefig('validation_{0}_graph.png'.format(plot_case.lower()))
        except Exception as err:
            raise err

    @staticmethod
    def plot_misclassified(misclassified_images, image_count=25):
      fig = plt.figure(figsize=(15, 15))
      for i in range(image_count):
        sub = fig.add_subplot(5, 5, i+1)
        plt.imshow(misclassified_images[i][0].cpu().numpy().squeeze(),cmap='gray',interpolation='none')
        sub.set_title("Predicted={0}, Actual={1}".format(str(misclassified_images[i][1].data.cpu().numpy()),str(misclassified_images[i][2].data.cpu().numpy())))
      plt.tight_layout()
      plt.show()
      fig.savefig("misclassified_images.png")

    @staticmethod
    def image_show(img):
        img = img / 2 + 0.5  # unnormalize
        npimg = img.numpy()
        plt.imshow(np.transpose(npimg, (1, 2, 0)))