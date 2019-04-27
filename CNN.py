# python CNN.py --data data/VC/final.json --output final_embeddings

import os
import json
import argparse
import torch
import torch.nn as nn
import torch.utils.data as Data
import torchvision
from torch.utils.data.dataset import Dataset  # For custom datasets
from torchvision import transforms
from torch.autograd import Variable
import matplotlib.pyplot as plt
from PIL import Image
import pickle
import tsv
import numpy as np

class CustomDatasetFromImages(Dataset):
    def __init__(self, file_path):
        self.to_tensor = transforms.ToTensor()
        self.file_path = file_path
        self.map = {}
        self.labels = []
        image_list = []

        # Read json file_path + "data.json"
        with open(file_path) as file:
            self.data = json.load(file)
        for i, idx in enumerate(self.data):
            self.map[i] = idx
            item = self.data[idx]
            label = int(item[0].split(" ")[0])
            self.labels.append(label)

            img_as_img = Image.open("data/VC/img_all/" + str(idx) +".jpg")
            img_as_img = img_as_img.convert('1')
            img_as_tensor = self.to_tensor(img_as_img)
            image_list.append(img_as_tensor)

        self.labels = torch.tensor(self.labels)
        self.labels.cuda()
        self.data_len = len(self.data)
        self.image_tensors = torch.Tensor(self.data_len, 32, 32)
        self.image_tensors.cuda()
        torch.cat(image_list, out=self.image_tensors)


    def __getitem__(self, id):
        # Get image name from the pandas df
        index = self.map[id]

        img_as_img = Image.open("data/VC/img_all/" + str(index) +".jpg")
        # one channel: b/w
        img_as_img = img_as_img.convert('1')
        img_as_tensor = self.to_tensor(img_as_img)
        item = self.data[index]

        # hack split TODO: multiple labels
        label_idx = item[0].split(" ")[0]
        label = item[2].split(" ")[0]
        # char_idx, char, img_as_tensor, label_idx, label
        return (index, item[1], img_as_tensor, int(label_idx), label)

    def __len__(self):
        return self.data_len

class CNN(nn.Module):
    def __init__(self, embedding_size):
        super(CNN, self).__init__()
        self.embedding_size = embedding_size

        self.conv1 = nn.Sequential(         # input shape (1, 32, 32)
            nn.Conv2d(
                in_channels=1,              # input height
                out_channels=32,            # n_filters
                kernel_size=3,              # filter size
                stride=1,                   # filter movement/step
                padding=1,                  # if want same width and length of this image after Conv2d, padding=(kernel_size-1)/2 if stride=1
            ),                              # output shape (16, 32, 32)
            nn.ReLU(),                      # activation
            nn.MaxPool2d(kernel_size=2),    # choose max value in 2x2 area, output shape (16, 16, 16)
        )
        self.conv2 = nn.Sequential(         # input shape (1, 32, 32)
            nn.Conv2d(32, 32, 3,stride=1, padding=1),     # output shape (32, 14, 14)
            nn.ReLU(),                      # activation
            nn.MaxPool2d(kernel_size=2),                # output shape (32, 7, 7)
        )
        self.conv3 = nn.Sequential(         # input shape (1, 32, 32)
            nn.Conv2d(32, 32, 3, stride=1, padding=1),     # output shape (32, 16, 16)
            nn.ReLU(),                      # activation
        )
        self.out = nn.Sequential(
             nn.Linear(32* 8 * 8, 32*4*4),   # fully connected layer, output 10 classes
             nn.ReLU(),
             nn.Linear(32*4*4, 32*4*4) #output layers >= classes
        )

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = x.view(-1,32*8*8)
        output = self.out(x)
        # print("Output", output.size())
        #
        # print("Embedding", x.size())
        # print("Embedding:", embedding.size())

        return output, x    # return x for visualization

def main():
    cnn = CNN(300) #embedding_size
    print(cnn)  # net architecture
    cnn.cuda()

    # torch.manual_seed(1)    # reproducible

    # Hyper Parameters
    EPOCH = 5              # train the training data n times, to save time, we just train 1 epoch
    BATCH_SIZE = 64
    LR = 1e-3           # learning rate
    # CLASSES =

    train_data = CustomDatasetFromImages(args.data)
    all_data = CustomDatasetFromImages(args.data)
    print("All:", all_data.data_len)
    # Cat: 123
    # Total:7351

    # plot one example
    # print(train_data.size())                 # (60000, 28, 28)
    # plt.imshow(train_data.train_data[0].numpy(), cmap='gray')
    # plt.title('%i' % train_data.train_labels[0])
    # plt.show()

    # Data Loader for easy mini-batch return in training, the image batch shape will be (50, 1, 28, 28)
    train_loader = Data.DataLoader(dataset=train_data, batch_size=BATCH_SIZE, shuffle=True)
    # pick 2000 samples to speed up testing

    test_data = CustomDatasetFromImages("data/VC/test.json")

    all_x = Variable(torch.unsqueeze(all_data.image_tensors, dim=1)).type(torch.FloatTensor)
    test_x = Variable(torch.unsqueeze(test_data.image_tensors, dim=1)).type(torch.FloatTensor)
    test_y = Variable(torch.unsqueeze(test_data.labels, dim=1)).type(torch.FloatTensor)

    # test_x = torch.unsqueeze(test_data.test_data, dim=1).type(torch.FloatTensor)[:2000]/255.   # shape from (2000, 28, 28) to (2000, 1, 28, 28), value in range(0,1)
    # test_y = test_data.test_labels[:2000]

    optimizer = torch.optim.Adam(cnn.parameters(), lr=LR)   # optimize all cnn parameters
    loss_func = nn.CrossEntropyLoss()                       # the target label is not one-hotted

    # following function (plot_with_labels) is for visualization, can be ignored if not interested
    from matplotlib import cm
    # try: from sklearn.manifold import TSNE; HAS_SK = True
    # except: HAS_SK = False; print('Please install sklearn for layer visualization')

    def plot_with_labels(lowDWeights, labels):
        plt.cla()
        X, Y = lowDWeights[:, 0], lowDWeights[:, 1]
        for x, y, s in zip(X, Y, labels):
            c = cm.rainbow(int(255 * s / 9)); plt.text(x, y, s, backgroundcolor=c, fontsize=9)
        plt.xlim(X.min(), X.max()); plt.ylim(Y.min(), Y.max()); plt.title('Visualize last layer'); plt.show(); plt.pause(0.01)

    plt.ion()
    # training and testing
    for epoch in range(EPOCH):
        for step, (char_index, char, img_as_tensor, label_index, label) in enumerate(train_loader):   # gives batch data, normalize x when iterate train_loader
            input = img_as_tensor.cuda()
            label_index = label_index.cuda()

            # print(input.is_cuda)
            output, embeddings = cnn(input)  # cnn output
            # print(output.is_cuda, label_index.is_cuda)
            loss = loss_func(output, label_index)   # cross entropy loss
            optimizer.zero_grad()           # clear gradients for this training step
            loss.backward()                 # backpropagation, compute gradients
            optimizer.step()                # apply gradients

            if step % 50 == 0:
                test_x = test_x.cuda()
                test_output, last_layer = cnn(test_x)

                pred_y = torch.max(test_output.cpu(), 1)[1].data.numpy()

                accuracy = float((pred_y == test_y.data.numpy()).astype(int).sum()) / float(test_y.size(0))

                print('Epoch: ', epoch, step, '| train loss: %.4f' % loss.data.cpu().numpy(), '| test accuracy: %.2f' % accuracy)
                # if HAS_SK:
                #     # Visualization of trained flatten layer (T-SNE)
                #     tsne = TSNE(perplexity=30, n_components=2, init='pca', n_iter=5000)
                #     plot_only = 500
                #     low_dim_embs = tsne.fit_transform(last_layer.data.numpy()[:plot_only, :])
                #     labels = test_y.numpy()[:plot_only]
                #     plot_with_labels(low_dim_embs, labels)
        all_x = all_x.cuda()
        test_output, embeddings = cnn(all_x)

        with open(str(epoch) + "_" + args.output +".pkl", 'wb') as output:  # Overwrites any existing file.
            embeddings = embeddings.cpu()
            print(embeddings.is_cuda)
            pickle.dump(embeddings, output, pickle.HIGHEST_PROTOCOL)
            print("Embedding saved")

    plt.ioff()

    writer = tsv.TsvWriter(open(args.output + ".tsv", "w"))
    output = embeddings.cpu()
    for idx, line in enumerate(output):
        if idx%100 == 0: print(idx)
        s = ""
        for n in line:
            n = n.detach().numpy()
            s += str(n) + "\t"
        writer.line(s)
    writer.close()
    print("Embedding saved")


    # print 10 predictions from test data
    # test_output, _ = cnn(test_x[:10])
    # pred_y = torch.max(test_output, 1)[1].data.numpy()
    # print(pred_y, 'prediction number')
    # print(test_y[:10].numpy(), 'real number')

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    # args.data, args.output
    parser.add_argument("--data", type=str, help="path to data file", default="data")
    parser.add_argument("--output", type=str, help="output file name")
    args = parser.parse_args()

    main()