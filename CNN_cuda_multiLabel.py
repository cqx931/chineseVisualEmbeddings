import os
import json
import argparse
# torch
import torch
import torch.nn as nn
import torch.utils.data as Data
import torchvision
from torch.utils.data.dataset import Dataset  # For custom datasets
from torchvision import transforms
from torch.autograd import Variable

# other third party libraries
from PIL import Image
import pickle
import tsv
import numpy as np

is_cuda = False

class CustomDatasetFromImages(Dataset):
    def __init__(self, file_path):
        self.to_tensor = transforms.ToTensor()
        self.file_path = file_path
        self.map = {}
        self.labels = []
        image_list = []
        labels_list = []

        # Read json file_path + "data.json"
        with open(file_path) as file:
            self.data = json.load(file)
        for i, idx in enumerate(self.data):
            self.map[i] = idx
            item = self.data[idx]
            labels = item[0].split(" ")
            labels_int = []
            for label in labels:
                labels_int.append(int(label))
            labels_list.append(torch.FloatTensor(labels_int))

            img_as_img = Image.open("data/VC/img_all/" + str(idx) +".jpg")
            img_as_img = img_as_img.convert('1')
            img_as_tensor = self.to_tensor(img_as_img)
            image_list.append(img_as_tensor)
        # print(self.labels)

        self.data_len = len(self.data)
        self.image_tensors = torch.Tensor(self.data_len, 32, 32)
        torch.cat(image_list, out=self.image_tensors)

        self.labels = torch.Tensor(self.data_len, 2) # 2 labels
        if is_cuda:
            self.image_tensors.cuda()
            self.labels.cuda()
        # torch.cat(labels_list, out=self.labels)
        # print(self.labels.size())

    def __getitem__(self, id):
        # Get image name from the pandas df
        index = self.map[id]

        img_as_img = Image.open("data/VC/img_all/" + str(index) +".jpg")
        # One channel: b/w
        img_as_img = img_as_img.convert('1')
        img_as_tensor = self.to_tensor(img_as_img)
        item = self.data[index]

        label_idx_str = item[0].split(" ") # get all label idx as string
        # convert to binary
        label_list = [0] * 256
        for label in label_idx_str:
            label_list[int(label)] = 1
        npa = np.array(label_list, dtype=np.float32) # convert to np array
        label_idx = torch.from_numpy(npa) # convert to tensor

        label = item[2].split(" ")

        # char_idx, char, img_as_tensor, label_idx, label
        return (index, item[1], img_as_tensor, label_idx, label)

    def __len__(self):
        return self.data_len

# Hyper Parameters
EPOCH = 1
BATCH_SIZE = 64
LR = 1e-3 # learning rate

class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
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
             nn.Linear(32 * 8 * 8, 32*4*4),
             nn.ReLU(),
             nn.Linear(32*4*4,256) #output layers >= classes
        )

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)

        x = x.view(-1,32*8*8)      # flatten the output of conv2 to (batch_size, 32 * 8 * 8)
        output = self.out(x)
        return output, x    # return x for visualization

def main(args):
    ##### STEP 1: CNN architecture #####
    cnn = CNN()
    print(cnn)
    if is_cuda:
        cnn.cuda()

    optimizer = torch.optim.Adam(cnn.parameters(), lr=LR)   # optimize all cnn parameters
    loss_func = nn.BCEWithLogitsLoss()
    # OR Hamming Loss
    # Alternatives: Focal Lost for imbalanced data: https://gombru.github.io/2018/05/23/cross_entropy_loss/
    # Also see: https://discuss.pytorch.org/t/how-to-implement-focal-loss-in-pytorch/6469/17

    ##### STEP 2: Data set #####
    all_data = CustomDatasetFromImages(args.data)
    train_data = all_data

    # Data Loader for easy mini-batch return in training, the image batch shape will be (50, 1, 28, 28)
    train_loader = Data.DataLoader(dataset=train_data, batch_size=BATCH_SIZE, shuffle=True)
    test_data = CustomDatasetFromImages(args.testData)

    all_x = Variable(torch.unsqueeze(all_data.image_tensors, dim=1)).type(torch.FloatTensor)
    test_x = Variable(torch.unsqueeze(test_data.image_tensors, dim=1)).type(torch.FloatTensor)
    test_y = Variable(torch.unsqueeze(test_data.labels, dim=1)).type(torch.FloatTensor)

    ##### STEP 3: Training and testing #####
    for epoch in range(EPOCH):
        for step, (char_index, char, img_as_tensor, label_index_tensor, label) in enumerate(train_loader):   # gives batch data, normalize x when iterate train_loader
            if is_cuda:
              label_index_tensor = label_index_tensor.cuda()
              img_as_tensor= img_as_tensor.cuda()
            input = img_as_tensor
            output, embeddings = cnn(input)  # cnn output
            loss = loss_func(output, label_index_tensor)   # loss function
            optimizer.zero_grad()           # clear gradients for this training step
            loss.backward()                 # backpropagation, compute gradients
            optimizer.step()                # apply gradients

            if step % 50 == 0:
                if is_cuda:
                    test_x = test_x.cuda()
                test_output, last_layer = cnn(test_x)

                pred_y = torch.max(test_output.cpu(), 1)[1].data.numpy()
                # TODO: how to calculate accuracy for multi label classification?
                print(pred_y)
                # accuracy = float(correct) / float(test_y.size(0))

                print('Epoch: ', epoch, step, '| train loss: %.4f' % loss.data.cpu().numpy())
                # print('Epoch: ', epoch, step, '| train loss: %.4f' % loss.data.cpu().numpy(), '| test accuracy: %.2f' % accuracy)

        if is_cuda:
            all_x = all_x.cuda()
        test_output, embeddings = cnn(all_x)

        # Comment out if we need to save embeddings during the training process
        # with open(str(epoch) + "_" + args.output +".pkl", 'wb') as output:  # Overwrites any existing file.
        #     if is_cuda:
        #         embeddings = embeddings.cpu()
        #     pickle.dump(embeddings, output, pickle.HIGHEST_PROTOCOL)
        #     print("Embedding saved")

    writer = tsv.TsvWriter(open(args.output, "w"))
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
    test_output, _ = cnn(test_x[:10])
    pred_y = torch.max(test_output, 1)[1].data.numpy()
    print(pred_y, 'prediction number')
    print(test_y[:10].numpy(), 'real number')

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, help="path to data file", default="data/VC/v3.2/v3.2_256C.json")
    parser.add_argument("--testData", type=str, help="path to test data file", default="data/VC/v3.2/v3.2_256C.json")
    parser.add_argument("--output", type=str, help="output file name", default="embeddings.tsv")
    args = parser.parse_args()

    main(args)
