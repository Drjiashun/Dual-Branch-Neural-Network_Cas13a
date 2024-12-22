
from sklearn.metrics import mean_squared_error
import numpy as np
import pandas as pd
from scipy.stats import pearsonr
from sklearn.preprocessing import StandardScaler
import pickle
from scipy.stats import spearmanr
import torch.utils.data
from torch.utils.data import DataLoader, Dataset
from torch import nn, optim
import torch
from torch.optim import lr_scheduler
import random
from sklearn.model_selection import train_test_split
from Model import MultiModalCNN
import argparse
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
set_seed(42)


data=pd.read_csv('../data/LbuCas13a_Regression_Data/binding-foldchange-specificities-for-adapt_handcrafted.csv')
y_train=data.iloc[:,0]
x_train=data.iloc[:,1:]

with open('../data/LbuCas13a_Regression_Data/binding-foldchange-specificities-for-adapt_onehot.pkl', 'rb') as f:
    data_dict = pickle.load(f)

x_train1= data_dict['embeddings']
y_train1 = data_dict['labels']

x_train1, x_test1, x_train, x_test, y_train, y_test = train_test_split(
    x_train, x_train1, y_train, test_size=0.2, random_state=42
)

x_train1, x_validate1, x_train, x_validate, y_train, y_validate = train_test_split(
    x_train1, x_train, y_train, test_size=0.25, random_state=42
)

scaler = StandardScaler()
x_train1 = scaler.fit_transform(x_train1)
x_test1 = scaler.transform(x_test1)
x_validate1 = scaler.transform(x_validate1)
x_train=np.reshape(x_train,(x_train.shape[0],1,x_train.shape[1],x_train.shape[2]))
x_test=np.reshape(x_test,(x_test.shape[0],1,x_test.shape[1],x_test.shape[2]))
x_validate=np.reshape(x_validate,(x_validate.shape[0],1,x_validate.shape[1],x_validate.shape[2]))

class CustomDataset(Dataset):
    def __init__(self, data1D, data2D, labels):
        self.data1D = data1D
        self.data2D = data2D
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        sample1D = self.data1D[idx]
        sample2D = self.data2D[idx]
        label = self.labels.iloc[idx]

        return sample1D, sample2D, label

custom_dataset_train = CustomDataset(x_train1, x_train, y_train)
custom_dataset_test = CustomDataset(x_test1, x_test, y_test)
custom_dataset_validate = CustomDataset(x_validate1, x_validate, y_validate)
batch_size = 256
custom_loader_train = DataLoader(custom_dataset_train, batch_size=batch_size, shuffle=True)
custom_loader_test = DataLoader(custom_dataset_test, batch_size=batch_size, shuffle=False)
custom_loader_validate = DataLoader(custom_dataset_validate, batch_size=batch_size, shuffle=False)

# hidden1=2048
# hidden2=256
# hidden3=512
# input_shape = (x_train.shape[2], x_train.shape[3])
# input_channels=1
# output_channels1 = 128
# output_channels2 = 128
# conv_kernel_sizes1 = (1, 8)
# conv_kernel_sizes2 = (4, 1)
# pool_kernel_sizes1 = (1, 1)
# pool_kernel_sizes2 = (2, 1)

# device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
# model = MultiModalCNN(hidden1, hidden2, hidden3, input_shape, input_channels, output_channels1, output_channels2,
#                  conv_kernel_sizes1, conv_kernel_sizes2,
#                  pool_kernel_sizes1, pool_kernel_sizes2).to(device)
#
# loss_function = nn.MSELoss()
# optimizer = optim.Adam(model.parameters(), lr=0.0001,weight_decay=0.00001)
# scheduler = lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.5)
# num_epochs = 150
# train_losses = []
# validate_losses = []
# best_mse = float('inf')
# patience = 10
# patience_counter = 0
# for epoch in range(num_epochs):
#     model.train()
#     total_loss = 0
#     total_train = 0
#     output_array = []
#     true_array = []
#
#     for (batch_data1D, batch_data2D, batch_labels) in custom_loader_train:
#
#         batch_data1D = batch_data1D.to(device)
#         batch_data2D = batch_data2D.to(device)
#         batch_labels = batch_labels.to(device)
#         batch_data1D = batch_data1D.to(torch.float32)
#         batch_data2D = batch_data2D.to(torch.float32)
#         optimizer.zero_grad()
#
#         output1D, output_class_1D = model.CNN1D_branch(batch_data1D)
#         output2D, output_class_2D = model.CNN2D_branch(batch_data2D)
#         combined_output,CNN1D_output,CNN2D_output = model(batch_data1D, batch_data2D)
#
#         alpha = nn.Parameter(torch.tensor(0.5))
#         beta = nn.Parameter(torch.tensor(0.5))
#         gamma = nn.Parameter(torch.tensor(0.5))
#         loss1 = loss_function(output_class_1D.ravel(), batch_labels.float())
#         loss2 = loss_function(output_class_2D.ravel(), batch_labels.float())
#         loss3 = loss_function(combined_output.ravel(), batch_labels.float())
#         loss = alpha * loss1 + beta * loss2 + gamma * loss3
#         loss.backward()
#         optimizer.step()
#         total_loss += loss.item()
#         output_array.extend(combined_output.squeeze().detach().cpu().numpy())
#         true_array.extend(batch_labels.cpu().numpy())
#     scheduler.step()
#     avg_loss = total_loss / len(custom_loader_train)
#     r, _ = pearsonr(output_array, true_array)
#     spearman_corr, _ = spearmanr(output_array, true_array)
#     print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {avg_loss:.4f}",f"train Pearson r: {r:.4f}",f"train Spearman: {spearman_corr:.4f}")
#     model.eval()
#     total_loss1 = 0
#     output_array = []
#     true_array = []
#
#     with torch.no_grad():
#         for (batch_data1D1, batch_data2D1, batch_labels1) in custom_loader_validate:
#             batch_data1D1 = batch_data1D1.to(device)
#             batch_data2D1 = batch_data2D1.to(device)
#             batch_labels1 = batch_labels1.to(device)
#             batch_data1D1 = batch_data1D1.to(torch.float32)
#             batch_data2D1 = batch_data2D1.to(torch.float32)
#
#             output1D1, output_class_1D1 = model.CNN1D_branch(batch_data1D1)
#             output2D1, output_class_2D1 = model.CNN2D_branch(batch_data2D1)
#             combined_output1,CNN1D_output1,CNN2D_output1 = model(batch_data1D1, batch_data2D1)
#
#             loss4 = loss_function(output_class_1D1.ravel(), batch_labels1.float())
#             loss5 = loss_function(output_class_2D1.ravel(), batch_labels1.float())
#             loss6 = loss_function(combined_output1.ravel(), batch_labels1.float())
#             loss7 = loss4 + loss5 + loss6
#             total_loss1 += loss7.item()
#             output_array.extend(combined_output1.squeeze().cpu().numpy())
#             true_array.extend(batch_labels1.cpu().numpy())
#
#     mse = mean_squared_error(true_array, output_array)
#     r, _ = pearsonr(output_array, true_array)
#     spearman_corr, _ = spearmanr(output_array, true_array)
#     print('[Epoch: %d] [loss avg: %.4f] [MSE: %.4f]' % (epoch + 1, total_loss1 / len(custom_loader_validate), mse),f"validate Pearson r: {r:.4f}",
#           f"validate Spearman: {spearman_corr:.4f}")
#     if mse < best_mse:
#         best_mse = mse
#         patience_counter = 0
#         # torch.save(model,
#         #            '/home/fujiashun/Pycharm/Pycharm project1/python_project/crRNA Activity Prediction/LbuCas13a-RNA /model/best_LbuCas13a_regression_model.pth')
#         print(f'save best model')
#
#     else:
#         patience_counter += 1
#     if patience_counter >= patience:
#         print('early_stopping')
#         break
#
#
# # model = torch.load('/home/fujiashun/Pycharm/Pycharm project1/python_project/crRNA Activity Prediction/LbuCas13a-RNA /model/best_LbuCas13a_regression_model.pth')
# # model.to(device)
# model.eval()
# output_array = []
# true_array = []
#
# for (batch_data1D1, batch_data2D1, batch_labels1) in custom_loader_test:
#     batch_data1D1 = batch_data1D1.to(device)
#     batch_data2D1 = batch_data2D1.to(device)
#     batch_labels1 = batch_labels1.to(device)
#     batch_data1D1 = batch_data1D1.to(torch.float32)
#     batch_data2D1 = batch_data2D1.to(torch.float32)
#     output1D1, output_class_1D1 = model.CNN1D_branch(batch_data1D1)
#     output2D1, output_class_2D1 = model.CNN2D_branch(batch_data2D1)
#     combined_output1,CNN1D_output1,CNN2D_output1 = model(batch_data1D1, batch_data2D1)
#
#     output_array.extend(combined_output1.squeeze().cpu().detach().numpy())
#     true_array.extend(batch_labels1.cpu().numpy())
#
# mse = mean_squared_error(true_array, output_array)
#
# print('mse',mse)
# r, _ = pearsonr(output_array, true_array)
# spearman_corr, _ = spearmanr(output_array, true_array)
# print(f"test Pearson r: {r:.4f}")
# print(f"test Spearman: {spearman_corr:.4f}")


# Define the training process as a function
def train_model(model, train_loader, val_loader, loss_function, optimizer, scheduler, num_epochs, patience, device):
    best_mse = float('inf')
    patience_counter = 0

    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        output_array = []
        true_array = []

        for batch_data1D, batch_data2D, batch_labels in train_loader:
            batch_data1D, batch_data2D, batch_labels = batch_data1D.to(device), batch_data2D.to(device), batch_labels.to(device)
            batch_data1D, batch_data2D = batch_data1D.float(), batch_data2D.float()

            optimizer.zero_grad()

            output1D, output_class_1D = model.CNN1D_branch(batch_data1D)
            output2D, output_class_2D = model.CNN2D_branch(batch_data2D)
            combined_output, _, _ = model(batch_data1D, batch_data2D)

            alpha, beta, gamma = nn.Parameter(torch.tensor(0.5)), nn.Parameter(torch.tensor(0.5)), nn.Parameter(torch.tensor(0.5))
            loss1 = loss_function(output_class_1D.ravel(), batch_labels.float())
            loss2 = loss_function(output_class_2D.ravel(), batch_labels.float())
            loss3 = loss_function(combined_output.ravel(), batch_labels.float())

            loss = alpha * loss1 + beta * loss2 + gamma * loss3
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            output_array.extend(combined_output.squeeze().detach().cpu().numpy())
            true_array.extend(batch_labels.cpu().numpy())

        scheduler.step()

        avg_loss = total_loss / len(train_loader)
        r, _ = pearsonr(output_array, true_array)
        spearman_corr, _ = spearmanr(output_array, true_array)
        print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {avg_loss:.4f}, Train Pearson r: {r:.4f}, Train Spearman: {spearman_corr:.4f}")

        # Validate the model
        model.eval()
        total_loss1 = 0
        output_array = []
        true_array = []

        with torch.no_grad():
            for batch_data1D1, batch_data2D1, batch_labels1 in val_loader:
                batch_data1D1, batch_data2D1, batch_labels1 = batch_data1D1.to(device), batch_data2D1.to(device), batch_labels1.to(device)
                batch_data1D1, batch_data2D1 = batch_data1D1.float(), batch_data2D1.float()

                output1D1, output_class_1D1 = model.CNN1D_branch(batch_data1D1)
                output2D1, output_class_2D1 = model.CNN2D_branch(batch_data2D1)
                combined_output1, _, _ = model(batch_data1D1, batch_data2D1)

                loss4 = loss_function(output_class_1D1.ravel(), batch_labels1.float())
                loss5 = loss_function(output_class_2D1.ravel(), batch_labels1.float())
                loss6 = loss_function(combined_output1.ravel(), batch_labels1.float())
                loss7 = loss4 + loss5 + loss6

                total_loss1 += loss7.item()
                output_array.extend(combined_output1.squeeze().cpu().numpy())
                true_array.extend(batch_labels1.cpu().numpy())

        mse = mean_squared_error(true_array, output_array)
        r, _ = pearsonr(output_array, true_array)
        spearman_corr, _ = spearmanr(output_array, true_array)
        print(f"Validate MSE: {mse:.4f}, Validate Pearson r: {r:.4f}, Validate Spearman: {spearman_corr:.4f}")

        # Check if model improves
        if mse < best_mse:
            best_mse = mse
            # torch.save(model, 'best_model.pth')
            print(f"Saving best model, MSE: {mse:.4f}")
            patience_counter = 0
        else:
            patience_counter += 1

        if patience_counter >= patience:
            print("Early stopping!")
            break
# Define the testing process as a function
def test_model(model, test_loader, device):
    model.eval()
    output_array = []
    true_array = []

    with torch.no_grad():
        for batch_data1D1, batch_data2D1, batch_labels1 in test_loader:
            batch_data1D1, batch_data2D1, batch_labels1 = batch_data1D1.to(device), batch_data2D1.to(device), batch_labels1.to(device)
            batch_data1D1, batch_data2D1 = batch_data1D1.float(), batch_data2D1.float()

            combined_output1, _, _ = model(batch_data1D1, batch_data2D1)

            output_array.extend(combined_output1.squeeze().cpu().detach().numpy())
            true_array.extend(batch_labels1.cpu().numpy())

    mse = mean_squared_error(true_array, output_array)
    r, _ = pearsonr(output_array, true_array)
    spearman_corr, _ = spearmanr(output_array, true_array)

    print(f"Test MSE: {mse:.4f}")
    print(f"Test Pearson r: {r:.4f}")
    print(f"Test Spearman: {spearman_corr:.4f}")

# Parse hyperparameters using argparse
def parse_args():
    parser = argparse.ArgumentParser(description="Train MultiModalCNN for crRNA Activity Prediction")

    # CNN Architecture Hyperparameters
    parser.add_argument('--hidden1', type=int, default=2048, help='First hidden layer size')
    parser.add_argument('--hidden2', type=int, default=256, help='Second hidden layer size')
    parser.add_argument('--hidden3', type=int, default=512, help='Third hidden layer size')

    parser.add_argument('--input_shape', type=tuple, default=(x_train.shape[2], x_train.shape[3]), help='Input shape (height, width)')
    parser.add_argument('--input_channels', type=int, default=1, help='Number of input channels')

    parser.add_argument('--output_channels1', type=int, default=128, help='Output channels for 1D CNN')
    parser.add_argument('--output_channels2', type=int, default=128, help='Output channels for 2D CNN')

    parser.add_argument('--conv_kernel_sizes1', type=tuple, default=(1, 8), help='Kernel size for 1D CNN')
    parser.add_argument('--conv_kernel_sizes2', type=tuple, default=(4, 1), help='Kernel size for 2D CNN')

    parser.add_argument('--pool_kernel_sizes1', type=tuple, default=(1, 1), help='Pooling kernel size for 1D CNN')
    parser.add_argument('--pool_kernel_sizes2', type=tuple, default=(2, 1), help='Pooling kernel size for 2D CNN')

    # Training Hyperparameters
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=0.00001, help='Weight decay (L2 regularization)')
    parser.add_argument('--num_epochs', type=int, default=150, help='Number of epochs to train')
    parser.add_argument('--patience', type=int, default=10, help='Patience for early stopping')

    return parser.parse_args()


# Main function to initialize and train the model
def main():
    args = parse_args()

    # Set device
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # Initialize the model
    model = MultiModalCNN(args.hidden1, args.hidden2, args.hidden3, args.input_shape, args.input_channels,
                          args.output_channels1, args.output_channels2, args.conv_kernel_sizes1,
                          args.conv_kernel_sizes2, args.pool_kernel_sizes1, args.pool_kernel_sizes2).to(device)

    # Loss function, optimizer, and scheduler
    loss_function = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.5)

    # Train the model
    train_model(model, custom_loader_train, custom_loader_validate, loss_function, optimizer, scheduler,
                args.num_epochs, args.patience, device)

    test_model(model, custom_loader_test, device)

if __name__ == '__main__':
    main()


