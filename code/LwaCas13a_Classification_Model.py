
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score, precision_score, \
    recall_score, confusion_matrix
from sklearn.preprocessing import StandardScaler
import pickle
import torch.utils.data
from torch.utils.data import DataLoader, Dataset
from torch import nn, optim
import torch
from torch.optim import lr_scheduler
import random
import argparse
from Model import MultiModalCNN
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
set_seed(42)

# Function to load and process pickled data
def load_pickle_data(file_path):
    with open(file_path, 'rb') as f:
        data_dict = pickle.load(f)
    x_data = data_dict['embeddings']  # Slice the data
    y_data = data_dict['labels']
    return x_data, y_data

# Function to process CSV data
def load_csv_data(file_path):
    data = pd.read_csv(file_path)
    y_data = data.iloc[:, 0]
    x_data = data.iloc[:, 1:]
    return x_data, y_data

# Load and process train, test, and validate data
x_train, y_train = load_pickle_data(
    '../data/LwaCas13a_Classification_Data/train_class_data_onehot.pkl')
x_test, y_test = load_pickle_data(
    '../data/LwaCas13a_Classification_Data/test_class_data_onehot.pkl')
x_validate, y_validate = load_pickle_data(
    '../data/LwaCas13a_Classification_Data/valiadte_class_data_onehot.pkl')

x_train = np.reshape(x_train, (x_train.shape[0], 1, x_train.shape[1], x_train.shape[2]))
x_test = np.reshape(x_test, (x_test.shape[0], 1, x_test.shape[1], x_test.shape[2]))
x_validate = np.reshape(x_validate, (x_validate.shape[0], 1, x_validate.shape[1], x_validate.shape[2]))
print(f"Pickle Data Shapes: {x_train.shape}, {x_test.shape}, {x_validate.shape}")

x_train1, y_train1= load_csv_data(
    '../data/LwaCas13a_Classification_Data/train_class_data_handcrafted.csv')
x_test1, y_test1= load_csv_data(
    '../data/LwaCas13a_Classification_Data/test_class_data_handcrafted.csv')
x_validate1, y_validate1 = load_csv_data(
    '../data/LwaCas13a_Classification_Data/validate_class_data_handcrafted.csv')

scaler = StandardScaler()
x_train1 = scaler.fit_transform(x_train1)
x_test1 = scaler.transform(x_test1)
x_validate1 = scaler.transform(x_validate1)
print(f"Handcrafted Data Shapes: {x_train1.shape}, {x_test1.shape}, {x_validate1.shape}")

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
        label = self.labels[idx]

        return sample1D, sample2D, label

custom_dataset_train = CustomDataset(x_train1, x_train, y_train)
custom_dataset_test = CustomDataset(x_test1, x_test, y_test)
custom_dataset_validate = CustomDataset(x_validate1, x_validate, y_validate)

batch_size = 256
custom_loader_train = DataLoader(custom_dataset_train, batch_size=batch_size, shuffle=False)
custom_loader_test = DataLoader(custom_dataset_test, batch_size=batch_size, shuffle=False)
custom_loader_validate = DataLoader(custom_dataset_validate, batch_size=batch_size, shuffle=False)
batch = next(iter(custom_loader_train))
sample1D_batch, sample2D_batch, label_batch = batch

print(sample1D_batch.shape)
print(sample2D_batch.shape)
print(label_batch.shape)


def train_model(model, train_loader, val_loader, loss_function, optimizer, scheduler, num_epochs,
                early_stopping_patience, device):
    min_val_loss = float('inf')
    epochs_no_improve = 0
    best_auc = 0
    best_auc_pr = 0

    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        correct_train = 0
        total_train = 0
        true_list = []
        predicted_list = []
        output_list = []

        for batch_data1D, batch_data2D, batch_labels in train_loader:
            batch_data1D, batch_data2D, batch_labels = batch_data1D.to(device), batch_data2D.to(
                device), batch_labels.to(device)
            batch_data1D, batch_data2D = batch_data1D.float(), batch_data2D.float()
            optimizer.zero_grad()

            output1D, output_class_1D = model.CNN1D_branch(batch_data1D)
            output2D, output_class_2D = model.CNN2D_branch(batch_data2D)
            combined_output, _, _ = model(batch_data1D, batch_data2D)

            loss1 = loss_function(output_class_1D.ravel(), batch_labels.float())
            loss2 = loss_function(output_class_2D.ravel(), batch_labels.float())
            loss3 = loss_function(combined_output.ravel(), batch_labels.float())
            loss = 0.5 * loss1 + 0.5 * loss2 + 0.5 * loss3

            loss.backward()
            optimizer.step()
            total_loss += loss.item()

            probs = torch.sigmoid(combined_output.squeeze())
            predicted = (probs > 0.5).int()
            correct_train += (predicted == batch_labels.squeeze()).sum().item()
            total_train += batch_labels.size(0)

            predicted_list.extend(predicted.cpu().numpy())
            true_list.extend(batch_labels.cpu().numpy())
            output_list.extend(combined_output.detach().cpu().numpy())

        scheduler.step()

        accuracy_train = 100 * correct_train / total_train
        auc_roc = roc_auc_score(true_list, output_list)

        print(
            f"[Epoch: {epoch + 1}] [loss avg: {total_loss / len(train_loader):.4f}] [accuracy train: {accuracy_train:.2f}%]")
        print(f"AUC ROC: {auc_roc:.4f}")

        # Validation
        model.eval()
        total_loss1 = 0
        correct_test = 0
        total_test = 0
        true_list = []
        predicted_list = []
        output_list = []

        with torch.no_grad():
            for batch_data1D1, batch_data2D1, batch_labels1 in val_loader:
                batch_data1D1, batch_data2D1, batch_labels1 = batch_data1D1.to(device), batch_data2D1.to(
                    device), batch_labels1.to(device)
                batch_data1D1, batch_data2D1 = batch_data1D1.float(), batch_data2D1.float()

                output1D1, output_class_1D1 = model.CNN1D_branch(batch_data1D1)
                output2D1, output_class_2D1 = model.CNN2D_branch(batch_data2D1)
                combined_output1, _, _ = model(batch_data1D1, batch_data2D1)

                loss1 = loss_function(output_class_1D1.ravel(), batch_labels1.float())
                loss2 = loss_function(output_class_2D1.ravel(), batch_labels1.float())
                loss3 = loss_function(combined_output1.ravel(), batch_labels1.float())
                loss = 0.5 * loss1 + 0.5 * loss2 + 0.5 * loss3
                total_loss1 += loss.item()

                probs = torch.sigmoid(combined_output1.squeeze())
                predicted = (probs > 0.5).int()

                correct_test += (predicted == batch_labels1.squeeze()).sum().item()
                total_test += batch_labels1.size(0)

                predicted_list.extend(predicted.cpu().numpy())
                true_list.extend(batch_labels1.cpu().numpy())
                output_list.extend(combined_output1.detach().cpu().numpy())

            accuracy_validate = 100 * correct_test / total_test
            auc_roc = roc_auc_score(true_list, output_list)
            auc_pr = average_precision_score(true_list, output_list)

            print(
                f"[Epoch: {epoch + 1}] [loss avg: {total_loss1 / len(val_loader):.4f}] [accuracy validate: {accuracy_validate:.2f}%]")
            print(f"AUC ROC: {auc_roc:.4f}",f"AUC PR: {auc_pr:.4f}")

            if auc_roc > best_auc and auc_pr > best_auc_pr:
                best_auc = auc_roc
                best_auc_pr = auc_pr
                epochs_no_improve = 0
                # torch.save(model, 'best_model.pth')
                print("Save best model")
            else:
                epochs_no_improve += 1

            if epochs_no_improve >= early_stopping_patience:
                print("Early stopping")
                break


def test_model(model, test_loader, device):
    model.eval()
    true_list = []
    predicted_list = []
    output_list = []
    correct_test = 0
    total_test = 0

    with torch.no_grad():
        for batch_data1D, batch_data2D, batch_labels in test_loader:
            batch_data1D, batch_data2D, batch_labels = batch_data1D.to(device), batch_data2D.to(device), batch_labels.to(device)
            batch_data1D, batch_data2D = batch_data1D.float(), batch_data2D.float()

            combined_output, _, _ = model(batch_data1D, batch_data2D)

            probs = torch.sigmoid(combined_output.squeeze())
            predicted = (probs > 0.5).int()

            correct_test += (predicted == batch_labels.squeeze()).sum().item()
            total_test += batch_labels.size(0)

            predicted_list.extend(predicted.cpu().numpy())
            true_list.extend(batch_labels.cpu().numpy())
            output_list.extend(combined_output.detach().cpu().numpy())

    accuracy_test = 100 * correct_test / total_test
    predicted_array = np.array(predicted_list)
    true_array = np.array(true_list)
    output_array = np.array(output_list)
    auc_roc = roc_auc_score(true_array, output_array)
    auc_pr = average_precision_score(true_array, output_array)
    precision = precision_score(true_array, predicted_array)
    recall = recall_score(true_array, predicted_array)
    conf_matrix = confusion_matrix(true_array, predicted_array)

    print('[accuracy test: %.2f%%]' % accuracy_test)
    print('AUC ROC:', auc_roc)
    print('AUC PR:', auc_pr)
    print('Precision:', precision)
    print('Recall:', recall)
    print('Confusion Matrix:\n', conf_matrix)




def parse_args():
    parser = argparse.ArgumentParser(description="Training parameters")
    parser.add_argument('--hidden1', type=int, default=2048, help='Hidden layer size 1')
    parser.add_argument('--hidden2', type=int, default=256, help='Hidden layer size 2')
    parser.add_argument('--hidden3', type=int, default=512, help='Hidden layer size 3')
    parser.add_argument('--input_shape', type=tuple, default=(x_train.shape[2], x_train.shape[3]), help='Input shape')
    parser.add_argument('--input_channels', type=int, default=1, help='Input channels')
    parser.add_argument('--output_channels1', type=int, default=256, help='Output channels for CNN1D')
    parser.add_argument('--output_channels2', type=int, default=64, help='Output channels for CNN2D')
    parser.add_argument('--conv_kernel_sizes1', type=tuple, default=(1, 8), help='Conv kernel sizes for CNN1D')
    parser.add_argument('--conv_kernel_sizes2', type=tuple, default=(2, 1), help='Conv kernel sizes for CNN2D')
    parser.add_argument('--pool_kernel_sizes1', type=tuple, default=(1, 1), help='Pool kernel sizes for CNN1D')
    parser.add_argument('--pool_kernel_sizes2', type=tuple, default=(2, 1), help='Pool kernel sizes for CNN2D')
    parser.add_argument('--lr', type=float, default=0.000005, help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=0.00001, help='Weight decay')
    parser.add_argument('--num_epochs', type=int, default=150, help='Number of epochs')
    parser.add_argument('--early_stopping_patience', type=int, default=10, help='Early stopping patience')

    return parser.parse_args()





def main():
    args = parse_args()

    # Set device
    device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")

    # Initialize the model
    model = MultiModalCNN(args.hidden1, args.hidden2, args.hidden3, args.input_shape, args.input_channels,
                          args.output_channels1, args.output_channels2, args.conv_kernel_sizes1,
                          args.conv_kernel_sizes2, args.pool_kernel_sizes1, args.pool_kernel_sizes2).to(device)

    # Loss function, optimizer, and scheduler
    loss_function = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([0.14], device=device))
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.4)

    # Train the model
    train_model(model, custom_loader_train, custom_loader_validate, loss_function, optimizer, scheduler,
                args.num_epochs, args.early_stopping_patience, device)

    test_model(model, custom_loader_test, device)
if __name__ == '__main__':
    main()








