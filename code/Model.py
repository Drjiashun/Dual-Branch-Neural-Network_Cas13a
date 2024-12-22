import torch.utils.data
from torch import nn
import torch


class CNN2D(nn.Module):
    def __init__(self, input_shape, input_channels, output_channels1, output_channels2, conv_kernel_size1,
                 conv_kernel_size2,
                 pool_kernel_size1, pool_kernel_size2):
        super(CNN2D, self).__init__()
        self.conv1 = nn.Conv2d(input_channels, output_channels1, kernel_size=conv_kernel_size1, padding=1)
        self.bn1 = nn.BatchNorm2d(output_channels1)
        self.pool1 = nn.MaxPool2d(kernel_size=pool_kernel_size1)

        self.conv2 = nn.Conv2d(output_channels1, output_channels2, kernel_size=conv_kernel_size2, padding=1)
        self.bn2 = nn.BatchNorm2d(output_channels2)
        self.pool2 = nn.MaxPool2d(kernel_size=pool_kernel_size2)

        pool_out_size1 = (input_shape[0] - conv_kernel_size1[0] + 2 * 1 + 1) // pool_kernel_size1[0]
        pool_out_size2 = (pool_out_size1 - conv_kernel_size2[0] + 2 * 1 + 1) // pool_kernel_size2[0]

        pool_out_size3 = (input_shape[1] - conv_kernel_size1[1] + 2 * 1 + 1) // pool_kernel_size1[1]
        pool_out_size4 = (pool_out_size3 - conv_kernel_size2[1] + 2 * 1 + 1) // pool_kernel_size2[1]
        self.fc1 = nn.Linear(output_channels2 * pool_out_size2 * pool_out_size4, 256)
        self.dropout = nn.Dropout(p=0.5)
        self.fc2 = nn.Linear(256, 1)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = nn.functional.elu(x)
        x = self.pool1(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = nn.functional.elu(x)
        x = self.pool2(x)

        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        x = self.dropout(x)
        class_output = self.fc2(x)
        return x, class_output


# 定义模型
class SimpleNN(nn.Module):
    def __init__(self, hidden1, hidden2, hidden3):
        super(SimpleNN, self).__init__()
        self.fc1 = nn.Linear(99, hidden1)
        self.bn1 = nn.BatchNorm1d(hidden1)
        self.fc2 = nn.Linear(hidden1, hidden2)
        self.bn2 = nn.BatchNorm1d(hidden2)
        self.fc3 = nn.Linear(hidden2, hidden3)
        self.bn3 = nn.BatchNorm1d(hidden3)
        self.dropout = nn.Dropout(p=0.5)
        self.fc4 = nn.Linear(hidden2, 1)

    def forward(self, x):
        x = self.fc1(x)
        x = self.bn1(x)
        x = nn.functional.elu(x)
        x = self.fc2(x)
        x = self.bn2(x)
        x = nn.functional.elu(x)
        x = self.dropout(x)
        class_output = self.fc4(x)
        return x, class_output


class MultiModalCNN(nn.Module):
    def __init__(self, hidden1, hidden2, hidden3, input_shape, input_channels, output_channels1, output_channels2,
                 conv_kernel_size1, conv_kernel_size2,
                 pool_kernel_size1, pool_kernel_size2):
        super(MultiModalCNN, self).__init__()
        self.CNN1D_branch = SimpleNN(hidden1, hidden2, hidden3)
        self.CNN2D_branch = CNN2D(input_shape, input_channels, output_channels1, output_channels2, conv_kernel_size1,
                                  conv_kernel_size2,
                                  pool_kernel_size1, pool_kernel_size2)
        self.fc_fusion = nn.Linear(256 + 256, 256)
        self.dropout = nn.Dropout(0.5)
        self.fc_final = nn.Linear(256, 1)

    def forward(self, CNN1D_input, CNN2D_input):
        CNN1D_output, CNN1D_class_output = self.CNN1D_branch(CNN1D_input)
        CNN2D_output, CNN2D_class_output = self.CNN2D_branch(CNN2D_input)

        combined_features = torch.cat((CNN1D_output, CNN2D_output), dim=1)

        fused_features = self.fc_fusion(combined_features)

        fused_features = nn.functional.elu(fused_features)
        fused_features = self.dropout(fused_features)

        combined_output = self.fc_final(fused_features)
        CNN1D_output = self.fc_final(CNN1D_output)
        CNN2D_output = self.fc_final(CNN2D_output)

        return combined_output, CNN1D_output, CNN2D_output
