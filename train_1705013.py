import numpy as np
import os
import cv2
import pandas as pd
import math
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
import seaborn as sns
from tqdm import tqdm
import pickle


class Convolution:
    def __init__(self, output_channels, filter_size, stride, padding):
        self.output_channels = output_channels
        self.filter_size = filter_size
        self.stride = stride
        self.padding = padding
        self.weights = None
        self.bias = None
        self.x = None
        self.x_pad = None
        self.dw = None
        self.db = None

    def forward(self, x):
        if self.weights is None:
            self.weights = np.random.randn(self.output_channels, self.filter_size, self.filter_size, x.shape[3]) / math.sqrt(self.filter_size * self.filter_size)
        if self.bias is None:
            self.bias = np.zeros(self.output_channels)
        self.x = x
        N, H, W, Depth = x.shape
        F, _, _, _ = self.weights.shape
        H_out = (H + 2 * self.padding - self.filter_size) // self.stride + 1
        W_out = (W + 2 * self.padding - self.filter_size) // self.stride + 1
        self.x_pad = np.pad(x, ((0, 0), (self.padding, self.padding), (self.padding, self.padding), (0, 0)), 'constant', constant_values=0)
        out = np.zeros((N, H_out, W_out, F))
        for out_row in range(H_out):
            for out_col in range(W_out):
                row_start = out_row * self.stride
                row_end = row_start + self.filter_size
                col_start = out_col * self.stride
                col_end = col_start + self.filter_size
                x_slice = self.x_pad[:, row_start:row_end, col_start:col_end, :].reshape(N, -1)
                out[:, out_row, out_col, :] = np.dot(x_slice, self.weights.reshape(self.output_channels, -1).T) + self.bias
        return out


    def backward(self, del_v):
        N, _, _, C = self.x.shape
        self.dw = np.zeros(self.weights.shape)
        self.db = np.zeros(self.bias.shape)
        din_pad = np.zeros(self.x_pad.shape)
         
        for out_row in range(del_v.shape[1]):
            for out_col in range(del_v.shape[2]):
                row_start = out_row * self.stride
                row_end = row_start + self.filter_size
                col_start = out_col * self.stride
                col_end = col_start + self.filter_size
                x_slice = self.x_pad[:, row_start:row_end, col_start:col_end, :].reshape(N, -1)
                self.dw += np.dot(del_v[:, out_row, out_col, :].T, x_slice).reshape(self.weights.shape)
                self.db += del_v[:, out_row, out_col, :].sum(axis=0)
                self.weights = np.rot90(self.weights, 2, axes=(1, 2))
                weight_column_matrix = self.weights.reshape(self.output_channels, -1)
                din_pad[:, row_start:row_end, col_start:col_end, :] += np.dot(del_v[:, out_row, out_col, :], weight_column_matrix).reshape(N, self.filter_size, self.filter_size, C)
        self.dw = np.clip(self.dw, -1, 1)
        self.db = np.clip(self.db, -1, 1)
        self.update_weights(0.0004)
        return din_pad[:, self.padding:-self.padding, self.padding:-self.padding, :]

    def update_weights(self, lr):
        self.weights -= lr * self.dw
        self.bias -= lr * self.db
#########################################################################RELUACTIVATION#####################################################################
class ReLUActivation:
    def forward(self, input):
        self.input = input
        return np.maximum(input, 0)
    
    def backward(self, delta_out):
        delta_in = np.array(delta_out, copy=True)
        delta_in[self.input <= 0] = 0
        return delta_in
###########################################################################MAXPOOLING#######################################################################
class MaxPooling:
    def __init__(self, pool_size, stride):
        self.pool_size = pool_size
        self.stride = stride
    
    def forward(self, x):
        N, H, W, C = x.shape
        H_out = (H - self.pool_size) // self.stride + 1
        W_out = (W - self.pool_size) // self.stride + 1
        out = np.zeros((N, H_out, W_out, C))

        for n in range(N):
            for c in range(C):
                for h in range(H_out):
                    for w in range(W_out):
                        out[n, h, w, c] = np.max(x[n, h * self.stride: h * self.stride + self.pool_size, w * self.stride: w * self.stride + self.pool_size, c])

        self.x = x
        self.out = out
        return out
    def backward(self, delta_out):
        N, H, W, C = self.x.shape
        H_out = (H - self.pool_size) // self.stride + 1
        W_out = (W - self.pool_size) // self.stride + 1
        delta_in = np.zeros_like(self.x)

        for n in range(N):
            for c in range(C):
                for h in range(H_out):
                    for w in range(W_out):
                        h_start = h * self.stride
                        h_end = h_start + self.pool_size
                        w_start = w * self.stride
                        w_end = w_start + self.pool_size
                        x_slice = self.x[n, h_start:h_end, w_start:w_end, c]
                        max_index = np.argmax(x_slice)
                        h_max, w_max = np.unravel_index(max_index, (self.pool_size, self.pool_size))
                        delta_in[n, h_start:h_end, w_start:w_end, c][h_max, w_max] = delta_out[n, h, w, c]

        return delta_in

#########################################################################FLATTENING#########################################################################

class Flattening:
    def forward(self, input_tensor):
        self.input_shape = input_tensor.shape
        return input_tensor.reshape(input_tensor.shape[0], -1)
    
    def backward(self, delta_out):
        return delta_out.reshape(self.input_shape)

#########################################################################FULLYCONNECTEDLAYER################################################################

class FullyConnectedLayer:
    def __init__(self, output_size, learning_rate):
        self.weights = None
        self.bias = np.zeros((1, output_size))
        # self.biases = np.zeros((output_size,1))
        self.learning_rate = learning_rate
        # print(self.biases)
        
    def forward(self, input_data):
        self.input_data = input_data
        # print(input_data.shape)
        if self.weights is None:
            self.weights = np.random.normal(0, 0.1, (input_data.shape[1], self.bias.shape[1]))
        self.output = np.dot(input_data, self.weights) + self.bias
        return self.output
    
    def backward(self, error):
        self.dw = np.dot(self.input_data.T, error)
        self.db = np.sum(error, axis=0, keepdims=True)
        self.dw = np.clip(self.dw, -1, 1)
        self.db = np.clip(self.db, -1, 1)
        self.weights -= self.learning_rate * self.dw
        self.bias -= self.learning_rate * self.db
        return np.dot(error, self.weights.T)
###########################################################################SOFTMAX############################################################################

class Softmax:
    def __init__(self, output_size, learning_rate):
        self.weights = None
        self.bias = np.zeros((1, output_size))
        self.learning_rate = learning_rate
        self.output = None

    def forward(self, input_data):
        input_data = input_data - np.max(input_data, axis=1, keepdims=True)
        output_data = np.exp(input_data)
        output_data = output_data / np.sum(output_data, axis=1, keepdims=True)
        self.output = output_data
        return output_data

    def backward(self, error):
        delta_in = np.copy(error)
        return delta_in

class ConvNet:
    def __init__(self, learning_rate):
        self.learning_rate = learning_rate
        self.conv = Convolution(output_channels=6, filter_size=5, stride=1, padding=1)
        self.relu = ReLUActivation()
        self.maxpool = MaxPooling(pool_size=2, stride=2)
        self.flatten = Flattening()
        self.fc = FullyConnectedLayer(output_size=100, learning_rate=learning_rate)
        self.relu2 = ReLUActivation()
        self.fc2 = FullyConnectedLayer(output_size=10, learning_rate=learning_rate)
        self.softmax = Softmax(output_size=10, learning_rate=learning_rate)
        print("ConvNet initialized")
        
    
    def forward(self, input):
        out = self.conv.forward(input)
        out = self.relu.forward(out)
        out = self.maxpool.forward(out)
        out = self.flatten.forward(out)
        out = self.fc.forward(out)
        out = self.relu2.forward(out)
        out = self.fc2.forward(out)
        out = self.softmax.forward(out)
        return out
    
    def backward(self, y_true, y_pred):
        # print(target_probs)
        dscore, _ = self.calcualte_cross_entropy_loss_dscore(y_pred, y_true)
        error = self.softmax.backward(dscore)
        error = self.fc2.backward(error)
        error = self.relu2.backward(error)
        error = self.fc.backward(error)
        error = self.flatten.backward(error)
        error = self.maxpool.backward(error)
        error = self.relu.backward(error)
        error = self.conv.backward(error)

    def calcualte_cross_entropy_loss_dscore(self, y_pred, y_true):
        #Calculation of cross Entropy Loss
        loss = -np.mean(y_true * np.log(y_pred))
        dscore = y_pred - y_true
        return dscore, loss
def save_model(model_path, model):
    parameters = []
    parameters.append(model.conv.weights)
    parameters.append(model.conv.bias)
    parameters.append(model.fc.weights)
    parameters.append(model.fc.bias)
    parameters.append(model.fc2.weights)
    parameters.append(model.fc2.bias)
    with open(model_path, 'wb') as f:
        pickle.dump(parameters, f)

def get_model(model_path):
    with open(model_path, 'rb') as f:
        parameters = pickle.load(f)
    model = ConvNet(learning_rate=0.0004)
    model.conv.weights = parameters[0]
    model.conv.bias = parameters[1]
    model.fc.weights = parameters[2]
    model.fc.bias = parameters[3]
    model.fc2.weights = parameters[4]
    model.fc2.bias = parameters[5]
    print("Model Loaded Successfully")
    return model

def PrepareTestData(image_path, csv_path, percentage):
    folder_path = image_path
    images = [f for f in os.listdir(folder_path) if f.endswith('.png')]
    target_size = (28, 28)
    imgs = []
    for img_name in tqdm(images):
        img_path = os.path.join(folder_path, img_name)
        img = cv2.imread(img_path, 0)
        gray = cv2.resize(img, target_size, interpolation=cv2.INTER_AREA)
        imgs.append(gray)
    mean = np.mean(imgs)
    std = np.std(imgs)
    imgs = (imgs - mean) / std
    imgs = np.array(imgs)
    imgs = np.expand_dims(imgs, axis=-1)

    file_Name = csv_path
    df = pd.read_csv(file_Name)
    labels = df['digit'].values
    labels = np.array(labels)

    return imgs[:int(imgs.shape[0]*percentage)], labels[:int(labels.shape[0]*percentage)]

def train(model, epochs, batch_size, isIndepentTest=False):
    
    X, y = PrepareTestData('./training-a//', 'training-a.csv', 1)
    training_set = int(X.shape[0]*0.8)
    if isIndepentTest:
        training_set = int(X.shape[0]*1)
    X_train, y_train = X[:training_set], y[:training_set]
    X_train_2, y_train_2 = PrepareTestData('./training-c//', 'training-c.csv', 1)
    X_train = np.concatenate((X_train, X_train_2), axis=0)
    y_train = np.concatenate((y_train, y_train_2), axis=0)

    X_train_2, y_train_2 = PrepareTestData('./training-b//', 'training-b.csv', 1)
    X_train = np.concatenate((X_train, X_train_2), axis=0)
    y_train = np.concatenate((y_train, y_train_2), axis=0)
    X_test, y_test = X[training_set:], y[training_set:]
    print("Image and label data loaded")
    print(X_train.shape)
    print(y_train.shape)
    # print(X_test.shape)
    # print(y_test.shape)
    validation_acc_per_epoch = []
    f1_per_epoch = []
    validation_loss_per_epoch = []
    train_loss_per_epoch = []
    loss = None
    for epoch in range(epochs):
        for i in tqdm(range(0, X_train.shape[0], batch_size)):
            batch_imgs = X_train[i:i+batch_size]
            batch_labels = y_train[i:i+batch_size]
            batch_labels_one_hot = np.zeros((batch_labels.shape[0], 10))
            for i, y_i in enumerate(batch_labels):
                batch_labels_one_hot[i, y_i] = 1
            probs = model.forward(batch_imgs)
            _, loss = model.calcualte_cross_entropy_loss_dscore(probs, batch_labels_one_hot)
            model.backward(batch_labels_one_hot, probs)
        if isIndepentTest == False:
            print(f'Epoch {epoch+1}/{epochs}: Training loss: {loss}')
            train_loss_per_epoch.append(loss)
            test_scores = model.forward(X_test)
            y_pred = np.argmax(test_scores, axis=1)
            # print(y_pred)
            valid_acc = accuracy_score(y_test, y_pred)    
            f1 = f1_score(y_test, y_pred, average='macro')
            one_hot = np.zeros((y_test.shape[0], 10))
            for i, y_i in enumerate(y_test):
                one_hot[i, y_i] = 1
            _, valid_loss = model.calcualte_cross_entropy_loss_dscore(test_scores, one_hot)
            validation_acc_per_epoch.append(valid_acc)
            f1_per_epoch.append(f1)
            validation_loss_per_epoch.append(valid_loss)
            print(f'Epoch {epoch+1}/{epochs}: Validation Accuracy: {valid_acc}')
            print(f'Epoch {epoch+1}/{epochs}: Validation F1: {f1}')
            print(f'Epoch {epoch+1}/{epochs}: Validation Loss: {valid_loss}')
    if isIndepentTest == False:
        X = np.arange(epochs)
        plt.plot(X, validation_acc_per_epoch, label='Validation Accuracy')
        plt.plot(X, f1_per_epoch, label='Validation F1')
        plt.legend()
        plt.title('Validation Accuracy and F1 LR='+str(model.learning_rate))
        plt.show()

        plt.plot(X, validation_loss_per_epoch, label='Validation Loss')
        plt.plot(X, train_loss_per_epoch, label='Train Loss')
        plt.legend()
        plt.title('Validation Loss and Train Loss LR='+str(model.learning_rate))
        plt.show()
    save_model('1705013_model.pkl', model)
    return model

# model = ConvNet(learning_rate=0.0004)
# model = train(model, epochs=10, batch_size=50, isIndepentTest=False)
def independentTest():    
    X, y = PrepareTestData('./training-d//', 'training-d.csv', 1)
    # training_set = int(X.shape[0]*0.8)
    X_test, y_test = X, y
    print("Image and label data loaded for independent test")
    print(X_test.shape)
    model = get_model('1705013_model.pkl')
    test_scores = model.forward(X_test)
    y_pred = np.argmax(test_scores, axis=1)
    # print(y_pred)
    acc = accuracy_score(y_test, y_pred)    
    f1 = f1_score(y_test, y_pred, average='macro')
    print(f'Independent Test Accuracy: {acc}')
    print(f'Independent Test F1: {f1}')

    print('\nConfusion Matrix: ')
    print(confusion_matrix(y_test, y_pred))
    #Plot the confusion matrix using seaborn
    plt.figure(figsize=(10,10))
    sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt='d')
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.show()

model = ConvNet(learning_rate=0.0004)
model = train(model, epochs=10, batch_size=50, isIndepentTest=True)
independentTest()