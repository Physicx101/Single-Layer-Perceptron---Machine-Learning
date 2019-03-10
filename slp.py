import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# memuat data dan menambah kolom untuk representasi biner
filename = 'iris.csv'
dataset = pd.read_csv(filename, usecols=[0, 1, 2, 3, 4])
dataset.columns = ['x1', 'x2', 'x3', 'x4', 'species']
dataset['class'] = dataset.species.replace(['Iris-setosa', 'Iris-versicolor'], [0.0, 1.0])


# fungsi aktivasi sigmoid
def sigmoid(result):
    activation = 1 / (1 + np.exp(-result))
    return activation


# membuat prediksi jika aktivasi >= 0.5 maka = 1
def predict(activation):
    return 1.0 if activation >= 0.5 else 0.0


def train_data(dataset, weight, learning_rate, n_epoch):
    dtheta = [0, 0, 0, 0, 0]
    actual = []
    predicted = []
    activation = []
    accuracy = []
    error = []

    for _ in range(n_epoch):
        for i in range(len(dataset)):
            # untuk menghitung hasil
            result = weight[0] * dataset['x1'][i] + weight[1] * dataset['x2'][i] + weight[2] * dataset['x3'][i] + \
                     weight[3] * dataset['x4'][i] + weight[4]

            act = sigmoid(result)

            for j in range(0, len(dtheta) - 1):
                dtheta[j] = 2 * dataset.iloc[i, j] * (dataset['class'][i] - act) * (1 - act) * act

            dtheta[4] = 2 * (dataset['class'][i] - act) * (1 - act) * act

            # weight diupdate jika training
            for x in range(len(weight)):
                weight[x] += learning_rate * dtheta[x]

            prediction = predict(act)

            actual.append(dataset['class'][i])
            activation.append(act)
            predicted.append(prediction)

            # mencari accuracy & error
            acc = accuracy_metric(actual, predicted)
            err = cost_function(actual, activation)

        accuracy.append(acc)
        error.append(err)
    return weight, accuracy, error


def validate_data(dataset, weight, n_epoch):
    actual = []
    predicted = []
    activation = []
    accuracy = []
    error = []

    for _ in range(n_epoch):
        for i in range(len(dataset)):
            # untuk menghitung hasil
            result = weight[0] * dataset['x1'][i] + weight[1] * dataset['x2'][i] + weight[2] * dataset['x3'][i] + \
                     weight[3] * dataset['x4'][i] + weight[4]

            act = sigmoid(result)

            prediction = predict(act)

            actual.append(dataset['class'][i])
            activation.append(act)
            predicted.append(prediction)

            # mencari accuracy & error
            acc = accuracy_metric(actual, predicted)
            err = cost_function(actual, activation)

        accuracy.append(acc)
        error.append(err)
    return accuracy, error


# mencari accuracy menggunakan confusion matrix
def accuracy_metric(actual, predicted):
    tp = 0
    fp = 0
    tn = 0
    fn = 0

    for i in range(len(predicted)):
        # True Positive
        if actual[i] == predicted[i] == 0:
            tp += 1
        # False Positive
        if actual[i] == 1 and actual[i] != predicted[i]:
            fp += 1
        # True Negative
        if actual[i] == predicted[i] == 1:
            tn += 1
        # False Negative
        if actual[i] == 0 and actual[i] != predicted[i]:
            fn += 1
    return (tp + tn) / (tp + fp + tn + fn)


# cost function untuk mencari error
def cost_function(actual, predicted):
    error = 0.0
    for i in range(len(actual)):
        error += 1 / 2 * (predicted[i] - actual[i]) ** 2
    mean = error / len(actual)
    return mean


# K-fold cross validation (k = 5)
data_copy = dataset

# crossval1 = f1+f2+f3+f4 train, f5=val
train_1 = data_copy[0:80]
val_1 = data_copy[80:100]

# crossval2 = f2+f3+f4+f5 train, f1=val
train_2 = data_copy[20:100]
val_2 = data_copy[0:20]

# crossval3 = f1+f2+f4+f5 train, f2=val
a = data_copy[0:20]
b = data_copy[40:100]
train_3 = a.append(b)
val_3 = data_copy[20:40]

# crossval4 = f1+f2+f3+f5 train, f3=val
a = data_copy[0:40]
b = data_copy[60:100]
train_4 = a.append(b)
val_4 = data_copy[40:60]

# crossval5 = f1+f2+f3+f5 train, f4=val
a = data_copy[0:60]
b = data_copy[80:100]
train_5 = a.append(b)
val_5 = data_copy[60:80]

# index direset supaya dimulai dari 0

train_1 = train_1.reset_index(drop=True)
val_1 = val_1.reset_index(drop=True)

train_2 = train_2.reset_index(drop=True)
val_2 = val_2.reset_index(drop=True)

train_3 = train_3.reset_index(drop=True)
val_3 = val_3.reset_index(drop=True)

train_4 = train_4.reset_index(drop=True)
val_4 = val_4.reset_index(drop=True)

train_5 = train_5.reset_index(drop=True)
val_5 = val_5.reset_index(drop=True)

learning_rate_a = 0.1
learning_rate_b = 0.8
n_epoch = 300

# inisialisasi weight awal (0.5)
weight = [0.5, 0.5, 0.5, 0.5, 0.5]

# 5 kali training, dengan epoch = 300, dan learning rate = 0.1
weight_1, accuracy_1, error_1 = train_data(train_1, weight, learning_rate_a, n_epoch)
weight_2, accuracy_2, error_2 = train_data(train_2, weight, learning_rate_a, n_epoch)
weight_3, accuracy_3, error_3 = train_data(train_3, weight, learning_rate_a, n_epoch)
weight_4, accuracy_4, error_4 = train_data(train_4, weight, learning_rate_a, n_epoch)
weight_5, accuracy_5, error_5 = train_data(train_5, weight, learning_rate_a, n_epoch)

# 5 kali validasi, dengan epoch = 300
val_acc_1, val_err_1 = validate_data(val_1, weight_1, n_epoch)
val_acc_2, val_err_2 = validate_data(val_2, weight_2, n_epoch)
val_acc_3, val_err_3 = validate_data(val_3, weight_3, n_epoch)
val_acc_4, val_err_4 = validate_data(val_4, weight_4, n_epoch)
val_acc_5, val_err_5 = validate_data(val_5, weight_5, n_epoch)

mean_training = []
mean_validation = []
mean_error = []
mean_error_v = []

for i in range(len(accuracy_1)):
    mean_training.append((accuracy_1[i] + accuracy_2[i] + accuracy_3[i] + accuracy_4[i] + accuracy_5[i]) / 5)
print(mean_training)

for i in range(len(val_acc_1)):
    mean_validation.append((val_acc_1[i] + val_acc_2[i] + val_acc_3[i] + val_acc_4[i] + val_acc_5[i]) / 5)
print(mean_validation)

# mean error dari train
for i in range(len(error_1)):
    mean_error.append((error_1[i] + error_2[i] + error_3[i] + error_4[i] + error_5[i]) / 5)
print(mean_error)

# mean error dari validasi
for i in range(len(val_err_1)):
    mean_error_v.append((val_err_1[i] + val_err_2[i] + val_err_3[i] + val_err_4[i] + val_err_5[i]) / 5)
print(mean_error_v)

# grafik akurasi
x = plt.figure()
plt.suptitle('Grafik Akurasi Learning Rate 0.1')
plt.xlabel('epoch')
plt.ylabel('accuracy')
plt.plot(mean_validation, '-ob')
plt.plot(mean_training, '-oy')
plt.gca().legend(('akurasi data validasi', 'akurasi data train'))
y = plt.figure()

# grafik error
y.suptitle('Grafik Error Learning Rate 0.1')
plt.xlabel('epoch')
plt.ylabel('error')
plt.plot(mean_error_v, '-ob')
plt.plot(mean_error, '-oy')
plt.gca().legend(('error data validasi', 'error data train'))
