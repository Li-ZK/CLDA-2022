import numpy as np

import scipy.io as sio

from sklearn import preprocessing

def flip(data):
    y_4 = np.zeros_like(data)
    y_1 = y_4
    y_2 = y_4
    first = np.concatenate((y_1, y_2, y_1), axis=1)
    second = np.concatenate((y_4, data, y_4), axis=1)
    third = first
    Data = np.concatenate((first, second, third), axis=0)
    return Data

def load_data(image_file, label_file):
    image_data = sio.loadmat(image_file)
    label_data = sio.loadmat(label_file)
    # print(image_data.keys()) #mine
   
    data_key = image_file.split('/')[-1].split('.')[0]
    label_key = label_file.split('/')[-1].split('.')[0]
    data_all = image_data[data_key]  # dic-> narray , KSC:ndarray(512,217,204)
    GroundTruth = label_data[label_key]

    Data_Band_Scaler = data_all
  
    # data = data_all.reshape(np.prod(data_all.shape[:2]), np.prod(data_all.shape[2:]))  # (111104,204)
    # data_scaler = preprocessing.scale(data)  #标准化 (X-X_mean)/X_std,
    # Data_Band_Scaler = data_scaler.reshape(data_all.shape[0], data_all.shape[1],data_all.shape[2])

    return Data_Band_Scaler, GroundTruth  # image:(512,217,3),label:(512,217)

def cubeData(file_path):
    total = sio.loadmat(file_path)

    data1 = total['DataCube1']
    data2 = total['DataCube2']
    gt1 = total['gt1']
    gt2 = total['gt2']

    print(total.keys())

    Data_Band_Scaler_s = data1
    Data_Band_Scaler_t = data2



    # data_s = data1.reshape(np.prod(data1.shape[:2]), np.prod(data1.shape[2:]))  # (111104,204)
    # data_scaler_s = preprocessing.scale(data_s)  #标准化 (X-X_mean)/X_std,
    # Data_Band_Scaler_s = data_scaler_s.reshape(data1.shape[0], data1.shape[1],data1.shape[2])
    # data_t = data2.reshape(np.prod(data2.shape[:2]), np.prod(data2.shape[2:]))  # (111104,204)
    # data_scaler_t = preprocessing.scale(data_t)  #标准化 (X-X_mean)/X_std,
    # Data_Band_Scaler_t = data_scaler_t.reshape(data2.shape[0], data2.shape[1],data2.shape[2])

    return Data_Band_Scaler_s,Data_Band_Scaler_t, gt1,gt2

def load_data03(image_file, label_file):
    image_data = sio.loadmat(image_file)
    label_data = sio.loadmat(label_file)

    # print(image_data.keys())
    # print(label_data.keys())
    data_all = image_data['ori_data']

    GroundTruth = label_data['map']

    data = data_all.reshape(np.prod(data_all.shape[:2]), np.prod(data_all.shape[2:]))  # (111104,204)
    data_scaler = preprocessing.scale(data)
    Data_Band_Scaler = data_scaler.reshape(data_all.shape[0], data_all.shape[1], data_all.shape[2])

    # Data_Band_Scaler = data_all
    return Data_Band_Scaler, GroundTruth

def all_data(Data_Band_Scaler, GroundTruth,HalfWidth):

    print(Data_Band_Scaler.shape)
    [nRow, nColumn, nBand] = Data_Band_Scaler.shape

    '''label start'''
    data_band_scaler = flip(Data_Band_Scaler)  # (1830, 1020, 103)
    groundtruth = flip(GroundTruth)  # (1830, 1020)

    HalfWidth = HalfWidth
    G = groundtruth[nRow - HalfWidth:2 * nRow + HalfWidth, nColumn - HalfWidth:2 * nColumn + HalfWidth]  # (642, 372)
    data = data_band_scaler[nRow - HalfWidth:2 * nRow + HalfWidth, nColumn - HalfWidth:2 * nColumn + HalfWidth, :]  # (642, 372, 103)

    [Row, Column] = np.nonzero(G)
    train = {}
    m = int(np.max(G))
    #取样
    for i in range(m):
        indices = [j for j, x in enumerate(Row.ravel().tolist()) if G[Row[j], Column[j]] == i + 1]  # G ndarray Row中的索引
        np.random.shuffle(indices)  #
        train[i] = indices

    train_indices = []
   
    for i in range(m):
        train_indices += train[i]
    np.random.shuffle(train_indices)
   
    print('the number of target:', len(train_indices))  # 520
    nTrain = len(train_indices)  # 520
     
    trainX = np.zeros([nTrain,  nBand, 2 * HalfWidth + 1, 2 * HalfWidth + 1], dtype=np.float32)
    trainY = np.zeros([nTrain], dtype=np.int64)
    
    RandPerm = train_indices
    RandPerm = np.array(RandPerm)
    for i in range(nTrain):
        trainX[i, :, :, :] = np.transpose(data[Row[RandPerm[i]] - HalfWidth: Row[RandPerm[i]] + HalfWidth + 1, \
                             Column[RandPerm[i]] - HalfWidth: Column[RandPerm[i]] + HalfWidth + 1, :], (2,0,1))
        trainY[i] = G[Row[RandPerm[i]], Column[RandPerm[i]]].astype(np.int64)
    trainY = trainY - 1

    print('all data shape',trainX.shape)
    print('all label shape',trainY.shape)

    return trainX,trainY,G,RandPerm,Row, Column

def train_test_preclass(Data_Band_Scaler, GroundTruth,HalfWidth,labeled_number):
    print(Data_Band_Scaler.shape)
    [nRow, nColumn, nBand] = Data_Band_Scaler.shape
    data_band_scaler = flip(Data_Band_Scaler)  # (1830, 1020, 103)
    groundtruth = flip(GroundTruth)  # (1830, 1020)

    HalfWidth = HalfWidth
    G = groundtruth[nRow - HalfWidth:2 * nRow + HalfWidth, nColumn - HalfWidth:2 * nColumn + HalfWidth]  # (642, 372)
    data = data_band_scaler[nRow - HalfWidth:2 * nRow + HalfWidth, nColumn - HalfWidth:2 * nColumn + HalfWidth, :]  # (642, 372, 103)

    [Row, Column] = np.nonzero(G)  # (42776,) (42776,) 根据G确定样本所在的行和列

    labels_loc = {}
    train = {}
    test = {}
    m = int(np.max(G))  # 7
    
    #标样本的个数
    nlabeled = labeled_number

    #取样
    for i in range(m):
        indices = [j for j, x in enumerate(Row.ravel().tolist()) if G[Row[j], Column[j]] == i + 1]  # G ndarray Row中的索引
        np.random.shuffle(indices)  #
        nb_val = nlabeled  # 200

        labels_loc[i] = indices

        train[i] = indices[:nb_val]  #
        test[i] = indices[nb_val:]

    train_indices = []
    test_indices = []
    
    for i in range(m):
        train_indices += train[i]
        test_indices += test[i]
    np.random.shuffle(train_indices) # let the same class has different index
    np.random.shuffle(test_indices)
   
    print('the number of the train samples:', len(train_indices))  # 520
    print('the number of the test samples:', len(test_indices))  # 520

    nTrain = len(train_indices)
    nTest = len(test_indices) 
    
    trainX = np.zeros([nTrain,  nBand, 2 * HalfWidth + 1, 2 * HalfWidth + 1], dtype=np.float32)
    trainY = np.zeros([nTrain], dtype=np.int64)
    testX = np.zeros([nTest,  nBand, 2 * HalfWidth + 1, 2 * HalfWidth + 1], dtype=np.float32)
    testY = np.zeros([nTest], dtype=np.int64)

    RandPerm = train_indices + test_indices
    RandPerm = np.array(RandPerm)
    for i in range(nTrain):
        trainX[i, :, :, :] = np.transpose(data[Row[RandPerm[i]] - HalfWidth: Row[RandPerm[i]] + HalfWidth + 1, \
                             Column[RandPerm[i]] - HalfWidth: Column[RandPerm[i]] + HalfWidth + 1, :], (2,0,1))
        trainY[i] = G[Row[RandPerm[i]], Column[RandPerm[i]]].astype(np.int64)
    trainY = trainY - 1

    for i in range(nTest):
        testX[i, :, :, :] = np.transpose(data[Row[RandPerm[i+nTrain]] - HalfWidth: Row[RandPerm[i+nTrain]] + HalfWidth + 1, \
                             Column[RandPerm[i+nTrain]] - HalfWidth: Column[RandPerm[i+nTrain]] + HalfWidth + 1, :], (2,0,1))
        testY[i] = G[Row[RandPerm[i+nTrain]], Column[RandPerm[i+nTrain]]].astype(np.int64)
    testY = testY - 1

    print('train data shape',trainX.shape)
    print('train label shape',trainY.shape)
    print('test data shape',testX.shape)
    print('test label shape',testY.shape)
    return trainX,trainY


