import os
import socket
from common import *
import threading

from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn import svm
from sklearn.model_selection import KFold

import pandas as pd
from io import StringIO
import numpy as np
import random
import math
import time
import re

# DataNode支持的指令有:
# 1. load 加载数据块
# 2. store 保存数据块
# 3. rm 删除数据块
# 4. format 删除所有数据块


class DataNode:
    def run(self):
        # 创建一个监听的socket
        listen_fd = socket.socket()
        thread_fds = []

        try:
            # 监听端口
            listen_fd.bind(("0.0.0.0", data_node_port))
            listen_fd.listen(MAXLOG)
            while True:
                # 等待连接，连接后返回通信用的套接字
                sock_fd, addr = listen_fd.accept()
                print("Received request from {}".format(addr))

                # 创建一个线程用于处理请求，主线程继续监听
                t_fd2 = threading.Thread(
                    target=self.process_request, args=(sock_fd,))
                # 启动线程
                t_fd2.start()
                print("启动线程，处理请求")
                # 将线程的描述符保存起来，用于安全退出
                thread_fds.append(t_fd2)

        except KeyboardInterrupt:
            print("Exiting...")
        except Exception as e:
            print(e)
        finally:
            # 确保所有子线程都已退出
            for t_fd in thread_fds:
                t_fd.join()
            listen_fd.close()

    def process_request(self, sock_fd):
        try:
            # 获取请求方发送的指令
            print('获取请求方发送的指令')
            request = str(recv_all(sock_fd), encoding='utf-8')
            request = request.split()  # 指令之间使用空白符分割
            print(request)

            cmd = request[0]  # 指令第一个为指令类型

            if cmd == "load":  # 加载数据块
                dfs_path = request[1]  # 指令第二个参数为DFS目标地址
                response = self.load(dfs_path)
            elif cmd == "store":  # 存储数据块
                dfs_path = request[1]  # 指令第二个参数为DFS目标地址
                response = self.store(sock_fd, dfs_path)
            elif cmd == "rm":  # 删除数据块
                dfs_path = request[1]  # 指令第二个参数为DFS目标地址
                response = self.rm(dfs_path)
            elif cmd == "format":  # 格式化DFS
                response = self.format()
            elif cmd == "classify_stacking":  # 格式化DFS
                receive = request[1]
                print(receive)
                algorithm = re.match(r'(.*?)\[(.*?)](.*?)', receive)
                if (algorithm == None):
                    algorithm = receive
                    canshu = None
                else:
                    algorithm = algorithm.group(1)
                    canshu = re.match(r'(.*?)\[(.*?)](.*?)', receive).group(2)
                    canshu = canshu.split(',')
                fold_k=int(request[2])
                k_num=int(request[3])
                response = self.classify_stacking(algorithm,fold_k,k_num,canshu)
            elif cmd == "classify_boosting":
                receive = request[1]
                print(receive)
                algorithm = re.match(r'(.*?)\[(.*?)](.*?)', receive)
                if (algorithm == None):
                    algorithm = receive
                    canshu= None
                else:
                    algorithm = algorithm.group(1)
                    canshu = re.match(r'(.*?)\[(.*?)](.*?)', receive).group(2)
                    canshu= canshu.split(',')
                response = self.classify_boosting(sock_fd,algorithm,canshu)
            elif cmd == "read_test_result":
                response = self.read_test_result()
            else:
                response = "Undefined command: " + " ".join(request)

            # 如果数据本身不是是bytes类型，则需要编传输码
            if not isinstance(response, bytes):
                response = bytes(response, encoding='utf-8')

            send_all(sock_fd, response)
        except KeyboardInterrupt:
            raise KeyboardInterrupt
        finally:
            sock_fd.close()

    def load(self, dfs_path):
        # 本地路径
        local_path = data_node_dir + dfs_path
        # 读取本地数据
        with open(local_path, 'rb') as f:
            chunk_data = f.read(dfs_blk_size)

        return chunk_data

    def store(self, sock_fd, dfs_path):
        # 从Client获取块数据
        chunk_data = recv_all(sock_fd)
        # 本地路径
        local_path = data_node_dir + dfs_path
        # 若目录不存在则创建新目录
        os.system("mkdir -p {}".format(os.path.dirname(local_path)))
        # 将数据块写入本地文件
        with open(local_path, "wb") as f:
            f.write(chunk_data)

        return "Store chunk {} successfully~".format(local_path)

    def rm(self, dfs_path):
        local_path = data_node_dir + dfs_path
        rm_command = "rm -rf " + local_path
        os.system(rm_command)

        return "Remove chunk {} successfully~".format(local_path)

    def format(self):
        format_command = "rm -rf {}/*".format(data_node_dir)
        os.system(format_command)

        return "Format datanode successfully~"

    def classify_boosting(self, sock_fd, algorithm,canshu):

        dataframe = pd.read_csv("/home/dsjxtjc/2019211331/dfs/phoneme.dat", header=None, skiprows=[0, 1], sep='\s+')
        data = dataframe.values
        random.shuffle(data)

        for i in range(data.shape[0]):
            if (data[i, -1] == 0):
                data[i, -1] = -1

        train_data = data[0:5000, :]
        test_data = data[5000:, :]
        train_x = train_data[:, 0:-1]
        train_y = train_data[:, -1]
        test_x = test_data[:, 0:-1]
        test_y = test_data[:, -1]

        # dataframe = pd.read_csv("/home/dsjxtjc/2019211331/dfs/test_img.csv", header=None)
        # test_x= dataframe.values
        # dataframe = pd.read_csv("/home/dsjxtjc/2019211331/dfs/test_labels.csv", header=None)
        # test_y= dataframe.values
        # dataframe = pd.read_csv("/home/dsjxtjc/2019211331/dfs/train_img.csv", header=None)
        # train_x= dataframe.values
        # dataframe = pd.read_csv("/home/dsjxtjc/2019211331/dfs/train_labels.csv", header=None)
        # train_y= dataframe.values
        #
        # test_y=np.sum(test_y,axis=1)
        # train_y=np.sum(train_y,axis=1)

        for i in range(train_y.shape[0]):
            if (train_y[i] == 0):
                train_y[i] = -1
        for i in range(test_y.shape[0]):
            if (test_y[i] == 0):
                test_y[i] = -1

        # 从server获取权重
        weight_file = str(recv_all(sock_fd), encoding='utf-8')
        time.sleep(1)

        result = pd.read_csv(StringIO(weight_file),header = None)
        result = result.values
        print('result',result.shape)
        weight = np.sum(result, axis=1)  # 按列求和
        print('weight',weight)

        def select(weight):
            num_ = [i for i in range(train_x.shape[0])]
            # 概率列表
            sum_ = 0
            ran = random.random()
            for num, r in zip(num_, weight):
                sum_ += r
                if ran < sum_: break
            return num


        def decision_tree(train_x,train_y,test_x,test_y,weight,canshu):
            #默认参数
            criterion='gini'
            max_depth = None
            min_sample_leaf=1
            if(canshu!=None):
                for line in canshu:
                    loc = line.index('=')
                    name = line[0:loc]
                    if(name=='citetion'):
                        criterion=line[loc + 1:]
                    if (name == 'max_depth'): max_depth = int(line[loc + 1:])
                    if (name == 'min_samples_leaf'): min_sample_leaf = int(line[loc + 1:])

            estimator = DecisionTreeClassifier(criterion=criterion,
                                               max_depth=max_depth,min_samples_leaf=min_sample_leaf)
            estimator.fit(train_x, train_y,sample_weight=weight)
            predict_y_train = estimator.predict(train_x)
            if_true_train=[predict_y_train==train_y]

            predict_y_test=estimator.predict(test_x)
            if_true_test =[predict_y_test==test_y]

            score = estimator.score(train_x, train_y,sample_weight=weight)
            print('训练集准确率',score)
            score = estimator.score(test_x, test_y)
            print('测试集准确率',score)
            return if_true_train,if_true_test

        def random_forest(train_x,train_y,test_x,test_y,weight,canshu):

            #默认参数
            criterion='gini'
            max_depth = None
            n_estimators=20
            if(canshu!=None):
                for line in canshu:
                    loc = line.index('=')
                    name = line[0:loc]
                    if(name=='citetion'):
                        criterion=line[loc + 1:]
                    if (name == 'max_depth'): max_depth = int(line[loc + 1:])
                    if (name == 'n_estimators'): n_estimators = int(line[loc + 1:])


            estimator = RandomForestClassifier(n_estimators = n_estimators,criterion=criterion,max_depth=max_depth)
            estimator.fit(train_x, train_y,sample_weight=weight)
            predict_y_train = estimator.predict(train_x)
            if_true_train=[predict_y_train==train_y]

            predict_y_test=estimator.predict(test_x)
            if_true_test =[predict_y_test==test_y]

            score = estimator.score(train_x, train_y,sample_weight=weight)
            print('训练集准确率',score)
            score = estimator.score(test_x, test_y)
            print('测试集准确率',score)
            return if_true_train,if_true_test

        def svm_func(train_x,train_y,test_x,test_y,weight,canshu):

            # 默认参数
            C = 1
            kernel = "rbf"
            if (canshu != None):
                for line in canshu:
                    loc = line.index('=')
                    name = line[0:loc]
                    if (name == 'kernel'):
                        kernel = line[loc + 1:]
                    if (name == 'C'): C = float(line[loc + 1:])

            svm_cla = svm.SVC(C=C,kernel=kernel)
            svm_cla.fit(train_x, train_y, sample_weight=weight)

            predict_y_train = svm_cla.predict(train_x)
            if_true_train=[predict_y_train==train_y]
            predict_y_test=svm_cla.predict(test_x)
            if_true_test =[predict_y_test==test_y]

            score = svm_cla.score(train_x, train_y,sample_weight=weight)
            print('训练集准确率',score)
            score = svm_cla.score(test_x, test_y)
            print('测试集准确率',score)

            return if_true_train,if_true_test

        def logistic_regression(train_x,train_y,test_x,test_y,weight):
            estimator = LogisticRegression()
            estimator.fit(train_x, train_y,sample_weight=weight)
            predict_y_train = estimator.predict(train_x)
            if_true_train=[predict_y_train==train_y]
            predict_y_test=estimator.predict(test_x)
            if_true_test =[predict_y_test==test_y]

            score = estimator.score(train_x, train_y,sample_weight=weight)
            print('训练集准确率',score)
            score = estimator.score(test_x, test_y)
            print('测试集准确率',score)

            return if_true_train,if_true_test

        def naive_bayes(train_x, train_y, test_x, test_y, weight):
            estimator = GaussianNB()
            estimator.fit(train_x, train_y, sample_weight=weight)
            predict_y_train = estimator.predict(train_x)
            if_true_train = [predict_y_train == train_y]
            predict_y_test = estimator.predict(test_x)
            if_true_test = [predict_y_test == test_y]

            score = estimator.score(train_x, train_y, sample_weight=weight)
            print('训练集准确率', score)
            score = estimator.score(test_x, test_y)
            print('测试集准确率', score)

            return if_true_train, if_true_test

        def knn(train_x, train_y, test_x, test_y, weight,canshu):
            # 默认参数
            n_neighbors = 2
            if (canshu != None):
                for line in canshu:
                    loc = line.index('=')
                    name = line[0:loc]
                    if (name == 'n_neighbors'): n_neighbors = int(line[loc + 1:])


            #按权重选择样本分布
            train_x_weight=[]
            train_y_weight=[]
            if(train_x.shape[0]>5000):
                sample_num = 5000
            else:
                sample_num = train_x.shape[0]
            for a in range(sample_num):
                num=select(weight)
                train_x_weight.append(train_x[num,:])
                train_y_weight.append(train_y[num])
            train_x_weight=np.array(train_x_weight)
            train_y_weight=np.array(train_y_weight)

            estimator = KNeighborsClassifier(n_neighbors=n_neighbors)
            estimator.fit(train_x_weight, train_y_weight)
            predict_y_train = estimator.predict(train_x)
            if_true_train = [predict_y_train == train_y]
            predict_y_test = estimator.predict(test_x)
            if_true_test = [predict_y_test == test_y]

            score = estimator.score(train_x, train_y)
            print('训练集准确率', score)
            score = estimator.score(test_x, test_y)
            print('测试集准确率', score)

            return if_true_train, if_true_test

        def neural_network(train_x, train_y, test_x, test_y, weight,canshu):

            #默认参数
            activation='relu'
            solver = 'adam'
            batch_size='auto'
            learning_rate=0.001
            max_iter=200
            if(canshu!=None):
                for line in canshu:
                    loc = line.index('=')
                    name = line[0:loc]
                    if(name=='activation'):
                        activation=line[loc + 1:]
                    if (name == 'solver'): solver = line[loc + 1:]
                    if (name == 'batch_size'): batch_size = int(line[loc + 1:])
                    if (name == 'learning_rate'): learning_rate = float(line[loc + 1:])
                    if (name == 'epoch'): max_iter = int(line[loc + 1:])


            #按权重选择样本分布
            train_x_weight=[]
            train_y_weight=[]
            if(train_x.shape[0]>5000):
                sample_num = 5000
            else:
                sample_num = train_x.shape[0]
            for a in range(sample_num):
                num=select(weight)
                train_x_weight.append(train_x[num,:])
                train_y_weight.append(train_y[num])
            train_x_weight=np.array(train_x_weight)
            train_y_weight=np.array(train_y_weight)

            estimator = MLPClassifier(activation=activation,solver=solver,
                                      batch_size=batch_size,learning_rate_init=learning_rate,max_iter=max_iter)
            estimator.fit(train_x_weight, train_y_weight)

            predict_y_train = estimator.predict(train_x)
            if_true_train = [predict_y_train == train_y]
            predict_y_test = estimator.predict(test_x)
            if_true_test = [predict_y_test == test_y]

            score = estimator.score(train_x, train_y)
            print('训练集准确率', score)
            score = estimator.score(test_x, test_y)
            print('测试集准确率', score)

            return if_true_train, if_true_test


        if algorithm== "decision_tree":
            if_true_train,if_true_test=decision_tree(train_x,train_y,test_x,test_y,weight,canshu)
        if algorithm== "random_forest":
            if_true_train,if_true_test=random_forest(train_x,train_y,test_x,test_y,weight,canshu)
        if algorithm == "svm":
            if_true_train,if_true_test=svm_func(train_x,train_y,test_x,test_y,weight,canshu)
        if algorithm == "LR":
            if_true_train, if_true_test = logistic_regression(train_x, train_y, test_x, test_y, weight)
        if algorithm == "naive_bayes":
            if_true_train, if_true_test = naive_bayes(train_x, train_y, test_x, test_y, weight)
        if algorithm == "knn":
            if_true_train, if_true_test = knn(train_x, train_y, test_x, test_y, weight,canshu)
        if algorithm == "nn":
            if_true_train, if_true_test = neural_network(train_x, train_y, test_x, test_y, weight,canshu)

        if_true_train = np.array(if_true_train)
        if_true_test = np.array(if_true_test)
        if_true_train=if_true_train[0]
        if_true_test=if_true_test[0]
        # print(if_true[0:10])
        print(if_true_test.shape)

        path_train_result= "/home/dsjxtjc/2019211331/dfs/result/train_score.csv"
        path_test_result = "/home/dsjxtjc/2019211331/dfs/result/test_score.csv"
        # 若目录不存在则创建新目录
        os.system("mkdir -p {}".format(os.path.dirname(path_train_result)))
        os.system("mkdir -p {}".format(os.path.dirname(path_test_result)))

        data1 = pd.DataFrame(if_true_train)
        data1.to_csv(path_train_result, mode= 'a',header=False, index=False)
        data2 = pd.DataFrame(if_true_test) #保存测试结果
        data2.to_csv(path_test_result, header=False, index=False)

        return data1.to_csv(header=False, index=False)

    def read_test_result(self):
        path_test_result = "/home/dsjxtjc/2019211331/dfs/result/test_score.csv"
        df=pd.read_csv(path_test_result,header=None)
        # test_result = df.to_csv(path, header=False, index=False)
        return df.to_csv(header=False, index=False)

    def classify_stacking(self,algorithm,fold_k,k_num,canshu):

        dataframe = pd.read_csv("/home/dsjxtjc/2019211331/dfs/phoneme.dat", header=None, skiprows=[0, 1], sep='\s+')
        data = dataframe.values
        random.shuffle(data)

        for i in range(data.shape[0]):
            if (data[i, -1] == 0):
                data[i, -1] = -1

        train_data = data[0:5000, :]
        test_data = data[5000:, :]
        train_x = train_data[:, 0:-1]
        train_y = train_data[:, -1]
        test_x = test_data[:, 0:-1]
        test_y = test_data[:, -1]

        # dataframe = pd.read_csv("/home/dsjxtjc/2019211331/dfs/test_img.csv", header=None)
        # test_x= dataframe.values
        # dataframe = pd.read_csv("/home/dsjxtjc/2019211331/dfs/test_labels.csv", header=None)
        # test_y= dataframe.values
        # dataframe = pd.read_csv("/home/dsjxtjc/2019211331/dfs/train_img.csv", header=None)
        # train_x= dataframe.values
        # dataframe = pd.read_csv("/home/dsjxtjc/2019211331/dfs/train_labels.csv", header=None)
        # train_y= dataframe.values
        # test_y=np.sum(test_y,axis=1)
        # train_y=np.sum(train_y,axis=1)


        for i in range(train_y.shape[0]):
            if (train_y[i] == 0):
                train_y[i] = -1
        for i in range(test_y.shape[0]):
            if (test_y[i] == 0):
                test_y[i] = -1


        train_index_list = []
        test_index_list = []
        kf = KFold(n_splits=fold_k, shuffle=False)
        for train_index, test_index in kf.split(train_x):
            train_index_list.append(train_index)
            test_index_list.append(test_index)

        train_x_k=train_x[train_index_list[k_num],:]
        train_y_k=train_y[train_index_list[k_num]]
        test_x_k=train_x[test_index_list[k_num],:]
        test_y_k=train_y[test_index_list[k_num]]

        def decision_tree(train_x_k,train_y_k,test_x_k,test_y_k,canshu):
            # 默认参数
            criterion = 'gini'
            max_depth = None
            min_sample_leaf = 1
            if (canshu != None):
                for line in canshu:
                    loc = line.index('=')
                    name = line[0:loc]
                    if (name == 'citetion'):
                        criterion = line[loc + 1:]
                    if (name == 'max_depth'): max_depth = int(line[loc + 1:])
                    if (name == 'min_samples_leaf'): min_sample_leaf = int(line[loc + 1:])

            estimator = DecisionTreeClassifier(criterion=criterion,
                                               max_depth=max_depth, min_samples_leaf=min_sample_leaf)
            estimator.fit(train_x_k, train_y_k)

            predict_y_k = estimator.predict_proba(test_x_k)
            # print('概率',predict_y_k)
            predict_y_test = estimator.predict_proba(test_x)

            score = estimator.score(train_x_k, train_y_k)
            print('训练集准确率', score)
            score = estimator.score(test_x, test_y)
            print('测试集准确率', score)

            return predict_y_k,predict_y_test

        def svm_func(train_x_k,train_y_k,test_x_k,test_y_k,canshu):
            # 默认参数
            C = 1
            kernel = "rbf"
            if (canshu != None):
                for line in canshu:
                    loc = line.index('=')
                    name = line[0:loc]
                    if (name == 'kernel'):
                        kernel = line[loc + 1:]
                    if (name == 'C'): C = float(line[loc + 1:])

            estimator = svm.SVC(C=C, kernel=kernel, probability=True)
            estimator.fit(train_x_k, train_y_k)

            predict_y_k = estimator.predict_proba(test_x_k)
            predict_y_k=predict_y_k[:,0]
            # print('概率',predict_y_k)
            predict_y_test = estimator.predict_proba(test_x)

            score = estimator.score(train_x_k, train_y_k)
            print('训练集准确率', score)
            score = estimator.score(test_x, test_y)
            print('测试集准确率', score)

            return predict_y_k,predict_y_test

        def random_forest(train_x_k,train_y_k,test_x_k,test_y_k,canshu):

            #默认参数
            criterion='gini'
            max_depth = None
            n_estimators=20
            if(canshu!=None):
                for line in canshu:
                    loc = line.index('=')
                    name = line[0:loc]
                    if(name=='citetion'):
                        criterion=line[loc + 1:]
                    if (name == 'max_depth'): max_depth = int(line[loc + 1:])
                    if (name == 'n_estimators'): n_estimators = int(line[loc + 1:])


            estimator = RandomForestClassifier(n_estimators = n_estimators,criterion=criterion,
                                               max_depth=max_depth)
            estimator.fit(train_x_k, train_y_k)

            predict_y_k = estimator.predict_proba(test_x_k)
            predict_y_k=predict_y_k[:,0]
            # print('概率',predict_y_k)
            predict_y_test = estimator.predict_proba(test_x)

            score = estimator.score(train_x_k, train_y_k)
            print('训练集准确率', score)
            score = estimator.score(test_x, test_y)
            print('测试集准确率', score)

            return predict_y_k,predict_y_test

        def neural_network(train_x_k,train_y_k,test_x_k,test_y_k,canshu):

            #默认参数
            activation='relu'
            solver = 'adam'
            batch_size='auto'
            learning_rate=0.001
            max_iter=200
            if(canshu!=None):
                for line in canshu:
                    loc = line.index('=')
                    name = line[0:loc]
                    if(name=='activation'):
                        activation=line[loc + 1:]
                    if (name == 'solver'): solver = line[loc + 1:]
                    if (name == 'batch_size'): batch_size = int(line[loc + 1:])
                    if (name == 'learning_rate'): learning_rate = float(line[loc + 1:])
                    if (name == 'epoch'): max_iter = int(line[loc + 1:])


            estimator = MLPClassifier(activation=activation,solver=solver,
                                      batch_size=batch_size,learning_rate_init=learning_rate,max_iter=max_iter)
            estimator.fit(train_x_k, train_y_k)

            predict_y_k = estimator.predict_proba(test_x_k)
            predict_y_k = predict_y_k[:, 0]
            # print('概率', predict_y_k)
            predict_y_test = estimator.predict_proba(test_x)

            score = estimator.score(train_x_k, train_y_k)
            print('训练集准确率', score)
            score = estimator.score(test_x, test_y)
            print('测试集准确率', score)

            return predict_y_k, predict_y_test

        def naive_bayes(train_x_k,train_y_k,test_x_k,test_y_k,canshu):
            estimator = GaussianNB()
            estimator.fit(train_x_k, train_y_k)

            predict_y_k = estimator.predict_proba(test_x_k)
            predict_y_k = predict_y_k[:, 0]
            # print('概率', predict_y_k)
            predict_y_test = estimator.predict_proba(test_x)

            score = estimator.score(train_x_k, train_y_k)
            print('训练集准确率', score)
            score = estimator.score(test_x, test_y)
            print('测试集准确率', score)

            return predict_y_k, predict_y_test

        def logistic_regression(train_x_k,train_y_k,test_x_k,test_y_k,canshu):

            estimator = LogisticRegression()
            estimator.fit(train_x_k, train_y_k)

            predict_y_k = estimator.predict_proba(test_x_k)
            predict_y_k = predict_y_k[:, 0]
            # print('概率', predict_y_k)
            predict_y_test = estimator.predict_proba(test_x)

            score = estimator.score(train_x_k, train_y_k)
            print('训练集准确率', score)
            score = estimator.score(test_x, test_y)
            print('测试集准确率', score)
            return predict_y_k, predict_y_test

        if algorithm== "decision_tree":
            predict_y_k,predict_y_test = decision_tree(train_x_k,train_y_k,test_x_k,test_y_k,canshu)
        if algorithm== "random_forest":
            predict_y_k,predict_y_test = random_forest(train_x_k,train_y_k,test_x_k,test_y_k,canshu)
        if algorithm == "svm":
            predict_y_k,predict_y_test =svm_func(train_x_k,train_y_k,test_x_k,test_y_k,canshu)
        if algorithm == "LR":
            predict_y_k,predict_y_test =logistic_regression(train_x_k,train_y_k,test_x_k,test_y_k,canshu)
        if algorithm == "naive_bayes":
            predict_y_k,predict_y_test =naive_bayes(train_x_k,train_y_k,test_x_k,test_y_k,canshu)
        # if algorithm == "knn":
        #     if_true_train, if_true_test = knn(train_x, train_y, test_x, test_y, weight,canshu)
        if algorithm == "nn":
            predict_y_k,predict_y_test =neural_network(train_x_k,train_y_k,test_x_k,test_y_k,canshu)

        path_k_result = "/home/dsjxtjc/2019211331/dfs/result/predict_k.csv"
        path_test_result = "/home/dsjxtjc/2019211331/dfs/result/predict_test.csv"
        # 若目录不存在则创建新目录
        os.system("mkdir -p {}".format(os.path.dirname(path_k_result)))
        os.system("mkdir -p {}".format(os.path.dirname(path_test_result)))

        data1 = pd.DataFrame(predict_y_k)
        data1.to_csv(path_k_result, header=False, index=False)

        data2 = pd.DataFrame(predict_y_test)  # 保存测试结果
        data2.to_csv(path_test_result, header=False, index=False)

        return data1.to_csv(header=False, index=False)



# 创建DataNode对象并启动
data_node = DataNode()
data_node.run()



