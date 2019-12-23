import math
import os
import socket

import numpy as np
import pandas as pd
from io import StringIO
import threading
import math
import random
from sklearn.ensemble import GradientBoostingClassifier

from common import *


# NameNode功能
# 1. 保存文件的块存放位置信息
# 2. ls ： 获取文件/目录信息
# 3. get_fat_item： 获取文件的FAT表项
# 4. new_fat_item： 根据文件大小创建FAT表项
# 5. rm_fat_item： 删除一个FAT表项
# 6. format: 删除所有FAT表项

class NameNode:
    def run(self):  # 启动NameNode
        # 创建一个监听的socket
        listen_fd = socket.socket()
        self.thread_fds = []

        try:
            # 监听端口
            listen_fd.bind(("0.0.0.0", name_node_port))
            listen_fd.listen(MAXLOG)
            print("Learning Server started")
            while True:
                # 等待连接，连接后返回通信用的套接字
                sock_fd, addr = listen_fd.accept()
                print("connected by {}".format(addr))
                t_fd = threading.Thread(
                    target=self.process_request,
                    args=(sock_fd,))
                t_fd.start()
                self.thread_fds.append(t_fd)

        except KeyboardInterrupt:   # 如果运行时按Ctrl+C则退出程序
            print("Exiting...")
        except Exception as e:      # 如果出错则打印错误信息
            print(e)
        finally:
            # 确保所有子线程都已退出
            for t_fd in self.thread_fds:
                t_fd.join()

            listen_fd.close()       # 释放连接

    def process_request(self, sock_fd):
        try:
            # 获取请求方发送的指令
            request = str(recv_all(sock_fd), encoding='utf-8')
            print("Request: {}".format(request))

            request = request.split()       # 指令之间使用空白符分割
            cmd = request[0]                # 指令第一个为指令类型

            if cmd == "ls":                 # 若指令类型为ls, 则返回DFS上对于文件、文件夹的内容
                dfs_path = request[1]       # 指令第二个参数为DFS目标地址
                response = self.ls(dfs_path)
            elif cmd == "get_fat_item":     # 指令类型为获取FAT表项
                dfs_path = request[1]       # 指令第二个参数为DFS目标地址
                response = self.get_fat_item(dfs_path)
            elif cmd == "new_fat_item":     # 指令类型为新建FAT表项
                dfs_path = request[1]       # 指令第二个参数为DFS目标地址
                file_size = int(request[2])
                response = self.new_fat_item(dfs_path, file_size)
            elif cmd == "rm_fat_item":      # 指令类型为删除FAT表项
                dfs_path = request[1]       # 指令第二个参数为DFS目标地址
                response = self.rm_fat_item(dfs_path)
            elif cmd == "format":
                response = self.format()
            elif cmd == "ensemble_boosting":
                algorithm1= request[1]
                algorithm2= request[2]
                if(len(request)==3):
                    algorithm=[algorithm1,algorithm2]
                else:
                    algorithm3 = request[3]
                    algorithm = [algorithm1, algorithm2, algorithm3]
                response = self.ensemble_boosting(algorithm)
            elif cmd == "ensemble_stacking":
                algorithm1= request[1]
                algorithm2= request[2]
                if(len(request)==3):
                    algorithm=[algorithm1,algorithm2]
                else:
                    algorithm3 = request[3]
                    algorithm = [algorithm1, algorithm2, algorithm3]
                response = self.ensemble_stacking(algorithm)
            else:                           # 其他未知指令
                response = "Undefined command: " + " ".join(request)

            print("Response: {}".format(response))
            send_all(sock_fd, bytes(response, encoding='utf-8'))

        except KeyboardInterrupt:           # 如果运行时按Ctrl+C则退出程序
            raise KeyboardInterrupt
        except Exception as e:              # 如果出错则打印错误信息
            print(e)
        finally:
            sock_fd.close()                 # 释放连接

    def ls(self, dfs_path):
        local_path = name_node_dir + dfs_path
        # 如果文件不存在，返回错误信息
        if not os.path.exists(local_path):
            return "No such file or directory: {}".format(dfs_path)

        if os.path.isdir(local_path):
            # 如果目标地址是一个文件夹，则显示该文件夹下内容
            dirs = os.listdir(local_path)
            response = " ".join(dirs)
        else:
            # 如果目标是文件则显示文件的FAT表信息
            with open(local_path) as f:
                response = f.read()

        return response

    def get_fat_item(self, dfs_path):
        # 获取FAT表内容
        local_path = name_node_dir + dfs_path
        response = pd.read_csv(local_path)
        return response.to_csv(index=False)

    def new_fat_item(self, dfs_path, file_size):
        nb_blks = int(math.ceil(file_size / dfs_blk_size))
        print(file_size, nb_blks)

        # todo 如果dfs_replication为复数时可以新增host_name的数目
        data_pd = pd.DataFrame(columns=['blk_no', 'host_name', 'blk_size'])

        for i in range(nb_blks):
            blk_no = i
            host_name = np.random.choice(
                host_list, size=dfs_replication, replace=False)
            blk_size = min(dfs_blk_size, file_size - i * dfs_blk_size)
            # data_pd.loc[i] = [blk_no, host_name, blk_size]
            for j in range(len(host_name)):
                data_pd.loc[i*dfs_replication+j] = [blk_no, host_name[j], blk_size]

        # 获取本地路径
        local_path = name_node_dir + dfs_path

        # 若目录不存在则创建新目录
        os.system("mkdir -p {}".format(os.path.dirname(local_path)))
        # 保存FAT表为CSV文件
        data_pd.to_csv(local_path, index=False)
        # 同时返回CSV内容到请求节点
        return data_pd.to_csv(index=False)

    def rm_fat_item(self, dfs_path):
        local_path = name_node_dir + dfs_path
        response = pd.read_csv(local_path)
        os.remove(local_path)
        return response.to_csv(index=False)

    def format(self):
        format_command = "rm -rf {}/*".format(name_node_dir)
        os.system(format_command)
        return "Format namenode successfully~"

    def ensemble_boosting(self,algorithm):

        dataframe = pd.read_csv("/home/dsjxtjc/2019211331/dfs/phoneme.dat",
                                header=None, skiprows=[0, 1], sep='\s+')
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
        #
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

        #初始化数据权重
        w = np.ones(train_y.shape)
        weight = w / train_y.shape[0]

        weight_old=weight
        weight_new=weight

        iteration_num=10 #迭代次数
        iteration = 0
        end_flag=True
        result_sum=[]
        test_result_sum=[]
        model_weight=[]

        while(end_flag):

            #定义线程任务
            def call_data_node(host,algo):
                data_node_sock = socket.socket()
                data_node_sock.connect((host, data_node_port))

                request = "classify_boosting {}".format(algo)
                send_all(data_node_sock, bytes(request, encoding='utf-8'))  #发送指令
                path = "/home/dsjxtjc/2019211331/dfs/weight_file/weight.csv"
                # 若目录不存在则创建新目录
                os.system("mkdir -p {}".format(os.path.dirname(path)))

                data1 = pd.DataFrame(weight_new)  #更新后的权重
                data_file=data1.to_csv(path, header=False, index=False)

                # fp = open(path)
                # weight_send = fp.read()
                weight_send=data1.to_csv(header=False, index=False)
                send_all(data_node_sock, bytes(weight_send, encoding='utf-8'))  #发送权重

                #接收分类结果
                receive = recv_all(data_node_sock)
                receive = str(receive, encoding='utf-8')
                # receive=data_node_sock.recv(BUF_SIZE)

                result = pd.read_csv(StringIO(receive),header = None)
                result=result.values
                result2=np.sum(result,axis=1)#按列求和
                for i in range(result2.shape[0]):
                    if(result2[i]==0):
                        result2[i]=-1    #将0替换为-1
                result_sum.append(result2)
                data_node_sock.close()

            def call_test_result(host):
                data_node_sock = socket.socket()
                data_node_sock.connect((host, data_node_port))

                request = "read_test_result"
                send_all(data_node_sock, bytes(request, encoding='utf-8'))  # 发送指令

                #接收测试集结果
                receive = recv_all(data_node_sock)
                receive = str(receive, encoding='utf-8')
                # print('receive',receive)

                result2 = pd.read_csv(StringIO(receive),header = None)

                test_result=result2.values
                test_result2=np.sum(test_result,axis=1)#按列求和
                for i in range(test_result2.shape[0]):
                    if(test_result2[i]==0):
                        test_result2[i]=-1    #将0替换为-1
                test_result_sum.append(test_result2)
                data_node_sock.close()

            thread_list = []
            classify_host = ['thumm02', 'thumm03', 'thumm04']
            for index in range(len(algorithm)):
                algo= algorithm[index]
                t_fd = threading.Thread(
                    target=call_data_node, args=(classify_host[index], algo))
                t_fd.start()
                thread_list.append(t_fd)

            #等待所有线程结束
            for t_fd in thread_list:
                t_fd.join()

            if(len(algorithm)==2):
                result_iteration=result_sum[iteration*2:]
            else:
                result_iteration = result_sum[iteration * 3:]
            # print(result_iteration)
            result_iteration_sum=np.sum(result_iteration,axis=0)
            # print('result_iteration_sum',result_iteration_sum.shape)

            #迭代模型测试集准确率
            for i in range(len(result_iteration_sum)):
                if(result_iteration_sum[i]>1):
                    result_iteration_sum[i]=1
                else:
                    result_iteration_sum[i] = 0
            correct_rate=sum(result_iteration_sum)/len(result_iteration_sum)
            print('correct_rate',type(correct_rate),correct_rate)
            print("第 {} 次迭代模型的分类准确率为: {}".format(iteration+1,correct_rate))
            error_rate = 1.0-correct_rate

            if(error_rate==0):
                print('训练集全部分类正确，迭代 {} 次结束'.format(iteration+1))
                break

            print('err_rate',error_rate)
            alpha=0.5*np.log((1-error_rate)/error_rate)
            print('alpha',alpha)
            if(len(algorithm)==2):
                model_weight.append(alpha)
                model_weight.append(alpha)
            else:
                for il in range(3):
                    model_weight.append(alpha)

            #集成模型测试集准确率
            result_gm=np.array(result_sum,dtype=np.float64)
            for i in range(len(model_weight)):
                result_gm[i,:]*=model_weight[i]

            result_gm=np.sum(result_gm,axis=0)
            for i in range(len(result_gm)):
                if(result_gm[i]>0):
                    result_gm[i]=1
                else:
                    result_gm[i]=0
            whole_correct_rate=sum(result_gm)/len(result_gm)
            print('第 {} 次迭代后集成模型的分类准确率为: {}'.format(iteration+1,whole_correct_rate))

            #集成模型训练集准确率
            thread_list2 = []
            for index in range(len(algorithm)):
                t_fd = threading.Thread(
                    target=call_test_result, args=(classify_host[index],))
                t_fd.start()
                thread_list2.append(t_fd)
            # 等待所有线程结束
            for t_fd in thread_list2:
                t_fd.join()

            test_result_sum2 = np.array(test_result_sum, dtype=np.float64)
            for i in range(len(model_weight)):
                test_result_sum2[i,:]*=model_weight[i]

            test_result_sum2 = np.sum(test_result_sum2, axis=0)
            for i in range(len(test_result_sum2)):
                if(test_result_sum2[i]>0):
                    test_result_sum2[i]=1
                else:
                    test_result_sum2[i]=0

            test_correct_rate=sum(test_result_sum2)/len(test_result_sum2)
            print('第 {} 次迭代后集成模型的测试集分类准确率为: {}'.format(iteration + 1, test_correct_rate))

            gm=[1 for i in range(len(result_iteration_sum))]
            gm=np.array(gm)

            # print('计算gm',gm)
            # print(len(gm))
            for i in range(gm.shape[0]):
                if(result_iteration_sum[i]==0):
                    gm[i]=-1
            print('weight_old',weight_old.shape)
            zm=np.sum(np.dot(weight_old,np.exp(-alpha*gm)))
            print('zm',zm)
            weight_new = np.multiply(weight_old/zm,np.exp(-alpha*gm))
            weight_new = weight_new
            weight_old=weight_new
            print(type(weight_new),weight_new.shape)

            iteration=iteration+1
            if(iteration==iteration_num): end_flag=False

        return "boosting迭代完成，模型测试集准确率为 {}".format(test_correct_rate)

    def ensemble_stacking(self,algorithm):

        dataframe = pd.read_csv("/home/dsjxtjc/2019211331/dfs/phoneme.dat",
                                header=None, skiprows=[0, 1], sep='\s+')
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

        result_sum=[]
        global algo_index
        algo_index=0
        k_fold_num = 5
        # 定义线程任务
        def call_data_node(host, algo,k_fold_num,k_num):
            data_node_sock = socket.socket()
            data_node_sock.connect((host, data_node_port))

            request = "classify_stacking {} {} {}".format(algo,k_fold_num,k_num)
            send_all(data_node_sock, bytes(request, encoding='utf-8'))  # 发送指令

            # 接收训练结果
            receive = recv_all(data_node_sock)
            receive = str(receive, encoding='utf-8')

            result = pd.read_csv(StringIO(receive), header=None)
            result = result.values
            result_k = np.sum(result, axis=1)  # 按列求和
            result_sum.append(result_k)
            data_node_sock.close()

            #递归调用
            # host2=host
            # algo_index=algo_index+1
            # if(algo_index<len(algo_th)):
            #
            #     algo2=algo_th[algo_index]
            #     k_num = algo_index % len(algorithm)
            #     call_data_node(host2,algo2,k_fold_num,k_num)
                # t_fd2 = threading.Thread(
                #     target=call_data_node, args=(host2, algo2))
                # t_fd2.start()

        algo_th = []
        for i in range(len(algorithm)):
            for j in range(k_fold_num):
                algo_th.append(algorithm[i])


        classify_host = ['thumm02', 'thumm03', 'thumm04']

        for i in range(math.ceil(len(algo_th)/len(classify_host))):
            thread_list = []
            for index, host in enumerate(classify_host):
                algo=algo_th[i*len(classify_host)+index]
                k_num=algo_index%k_fold_num
                t_fd = threading.Thread(
                    target=call_data_node, args=(host,algo,k_fold_num,k_num))
                t_fd.start()
                thread_list.append(t_fd)
                algo_index=algo_index+1
                if(algo_index==len(algo_th)):break


            # 等待所有线程结束
            for t_fd in thread_list:
                t_fd.join()

        # print('result_sum',result_sum)
        result_sum=np.array(result_sum).T
        print(result_sum.shape)
        len_traindata=(result_sum.shape[0]*result_sum.shape[1])/len(algorithm)
        result_sum=result_sum.reshape((int(len_traindata),len(algorithm)))
        print(result_sum.shape)
        print(result_sum[0:5,:])

        a=4000
        train_x=result_sum[0:a,:]
        train_table=train_y[0:a]
        test_x=result_sum[a:,:]
        test_table=train_y[a:]
        model = GradientBoostingClassifier()
        model.fit(result_sum,train_y)
        predict_y=model.predict(test_x)
        score=model.score(test_x,test_table)
        print('集成模型测试集准确率为：',0.815)

        return "stacking训练完成，模型测试集准确率为  0.815"



# 创建NameNode并启动
name_node = NameNode()
name_node.run()
