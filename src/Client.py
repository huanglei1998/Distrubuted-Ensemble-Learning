import sys
import os
import socket
import time
import copy
import math

from io import StringIO

import pandas as pd

from common import *
import threading


class Client:
    def __init__(self):
        self.name_node_sock = socket.socket()
        self.name_node_sock.connect((name_node_host, name_node_port))

    def __del__(self):
        self.name_node_sock.close()

    def ls(self, dfs_path):
        # 向NameNode发送请求，查看dfs_path下文件或者文件夹信息
        try:
            request = "ls {}".format(dfs_path)
            send_all(self.name_node_sock, bytes(request, encoding='utf-8'))
            response_msg = str(recv_all(self.name_node_sock), encoding='utf-8')
            print(response_msg)
        except Exception as e:
            print(e)
        finally:
            pass

    def copyFromLocal(self, local_path, dfs_path):
        file_size = os.path.getsize(local_path)
        print("File size: {}".format(file_size))

        request = "new_fat_item {} {}".format(dfs_path, file_size)
        print("Request: {}".format(request))

        # 从NameNode获取一张FAT表
        send_all(self.name_node_sock, bytes(request, encoding='utf-8'))
        fat_pd = recv_all(self.name_node_sock)

        # 打印FAT表，并使用pandas读取
        fat_pd = str(fat_pd, encoding='utf-8')
        print("Fat: \n{}".format(fat_pd))
        fat = pd.read_csv(StringIO(fat_pd))

        # 用于保存线程描述符
        thread_fds = []

        # 定义线程任务
        def send_block(row, data):
            data_node_sock = socket.socket()
            data_node_sock.connect((row['host_name'], data_node_port))
            blk_path = dfs_path + ".blk{}".format(row['blk_no'])

            request = "store {}".format(blk_path)
            send_all(data_node_sock, bytes(request, encoding='utf-8'))
            send_all(data_node_sock, data)
            data_node_sock.close()

        # 根据FAT表逐个向目标DataNode发送数据块
        with open(local_path, "rb") as fp:
            for idx, row in fat.iterrows():
                data = fp.read(int(row['blk_size']))

                # 创建线程并启动
                t_fd = threading.Thread(
                    target=send_block, args=(copy.deepcopy(row), copy.deepcopy(data)))
                t_fd.start()
                t_fd.join()
                # 将线程描述符保存在列表中
                thread_fds.append(t_fd)

        # 等待所有线程结束
        # for t_fd in thread_fds:
        #     t_fd.join()

    def copyToLocal(self, dfs_path, local_path):
        request = "get_fat_item {}".format(dfs_path)
        print("Request: {}".format(request))

        # 从NameNode获取一张FAT表
        send_all(self.name_node_sock, bytes(request, encoding='utf-8'))
        fat_pd = recv_all(self.name_node_sock)

        # 打印FAT表，并使用pandas读取
        fat_pd = str(fat_pd, encoding='utf-8')
        print("Fat: \n{}".format(fat_pd))
        fat = pd.read_csv(StringIO(fat_pd))

        def recv_block(row, buf: list):
            data_node_sock = socket.socket()
            data_node_sock.connect((row['host_name'], data_node_port))
            blk_path = dfs_path + ".blk{}".format(row['blk_no'])

            request = "load {}".format(blk_path)
            send_all(data_node_sock, bytes(request, encoding='utf-8'))
            data = recv_all(data_node_sock)
            buf.append(data)

            data_node_sock.close()

        buf_list = []
        thread_fds = []
        # 根据FAT表逐个从目标DataNode请求数据块，写入到本地文件中
        for idx, row in fat.iterrows():
            buf = []
            t_fd = threading.Thread(
                target=recv_block, args=(row, buf))
            t_fd.start()
            thread_fds.append(t_fd)
            buf_list.append(buf)

        for t_fd in thread_fds:
            t_fd.join()

        with open(local_path, "wb") as fp:
            for buf in buf_list:
                fp.write(buf[0])

    def rm(self, dfs_path):
        request = "rm_fat_item {}".format(dfs_path)
        print("Request: {}".format(request))

        # 从NameNode获取改文件的FAT表，获取后删除
        send_all(self.name_node_sock, bytes(request, encoding='utf-8'))
        fat_pd = recv_all(self.name_node_sock)

        # 打印FAT表，并使用pandas读取
        fat_pd = str(fat_pd, encoding='utf-8')
        print("Fat: \n{}".format(fat_pd))
        fat = pd.read_csv(StringIO(fat_pd))

        # 根据FAT表逐个告诉目标DataNode删除对应数据块
        for idx, row in fat.iterrows():
            data_node_sock = socket.socket()
            data_node_sock.connect((row['host_name'], data_node_port))
            blk_path = dfs_path + ".blk{}".format(row['blk_no'])

            request = "rm {}".format(blk_path)
            send_all(data_node_sock, bytes(request, encoding='utf-8'))
            response_msg = recv_all(data_node_sock)
            print(response_msg)

            data_node_sock.close()

    def format(self):
        request = "format"
        print(request)
        send_all(self.name_node_sock, bytes(request, encoding='utf-8'))
        response = str(recv_all(self.name_node_sock), encoding='utf-8')
        print(response)

        for host in host_list:
            data_node_sock = socket.socket()
            data_node_sock.connect((host, data_node_port))
            send_all(data_node_sock, bytes("format", encoding='utf-8'))
            response = str(recv_all(data_node_sock), encoding='utf-8')
            print(response)
            data_node_sock.close()

    def ensemble_boosting(self,algorithm):
        if(len(algorithm)==2):
            request = "ensemble_boosting {} {}".format(algorithm[0],algorithm[1])
        else:
            request = "ensemble_boosting {} {} {}".format(algorithm[0], algorithm[1],algorithm[2])
        print("Request: {}".format(request))
        send_all(self.name_node_sock, bytes(request, encoding='utf-8'))
        response = str(recv_all(self.name_node_sock), encoding='utf-8')
        print(response)

    def ensemble_stacking(self,algorithm):
        if(len(algorithm)==2):
            request = "ensemble_stacking {} {}".format(algorithm[0],algorithm[1])
        else:
            request = "ensemble_stacking {} {} {}".format(algorithm[0], algorithm[1],algorithm[2])
        print("Request: {}".format(request))
        send_all(self.name_node_sock, bytes(request, encoding='utf-8'))
        response = str(recv_all(self.name_node_sock), encoding='utf-8')
        print(response)



# 解析命令行参数并执行对于的命令

argv = sys.argv
argc = len(argv) - 1

client = Client()

cmd = argv[1]
if cmd == '-ls':
    if argc == 2:
        dfs_path = argv[2]
        client.ls(dfs_path)
    else:
        print("Usage: python client.py -ls <dfs_path>")
elif cmd == "-rm":
    if argc == 2:
        dfs_path = argv[2]
        client.rm(dfs_path)
    else:
        print("Usage: python client.py -rm <dfs_path>")
elif cmd == "-copyFromLocal":
    if argc == 3:
        local_path = argv[2]
        dfs_path = argv[3]
        client.copyFromLocal(local_path, dfs_path)
    else:
        print("Usage: python client.py -copyFromLocal <local_path> <dfs_path>")
elif cmd == "-copyToLocal":
    if argc == 3:
        dfs_path = argv[2]
        local_path = argv[3]
        client.copyToLocal(dfs_path, local_path)
    else:
        print("Usage: python client.py -copyFromLocal <dfs_path> <local_path>")
elif cmd == "-format":
    client.format()
elif cmd == "-ensemble_boosting":
    if argc == 3:
        algorithm1= argv[2]
        algorithm2=argv[3]
        algorithm=[algorithm1, algorithm2]
        client.ensemble_boosting(algorithm)
    else:
        algorithm1= argv[2]
        algorithm2=argv[3]
        algorithm3 = argv[4]
        algorithm=[algorithm1, algorithm2,algorithm3]
        client.ensemble_boosting(algorithm)
elif cmd == "-ensemble_stacking":
    if argc == 3:
        algorithm1= argv[2]
        algorithm2=argv[3]
        algorithm=[algorithm1, algorithm2]
        client.ensemble_stacking(algorithm)
    else:
        algorithm1= argv[2]
        algorithm2=argv[3]
        algorithm3 = argv[4]
        algorithm=[algorithm1, algorithm2,algorithm3]
        client.ensemble_stacking(algorithm)

else:
    print("Undefined command: {}".format(cmd))
    print("Usage: python client.py <-ls | -copyFromLocal | -copyToLocal | -rm | -format> other_arguments")
