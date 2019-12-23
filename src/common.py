import socket
import struct
dfs_blk_size = 128 * 1024 * 1024

# NameNode和DataNode数据存放位置
name_node_dir = "./dfs/name"
data_node_dir = "./dfs/data"

data_node_port = 11331  # DataNode程序监听端口
name_node_port = 21331  # NameNode监听端口

# 集群中的主机列表
# ['thumm01', 'thumm02', 'thumm03', 'thumm04', 'thumm05']
host_list = ['thumm02', 'thumm03', 'thumm04']
classify_host = ['thumm02', 'thumm03','thumm04']
name_node_host = "thumm01"

# 每个数据块被保存的次数
dfs_replication = 2

# 数据接收的缓冲区大小
BUF_SIZE = 4096

# 这个用于设定最大连接数
MAXLOG = 100


def send_all(sock_fd: socket.socket, data: bytes):
    # 获取数据的大小
    data_size = len(data)
    # 告诉接收端数据大小
    sock_fd.send(struct.pack("q", data_size))

    # 使用send_all发送所有数据
    res = sock_fd.sendall(data)

    return res


def recv_all(sock_fd: socket.socket):
    # 获取待接收的数据的大小
    data_size = struct.unpack("q", sock_fd.recv(8))[0]
    recv_len = 0   # 记录已接收数据的大小
    recv_buf = []  # 用于存放已接收的数据
    while recv_len < data_size:
        # 每次接收一部分数据
        part_data = sock_fd.recv(min(BUF_SIZE, data_size-recv_len))
        # 将数据存放到缓冲区
        recv_buf.append(part_data)
        # 更新已接收数据的长度
        recv_len += len(part_data)

    # 将接收的数据进行拼接
    return bytes().join(recv_buf)
