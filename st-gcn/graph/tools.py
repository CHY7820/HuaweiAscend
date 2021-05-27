import numpy as np


# 将边的连接关系转化为矩阵
def edge2mat(link, num_node):
    A = np.zeros((num_node, num_node))
    for i, j in link:
        A[j, i] = 1  # 注意这里i j的位置换了，当前的连接关系是基于i点的。j作行，i作列。最后卷积乘是刚好对应起来的
    return A


# 除以每列的和。因为不能因为有的关节连得关节多，就增加他在图里的权重。
def normalize_digraph(A):
    Dl = np.sum(A, 0)
    h, w = A.shape
    Dn = np.zeros((w, w))
    for i in range(w):
        if Dl[i] > 0:
            Dn[i, i] = Dl[i]**(-1)
    AD = np.dot(A, Dn)
    return AD


def get_spatial_graph(num_node, self_link, inward, outward):
    I = edge2mat(self_link, num_node)  #自身连接，肯定就一个，不需要归一化
    In = normalize_digraph(edge2mat(inward, num_node))
    Out = normalize_digraph(edge2mat(outward, num_node))
    A = np.stack((I, In, Out))
    return A  #A是三个图，依次是自身图、内图、外图。A：3*num_node*num_node
