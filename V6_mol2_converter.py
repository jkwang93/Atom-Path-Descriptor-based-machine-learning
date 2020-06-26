
# _*_ encoding: utf-8 _*_
import copy
import multiprocessing

from rdkit import Chem
from sklearn.preprocessing import MinMaxScaler

__author__ = 'wjk'
__date__ = '2018/8/3 0003 上午 10:38'

import pandas as pd
import numpy as np
import re
import os

# atom_weight = pd.read_csv('atom_weight.csv')
# atom_weight = atom_weight.to_dict(orient='list')


elementdict = {'zNone': 0, 'H': 10, 'C.cat': 64, 'C.ar': 63, 'C.2': 62, 'C.3': 61, 'C.1': 60, 'N.3': 70, 'N.2': 71,
               'N.1': 72, 'N.ar': 73, 'N.am': 74, 'N.pl3': 75, 'N.4': 76, 'O.3': 80, 'O.2': 81, 'O.co2': 82, 'P.3': 150,
               'S.3': 160,
               'S.2': 161, 'S.O': 162, 'S.O2': 163, 'F': 90, 'Cl': 170, 'Br': 350, 'I': 530}


def bond_length(atom_X, atom_Y):
    radius = np.sqrt(np.square(atom_X[0] - atom_Y[0]) + np.square(atom_X[1] - atom_Y[1]) + np.square(
        atom_X[2] - atom_Y[2]))
    return radius


def atom_direction(atom_X, atom_Y):
    direction = np.sqrt(np.square(atom_X[0] - atom_Y[0]) + np.square(atom_X[1] - atom_Y[1]) + np.square(
        atom_X[2] - atom_Y[2]))
    return direction


# 假递归：中心原子center_atom_index， 现原子atom_index， 最大半径max_R， 先半径R，原子坐标表table_coordinate， 原子名称还有电荷表table_atom_name_charge
# 为了避免环状结构，我们在取到对应的atom_index后，先筛选，再转换，故在preorder_tree_atom中不进行转换，直接返回atom_index
def preorder_tree_atom_index(center_atom_index, atom_index, pre_atom, R, max_R, table_re, list_atom_index):
    if R <= max_R:
        R += 1
        fix_pre_atom = pre_atom
        for item in table_re[atom_index]:
            if item != -1 and item != fix_pre_atom:  # 排除none以及atom_index本身
                pre_atom = atom_index
                preorder_tree_atom_index(center_atom_index, item, pre_atom, R, max_R, table_re, list_atom_index)
    else:
        # list_distance.append(atom_direction(table_coordinate[center_atom_index], table_coordinate[atom_index]))  # 距离
        # list_atom_type.append(table_atom_name_charge[atom_index][0])  # atom_type
        list_atom_index.append(atom_index)
    return list_atom_index


'''因为必须判断R=3时，atom_index是否含有R=1,R=2时的index，复杂度太高pre_index

'''


def atom_type_distance(max_R, list_max_length, pre_index, table_re, table_atom_charge, table_coordinate):
    table_atom_type = []
    table_atom_distance = []
    updata_pre_index = []
    for i in range(0, len(table_re)):
        list_atom_index = []
        list_atom_type = []
        list_distance = []
        list_atom_index = preorder_tree_atom_index(i, i, -10, 1, max_R, table_re, list_atom_index)
        list_atom_index = list(set(list_atom_index))  # 去掉重复的atom_index
        updata_list_atom_index = copy.deepcopy(list_atom_index)

        for j in list_atom_index:
            if j in pre_index[i]:
                updata_list_atom_index.remove(j)  # ???????
            if j not in pre_index[i]:
                list_atom_type.append(table_atom_charge[j][0])  # atom_type
                list_distance.append(atom_direction(table_coordinate[i], table_coordinate[j]))  # 距离

        updata_pre_index.append(updata_list_atom_index)
        table_atom_type.append(list_atom_type)
        table_atom_distance.append(list_distance)

    table_atom_type = list(map(lambda l: l + ['zNone'] * (list_max_length - len(l)), table_atom_type))  # 0补齐，使得每个长度都一样
    table_atom_distance = list(
        map(lambda l: l + [0] * (list_max_length - len(l)), table_atom_distance))  # znone补齐，使得每个长度都一样

    return [table_atom_type, table_atom_distance, updata_pre_index]


def read_mol2(mol2_path):
    # DDEC_charges = []
    # m = Chem.MolFromMolFile(sdf_path, removeHs=False)
    # loop over atoms in molecule
    # for at in m.GetAtoms():
    #     aid = at.GetIdx()
    #
    #     # read reference partial charge
    #     q = float(m.GetAtomWithIdx(aid).GetProp('molFileAlias'))
    #     DDEC_charges.append(q)

    with open(mol2_path, "rb") as f:
        x1 = []
        x3 = []
        table_atom_charge = []
        table_coordinate = []
        n = 0
        # lines下标从0开始
        lines = f.readlines()
        for line in lines:
            line = line.decode('utf-8')  # 正则匹配前，python3需要加上此代码
            if re.search('ATOM', line):
                atom_start_num = n + 1
            if re.search('BOND', line):
                atom_end_num = n - 1
                bond_start_num = n + 1
            # if re.search('SUBSTRUCTURE', line):
            #     bond_end_num = n - 1
            n += 1

        bond_end_num = n
        lines_num = atom_end_num - atom_start_num + 1
        # 构建一个二维数组,存入数据
        # a = np.zeros(shape=(lines_num, 3))
        for line_atom in lines[atom_start_num:atom_end_num + 1]:
            line_atom = line_atom.decode('utf-8').replace("\r\n", "").split()  # 正则匹配前，python3需要加上此代码
            # 去掉元素上的数字
            # table_item1 = [line_atom[1][0]] + [line_atom[5]] + [line_atom[8]]
            table_item1 = [line_atom[5]] + [line_atom[8]]
            table_item2 = [line_atom[2]] + [line_atom[3]] + [line_atom[4]]
            table_atom_charge.append(table_item1)
            table_coordinate.append(table_item2)
        table_atom_charge = np.array(table_atom_charge)
        table_coordinate = np.array(table_coordinate).astype(np.float32)

        # 获取元素关系(important是一个矩阵0,1)
        table2 = np.zeros(shape=(lines_num, lines_num), dtype=int)  # 新建一个二维数组
        for line_bond in lines[bond_start_num:bond_end_num + 1]:
            line_bond = line_bond.decode('utf-8').replace("\r\n", "").split()  # 正则匹配前，python3需要加上此代码
            x1 = x1 + [line_bond[1]] + [line_bond[2]]
        x1 = list(map(int, x1))
        t1 = 0
        while t1 <= len(x1) - 1:
            i = x1[t1] - 1
            j = x1[t1 + 1] - 1
            table2[i][j] = 1
            table2[j][i] = 1
            t1 += 2
        table2 = np.array(table2)
        # print(np.sum(table2))

        # 将table2转化为原子序号关系表
        table_re = [[-1 for i in range(4)] for j in range(lines_num)]  # 新建一个二维数组
        for ind1, item1 in enumerate(table2):
            l = 0
            for ind2, item2 in enumerate(item1):
                if item2 == 1:
                    table_re[ind1][l] = ind2
                    l += 1

        '''
        根据table_re记录每个原子跟每个原子连接的原子的坐标：二级原子距离、类型
                # 二级原子距离、类型
        table_re_coordinate = [[0 for i in range(12)] for j in range(lines_num)]  # 新建一个二维数组
        for ind1, item1 in enumerate(table_re):
            l = 0
            for ind2, item2 in enumerate(item1):
                if item2 == -1:
                    l += 3
                else:
                    for ind3, item3 in enumerate(table_re[item2]):
                        if item3 != -1 and item3 != ind1:
                            table_re_coordinate[ind1][l] = atom_direction(table_coordinate[ind1],
                                                                          table_coordinate[item3])
                            l += 1
       '''


        # 一级原子类型+一级原子距离

        pre_index = np.array([-0.1] * len(table_re)).reshape(len(table_re), 1)  # 初始化一个pre_index
        table_1_atom_type, table_1_atom_distance, pre_index1 = atom_type_distance(1, 4, pre_index,
                                                                                  table_re, table_atom_charge,
                                                                                  table_coordinate)

        # 二级原子类型+二级原子距离
        table_2_atom_type, table_2_atom_distance, pre_index2 = atom_type_distance(2, 12, pre_index1,
                                                                                  table_re, table_atom_charge,
                                                                                  table_coordinate)

        # 将pre1跟pre2合并 pre_index_1_and_2 = pre_index1不能直接赋值，参见浅拷贝，深拷贝
        pre_index_1_and_2 = copy.deepcopy(pre_index1)
        for ind, item in enumerate(pre_index2):
            pre_index_1_and_2[ind].extend(item)

        # 三级原子类型+三级原子距离
        table_3_atom_type, table_3_atom_distance, pre_index3 = atom_type_distance(3, 36, pre_index_1_and_2,
                                                                                  table_re, table_atom_charge,
                                                                                  table_coordinate)

        # 将pre1跟pre2合并 pre_index_1_and_2 = pre_index1不能直接赋值，参见浅拷贝，深拷贝
        pre_index_1_and_2_and_3 = copy.deepcopy(pre_index_1_and_2)
        for ind, item in enumerate(pre_index3):
            pre_index_1_and_2_and_3[ind].extend(item)

        # 四级原子类型+四级原子距离
        table_4_atom_type, table_4_atom_distance, pre_index4 = atom_type_distance(4, 36 * 3, pre_index_1_and_2_and_3,
                                                                                  table_re, table_atom_charge,
                                                                                  table_coordinate)

        # 将pre1跟pre2合并 pre_index_1_and_2 = pre_index1不能直接赋值，参见浅拷贝，深拷贝
        pre_index_1_4 = copy.deepcopy(pre_index_1_and_2_and_3)
        for ind, item in enumerate(pre_index4):
            pre_index_1_4[ind].extend(item)

        # 五级原子类型+四级原子距离
        table_5_atom_type, table_5_atom_distance, pre_index5 = atom_type_distance(5, 36 * 3 * 3, pre_index_1_4,
                                                                                  table_re, table_atom_charge,
                                                                                  table_coordinate)

        # 二级原子类型组



        # 获取键的类型7种ar什么的
        table_bond = [['zNone' for i in range(lines_num)] for j in range(lines_num)]  # 新建一个二维数组
        for line_bond in lines[bond_start_num:bond_end_num + 1]:
            line_bond = line_bond.decode('utf-8').replace("\r\n", "").split()  # 正则匹配前，python3需要加上此代码
            line_bond[1] = int(line_bond[1])
            line_bond[2] = int(line_bond[2])
            x3 = x3 + [line_bond[1]] + [line_bond[2]] + [line_bond[3]]
        # x1 = list(map(int, x1))
        t3 = 0
        while t3 <= len(x3) - 1:
            i = x3[t3] - 1
            j = x3[t3 + 1] - 1
            table_bond[i][j] = x3[t3 + 2]
            table_bond[j][i] = x3[t3 + 2]
            t3 += 3
            # print(table_bond)
        table_bond = np.array(table_bond)



        # 创建一个二维数组，存放每个element的bond
        atom_bond = [['zNone' for i in range(4)] for j in range(lines_num)]
        # 取第0列
        for m in range(len(table_bond)):
            l = 0
            for n in range(len(table_bond[0])):
                if table_bond[m][n] != 'zNone':
                    atom_bond[m][l] = table_bond[m][n]
                    l += 1
            # atom_bond[m] = [atom_bond[m]]

        '''
        排序：atom_type按照元素周期表排序，其他的由大到小排序
        '''
        table_1_atom_type = np.array(table_1_atom_type)
        atom_bond = np.array(atom_bond)
        table_1_atom_distance = np.array(table_1_atom_distance)
        table_2_atom_type = np.array(table_2_atom_type)
        table_2_atom_distance = np.array(table_2_atom_distance)
        table_3_atom_type = np.array(table_3_atom_type)
        table_3_atom_distance = np.array(table_3_atom_distance)
        table_4_atom_type = np.array(table_4_atom_type)
        table_4_atom_distance = np.array(table_4_atom_distance)
        table_5_atom_type = np.array(table_5_atom_type)
        table_5_atom_distance = np.array(table_5_atom_distance)

        # table_1_atom_type table_2_atom_type按照元素周期表转换成数字
        for i in range(len(table_1_atom_type[0])):
            table_1_atom_type[:, i] = list(map(lambda x: elementdict[x], table_1_atom_type[:, i]))

        for i in range(len(table_2_atom_type[0])):
            table_2_atom_type[:, i] = list(map(lambda x: elementdict[x], table_2_atom_type[:, i]))

        for i in range(len(table_3_atom_type[0])):
            table_3_atom_type[:, i] = list(map(lambda x: elementdict[x], table_3_atom_type[:, i]))

        for i in range(len(table_4_atom_type[0])):
            table_4_atom_type[:, i] = list(map(lambda x: elementdict[x], table_4_atom_type[:, i]))

        for i in range(len(table_5_atom_type[0])):
            table_5_atom_type[:, i] = list(map(lambda x: elementdict[x], table_5_atom_type[:, i]))

        table_1_atom_type = table_1_atom_type.astype(np.int32)
        table_2_atom_type = table_2_atom_type.astype(np.int32)
        table_3_atom_type = table_3_atom_type.astype(np.int32)
        table_4_atom_type = table_4_atom_type.astype(np.int32)
        table_5_atom_type = table_5_atom_type.astype(np.int32)


        # 数字类型
        atom_bond.sort(axis=1)  # bond_type
        table_1_atom_distance = -np.sort(-table_1_atom_distance, axis=1)  # RA-RD
        table_2_atom_distance = -np.sort(-table_2_atom_distance, axis=1)
        table_3_atom_distance = -np.sort(-table_3_atom_distance, axis=1)
        table_4_atom_distance = -np.sort(-table_4_atom_distance, axis=1)
        table_5_atom_distance = -np.sort(-table_5_atom_distance, axis=1)


        # 字符型
        table_1_atom_type = -np.sort(-table_1_atom_type, axis=1)
        table_2_atom_type = -np.sort(-table_2_atom_type, axis=1)
        table_3_atom_type = -np.sort(-table_3_atom_type, axis=1)
        table_4_atom_type = -np.sort(-table_4_atom_type, axis=1)
        table_5_atom_type = -np.sort(-table_5_atom_type, axis=1)

        table_atom_charge = pd.DataFrame(table_atom_charge)
        table_1_atom_type = pd.DataFrame(table_1_atom_type)
        atom_bond = pd.DataFrame(atom_bond)
        table_1_atom_distance = pd.DataFrame(table_1_atom_distance)
        table_2_atom_distance = pd.DataFrame(table_2_atom_distance)
        table_2_atom_type = pd.DataFrame(table_2_atom_type)

        table_3_atom_type = pd.DataFrame(table_3_atom_type)
        table_3_atom_distance = pd.DataFrame(table_3_atom_distance)

        table_4_atom_type = pd.DataFrame(table_4_atom_type)
        table_4_atom_distance = pd.DataFrame(table_4_atom_distance)

        table_5_atom_type = pd.DataFrame(table_5_atom_type)
        table_5_atom_distance = pd.DataFrame(table_5_atom_distance)

        # 修改表头
        table_atom_charge.columns = ['atom_type', 'charge']


        # atom_type
        table_atom_type = pd.concat(
            [table_atom_charge.atom_type, table_1_atom_type, table_2_atom_type, table_3_atom_type, table_4_atom_type, table_5_atom_type],
            axis=1)

        # atom_bond
        table_bond_type = atom_bond

        # atom_distance
        table_distance = pd.concat(
            [table_1_atom_distance, table_2_atom_distance, table_3_atom_distance, table_4_atom_distance, table_5_atom_distance], axis=1)

        # table_charge = pd.DataFrame(DDEC_charges)
        # table_mol2 = pd.concat(
        #     [table_atom_type, table_bond_type, table_distance], axis=1)
    return [table_atom_type, table_bond_type, table_distance]


def Preprocessing_no_one_hot(table_atom_type, table_bond_type, table_distance, atom_type_name, chunksize):
        element_list = ['H', 'C.cat', 'C.ar', 'C.2', 'C.3', 'C.1', 'N.3', 'N.2', 'N.1', 'N.ar', 'N.am', 'N.pl3',
                        'N.4', 'O.3', 'O.2', 'O.co2', 'P.3', 'S.3', 'S.2', 'S.O', 'S.O2', 'F', 'Cl', 'Br', 'I']

        bond_dict = {'1': 1, '2': 2, '3': 3, 'ar': 4, 'am': 5, 'zNone': 6}

        chunksize = chunksize

        atom_type_all = pd.read_csv(table_atom_type, header=None, chunksize=chunksize)
        bond_type_all = pd.read_csv(table_bond_type, header=None, chunksize=chunksize)
        distance_all = pd.read_csv(table_distance, header=None, chunksize=chunksize)
        # .iloc[0:total_num, ]
        # print(data['atom_type'].describe())
        bond_type = pd.DataFrame()
        charge = pd.DataFrame()
        distance = pd.DataFrame()
        atom_type = pd.DataFrame()
        # 原子种类
        atom_type_num = 0
        #
        # # 成键种类
        bond_type_num = 0

        chucksize_count = 0

        # 选择原子类型
        element_list.remove(atom_type_name)
        delete_kind = element_list


        for a, b, c, d in zip(atom_type_all, bond_type_all, distance_all):

            drop_index = a[a.iloc[:, 0].isin(delete_kind)].index

            x = b.drop(drop_index)
            bond_type = bond_type.append(x)

            x = c.drop(drop_index)
            charge = charge.append(x)

            x = d.drop(drop_index)
            distance = distance.append(x)

            x = a.drop(drop_index)
            atom_type = atom_type.append(x)

            chucksize_count += 1
            # print(chucksize_count)
        # .reset_index(drop=True)
        bond_type = bond_type.reset_index(drop=True)
        charge = charge.reset_index(drop=True)
        distance = distance.reset_index(drop=True)
        atom_type = atom_type.reset_index(drop=True)

        print(atom_type.iloc[:, 0][0])

        # 将数据转为ndarray,准备切片
        # feature = data.drop(['charge'], axis=1)
        # feature = feature.values



        '''
        distance归一化
        '''
        X_Distance = distance.values

        del distance

        # 先转置
        X_Distance = X_Distance.T

        del_X_Distance_ind = []

        # 去掉X_Distance中全为0的列
        for ind, item in enumerate(X_Distance):
            if sum(item) == 0:
                del_X_Distance_ind.append(ind)

        # print(X_Distance.shape)
        X_Distance = np.delete(X_Distance, del_X_Distance_ind, axis=0)

        # print(X_Distance.shape)

        mm = MinMaxScaler()

        for ind, item in enumerate(X_Distance):
            tes = item.reshape(-1, 1)
            mm_data_tes = mm.fit_transform(tes).reshape(-1)
            X_Distance[ind] = mm_data_tes

        # 还原转置
        X_Distance = X_Distance.T

        '''
        atom_type
        '''

        # 提取出原子类型相关的列
        feature_atom_type = atom_type.iloc[:, 1:].values

        del atom_type

        feature_atom_type = feature_atom_type.T

        del_feature_atom_type_ind = []

        # 去掉feature_atom_type中全为0的列
        for ind, item in enumerate(feature_atom_type):
            if sum(item) == 0:
                del_feature_atom_type_ind.append(ind)
        feature_atom_type = np.delete(feature_atom_type, del_feature_atom_type_ind, axis=0)

        feature_atom_type = feature_atom_type.T

        # 提取出跟成键类型相关的列
        # feature_bond_type = np.array(bond_type)
        feature_bond_type = bond_type.values

        for i in range(len(feature_bond_type[0])):
            feature_bond_type[:, i] = list(map(lambda x: bond_dict[x], feature_bond_type[:, i]))

        feature_bond_type = feature_bond_type.astype(np.int32)

        del bond_type

        '''
        将特征拼接在一起feature_atom_type， feature_bond_type， feature_R，feature_D
        '''

        del mm_data_tes, mm,
        return [atom_type_num, bond_type_num, feature_atom_type, feature_bond_type, X_Distance]
