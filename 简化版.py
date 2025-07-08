import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# 生成服从正态分布的随机数
X = np.random.normal(0, 0.05, (32, 1))

# 绘制直方图
plt.hist(X, bins=10, density=True, alpha=0.6, color='g')

# 绘制概率密度函数曲线
mu, sigma = 0, 0.5
x = np.linspace(mu - 3*sigma, mu + 3*sigma, 100)
plt.plot(x, 1/(sigma * np.sqrt(2 * np.pi)) * np.exp(- (x - mu)**2 / (2 * sigma**2)), color='red', alpha=0.8)

plt.title('Normal Distribution')
plt.xlabel('Value')
plt.ylabel('Probability Density')
plt.grid(True)
plt.show()

import numpy as np

# 创建一个 6x6 的矩阵
matrix = np.zeros((6, 6))

matrix[0, 0] = 1
matrix[1, 0] = 1
matrix[2, 0] = 1
matrix[3, 1] = 1
matrix[4, 1] = 1
matrix[5, 2] = 1
matrix[5, 4] = 1

matrix[0, 4] = -0.5
matrix[0, 5] = -0.5
matrix[2, 4] = -0.5
matrix[2, 5] = 0.5
matrix[3, 3] = 0.5
matrix[3, 5] = 0.5
matrix[4, 3] = -0.5
matrix[4, 5] = -1.5

# 求逆矩阵
Fs = np.linalg.inv(matrix)

print("原始矩阵：")  # F逆
print(matrix)
print("\n逆矩阵：")  # Fs
print(Fs)


import numpy as np

# 创建一个 6x6 的矩阵
matrix = np.zeros((6, 6))

matrix[0, 0] = 1
matrix[1, 0] = 1
matrix[2, 0] = 1
matrix[3, 1] = 1
matrix[4, 2] = 1
matrix[5, 2] = 1


matrix[0, 4] = 0.6
matrix[1, 4] = -0.6
matrix[1, 5] = 0.6
matrix[2, 4] = -0.6
matrix[2, 5] = -0.6
matrix[3, 5] = 1.5
matrix[4, 3] = 0.5
matrix[4, 4] = -0.5
matrix[5, 3] = -0.5
matrix[5, 4] = -1.5
# 求逆矩阵
Fs2 = np.linalg.inv(matrix)

print("原始矩阵：")  # F逆
print(matrix)
print("\n逆矩阵：")  # Fs
print(Fs2)


phi_1 = np.eye((6))
phi_1[1,5] = 2
phi_1[2,4] = -2
print(phi_1)

phi_2 = np.eye((6))
phi_2[1,5] = 4
phi_2[2,4] = -4
print(phi_2)

N = 50
for i in range(N):
    f_1 = np.random.normal(0, 0.005, (6,1))
    #f_1[5]=np.random.normal(0,0.001)
    # print(f_1)
    f_2 = np.random.normal(0, 0.005, (6,1))
    #f_2[5]=np.random.normal(0,0.001)
    # print(f_2)
    f_3 = np.random.normal(0, 0.005, (6,1))
    #f_3[5]=np.random.normal(0,0.001)
    # print(f_3)

    X1 = Fs @ f_1 # 矩阵的乘法
    X21 = phi_1 @ X1
    X31 = phi_2 @ X1

    X22 = Fs @ f_2
    X2 = X21 + X22

    X32 = phi_1 @ X22
    X33 = Fs @ f_3
    X3 = X31 + X32 + X33

    X1 = X1.reshape(6)
    X2 = X2.reshape(6)
    X3 = X3.reshape(6)
    X123 = np.stack([X1, X2, X3], axis=0)
    if i == 0:
        X123 = X123[np.newaxis, :]
        contents = X123
    else:
        # print(contents.shape)
        # print(X123.shape)
        X123 = X123[np.newaxis, :]
        contents = np.concatenate([contents, X123], axis=0)

print(contents.shape)
print(contents)


# N = 50
# for i in range(N):
#     f_1 = np.random.normal(0, 0.002, (6,1))
#     #f_1[5]=np.random.normal(0,0.001)
#     # print(f_1)
#     f_2 = np.random.normal(0, 0.002, (6,1))
#     #f_2[5]=np.random.normal(0,0.001)
#     # print(f_2)
#     f_3 = np.random.normal(0, 0.002, (6,1))
#     #f_3[5]=np.random.normal(0,0.001)
#     # print(f_3)
# 
#     X1 = Fs2 @ f_1
#     X21 = phi_1 @ X1
#     X31 = phi_2 @ X1
# 
#     X22 = Fs2 @ f_2
#     X2 = X21 + X22
# 
#     X32 = phi_1 @ X22
#     X33 = Fs2 @ f_3
#     X3 = X31 + X32 + X33
# 
#     X1 = X1.reshape(6)
#     X2 = X2.reshape(6)
#     X3 = X3.reshape(6)
#     X123 = np.stack([X1, X2, X3], axis=0)
#     if i == 0:
#         X123 = X123[np.newaxis, :]
#         contents = X123
#     else:
#         # print(contents.shape)
#         # print(X123.shape)
#         X123 = X123[np.newaxis, :]
#         contents = np.concatenate([contents, X123], axis=0)
# 
# print(contents.shape)
# print(contents)

# contents = np.random.randn(50, 3, 6)

x_1=contents[:,0,0]
y_1=contents[:,0,1]
z_1=contents[:,0,2]
a_1=contents[:,0,3]
b_1=contents[:,0,4]
g_1=contents[:,0,5]

x_2=contents[:,1,0]
y_2=contents[:,1,1]
z_2=contents[:,1,2]
a_2=contents[:,1,3]
b_2=contents[:,1,4]
g_2=contents[:,1,5]

x_3=contents[:,2,0]
y_3=contents[:,2,1]
z_3=contents[:,2,2]
a_3=contents[:,2,3]
b_3=contents[:,2,4]
g_3=contents[:,2,5]

print(x_1)

aa = [0] * 50
bb = [1] * 50
cc = [2] * 50
dd = [3] * 50
ee = [4] * 50
ff = [5] * 50
gg = [6] * 50
hh = [7] * 50
ii = [8] * 50

import matplotlib.pyplot as plt

# 准备数据
#x = [1, 1, 1, 1, 1, 1, 1]  # 横坐标，假设所有点都在 x = 1 的位置
#y = [10, 15, 20, 25, 30, 35, 40]  # 纵坐标

xz = [0, 1, 2, 3, 4, 5, 6, 7, 8]  # 横坐标
labels = [r'$Δx_1$', r'$Δy_1$', r'$Δz_1$', r'$Δx_2$', r'$Δy_2$', r'$Δz_2$',r'$Δx_3$', r'$Δy_3$', r'$Δz_3$']  # 自定义横坐标标签
# labels = [r'$Δ\alpha_1$', r'$Δ\beta_1$', r'$Δ\gamma_1$', r'$Δ\alpha_2$', r'$Δ\beta_2$', r'$Δ\gamma_2$', r'$Δ\alpha_3$', r'$Δ\beta_3$', r'$Δ\gamma_3$']  # 自定义横坐标标签

plt.rcParams['font.sans-serif'] = ['SimHei']  # 使用黑体
plt.rcParams['axes.unicode_minus'] = False  # 负号显示正常

plt.figure(1, figsize=(12, 9))  # 设置图形大小

# 整合数据用于小提琴图
data = {
    'Value': np.concatenate([x_1, y_1, z_1, x_2, y_2, z_2, x_3, y_3, z_3]),
    'Category': (['x_1'] * 50 + ['y_1'] * 50 + ['z_1'] * 50 +
                 ['x_2'] * 50 + ['y_2'] * 50 + ['z_2'] * 50 +
                 ['x_3'] * 50 + ['y_3'] * 50 + ['z_3'] * 50)
}

custom_palette = ['#fcbcbb', '#f7f0bc', '#bbdff7']
edges = ['#AE0D1D', '#b5a701', '#065AB0']
violin_parts = sns.violinplot(x='Category', y='Value', data=data, palette=custom_palette,
               inner=None, alpha=0.5)

for i, violin in enumerate(violin_parts.collections):
    violin.set_edgecolor(edges[i % 3])  # 设置边线颜f7f0bc色
    violin.set_linewidth(2)  # 设置边线宽度

# 设置散点大小
point_size = 20  # 设置散点的大小

plt.scatter(aa, x_1, s=point_size,c='#AE0D1D')  # 绘制散点图并设置大小
plt.scatter(bb, y_1, s=point_size,c='#b5a701')  # 绘制散点图并设置大小
plt.scatter(cc, z_1, s=point_size,c='#065AB0')  # 绘制散点图并设置大小
plt.scatter(dd, x_2, s=point_size,c='#AE0D1D')  # 绘制散点图并设置大小
plt.scatter(ee, y_2, s=point_size,c='#b5a701')  # 绘制散点图并设置大小
plt.scatter(ff, z_2, s=point_size,c='#065AB0')  # 绘制散点图并设置大小
plt.scatter(gg, x_3, s=point_size,c='#AE0D1D')  # 绘制散点图并设置大小
plt.scatter(hh, y_3, s=point_size,c='#b5a701')  # 绘制散点图并设置大小
plt.scatter(ii, z_3, s=point_size,c='#065AB0')  # 绘制散点图并设置大小

# 替换横坐标的数值为文本标签
plt.xticks(xz, labels)

red_dot = plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='#AE0D1D', label=r'舱段$\it{i}$在x方向的平动误差($\it{Δx_i}$)', markersize=10)
green_dot = plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='#b5a701', label=r'舱段$\it{i}$在y方向的平动误差($Δy_i$)', markersize=10)
blue_dot = plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='#065AB0', label=r'舱段$\it{i}$在z方向的平动误差($Δz_i$)', markersize=10)


# plt.legend(handles=[line1, line2, red_dot], title='函数', fontsize=12)
plt.legend(handles = [red_dot, green_dot, blue_dot], fontsize=20)

# 添加标题和标签
#plt.title('Reference Point 1')  # 添加标题
plt.xlabel('图(c)舱段i的平动误差',fontsize=22)  # 添加横坐标标签
plt.ylabel('平动误差的数值/米',fontsize=22)  # 添加纵坐标标签
plt.ylim([-0.3, 0.3])

plt.tick_params(axis='both', labelsize=20)  # 设置坐标轴上数值的字体大小为12

# 显示图形
# 添加网格线
plt.show()

# 绘制散点图
plt.figure(2, figsize=(12, 9))  # 设置图形大小

# 整合数据用于小提琴图
data = {
    'Value': np.concatenate([a_1, b_1, g_1, a_2, b_2, g_2, a_3, b_3, g_3]),
    'Category': (['x_1'] * 50 + ['y_1'] * 50 + ['z_1'] * 50 +
                 ['x_2'] * 50 + ['y_2'] * 50 + ['z_2'] * 50 +
                 ['x_3'] * 50 + ['y_3'] * 50 + ['z_3'] * 50)
}

custom_palette_2 = ['#fcd9bb', '#bffbb8', '#d0bbfc']
violin_parts = sns.violinplot(x='Category', y='Value', data=data, palette=custom_palette_2, inner=None, color='lightgray', alpha=0.5)

egdes_2 = ["#ab4e09", "#05B103", "#2f059b"]

# 散点图
# 设置散点大小
for i, violin in enumerate(violin_parts.collections):
    violin.set_edgecolor(egdes_2[i % 3])  # 设置边线颜色
    violin.set_linewidth(2)  # 设置边线宽度
point_size = 20  # 设置散点的大小


plt.scatter(aa, a_1, s=point_size,c='#ab4e09')  # 绘制散点图并设置大小
plt.scatter(bb, b_1, s=point_size,c='#05B103')  # 绘制散点图并设置大小
plt.scatter(cc, g_1, s=point_size,c='#2f059b')  # 绘制散点图并设置大小
plt.scatter(dd, a_2, s=point_size,c='#ab4e09')  # 绘制散点图并设置大小
plt.scatter(ee, b_2, s=point_size,c='#05B103')  # 绘制散点图并设置大小
plt.scatter(ff, g_2, s=point_size,c='#2f059b')  # 绘制散点图并设置大小
plt.scatter(gg, a_3, s=point_size,c='#ab4e09')  # 绘制散点图并设置大小
plt.scatter(hh, b_3, s=point_size,c='#05B103')  # 绘制散点图并设置大小
plt.scatter(ii, g_3, s=point_size,c='#2f059b')  # 绘制散点图并设置大小

# 替换横坐标的数值为文本标签
labels = [r'$Δ\alpha_1$', r'$Δ\beta_1$', r'$Δ\gamma_1$', r'$Δ\alpha_2$', r'$Δ\beta_2$', r'$Δ\gamma_2$', r'$Δ\alpha_3$', r'$Δ\beta_3$', r'$Δ\gamma_3$']  # 自定义横坐标标签
plt.xticks(xz, labels)

orange_dot = plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='#AC5216', label=r'舱段$\it{i}$绕x轴的转动误差($\it{Δ\alpha_i}$)', markersize=10)
yellow_dot = plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='#05B103', label=r'舱段$\it{i}$绕y轴的转动误差($Δ\beta_i$)', markersize=10)
purple_dot = plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='#2800A3', label=r'舱段$\it{i}$绕z轴的转动误差($Δ\gamma_i$)', markersize=10)


# plt.legend(handles=[line1, line2, red_dot], title='函数', fontsize=12)
plt.legend(handles = [orange_dot, yellow_dot, purple_dot], fontsize=20)

#plt.title('Reference Point 1')  # 添加标题# 添加标题和标签

plt.xlabel('图(d)舱段i的转动误差',fontsize=22)  # 添加横坐标标签
plt.ylabel('转动误差的数值/弧度',fontsize=22)  # 添加纵坐标标签
plt.ylim([-0.1, 0.1])

plt.tick_params(axis='both', labelsize=20)  # 设置坐标轴上数值的字体大小为12

# 显示图形
# 添加网格线
plt.show()