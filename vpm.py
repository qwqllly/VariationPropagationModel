import numpy as np
import math

########################输入数据################################

# 舱段2上：定位器P1-P5在局部坐标系下的坐标，由定位器测出
x1, y1, z1 = 000, 000, 000
x2, y2, z2 = 400, 400, 100
x3, y3, z3 = 800, 100, 700
x4, y4, z4 = 800, 700, 700
x5, y5, z5 =  50, 550, 100

# 舱段1上：定位器P6-P10在局部坐标系下的坐标，由定位器测出
x1_1, y1_1, z1_1 = 000, 000, 000
x2_1, y2_1, z2_1 = 200, 350,  90
x3_1, y3_1, z3_1 = 750, 90,  680
x4_1, y4_1, z4_1 = 780, 680, 660
x5_1, y5_1, z5_1 = -50, 600,  60

# 舱段3上：定位器P11-P15在局部坐标系下的坐标，由定位器测出
x1_3, y1_3, z1_3 = 000, 000, 000
x2_3, y2_3, z2_3 = 300, 380,  85
x3_3, y3_3, z3_3 = 760,  86, 650
x4_3, y4_3, z4_3 = 780, 650, 680
x5_3, y5_3, z5_3 = -30, 580,  80

# 创建全局坐标系
x_global_vector = np.array([1, 0, 0])
y_global_vector = np.array([0, 1, 0])
z_global_vector = np.array([0, 0, 1])

# 在舱段2上创建局部坐标系
# 创建全局坐标系下主平面的法向量 (nx, ny, nz)
nx,ny,nz = 2/3,1/3,2/3

##### 在舱段1上创建局部坐标系 #####
# 创建全局坐标系下主平面的法向量 (nx, ny, nz)
nx_1, ny_1, nz_1 = 2 / math.sqrt(6), 1 / math.sqrt(6), 1 / math.sqrt(6)

# 输入新数据，此处是第一个舱段五个定位点在第二个舱段局部坐标系下的坐标
x1_root_1, y1_root_1, z1_root_1 = -1003, 2.5, -0.8
x2_root_1, y2_root_1, z2_root_1 = -796, 349, 92
x3_root_1, y3_root_1, z3_root_1 = -52, 88, 677
x4_root_1, y4_root_1, z4_root_1 = -21, 676, 663
x5_root_1, y5_root_1, z5_root_1 =  -1048, 602, 59

# Q中运用的坐标值
x_Omega_1, y_Omega_1, z_Omega_1 = 1002, 1, 2

##### 在舱段3上创建局部坐标系 #####
# 创建全局坐标系下主平面的法向量 (nx, ny, nz)
nx_3, ny_3, nz_3 = 1 / 2, 1 / 2, math.sqrt(2) / 2

# 输入新数据，此处是第三个舱段五个定位点在第二个舱段局部坐标系下的坐标
x1_root_3, y1_root_3, z1_root_3 = 998, 0.8, -0.7
x2_root_3, y2_root_3, z2_root_3 = 1304, 382, 88
x3_root_3, y3_root_3, z3_root_3 = 1758, 84, 652
x4_root_3, y4_root_3, z4_root_3 = 1782, 649, 682
x5_root_3, y5_root_3, z5_root_3 = 976, 578, 82

# Q中运用的坐标值，
x_Omega2, y_Omega2, z_Omega2 = -998, 1, 0.5
x_Omega3, y_Omega3, z_Omega3 = -2001, 2, 1



########################函数部分################################

# 用于创建坐标系转换矩阵中的矩阵扩展
def expand_matrix(original_matrix,n):
    # 获取原矩阵的形状
    original_shape = original_matrix.shape
    rows, cols = original_shape

    # 创建一个6x6的零矩阵
    expanded_matrix = np.zeros((n, n))

    # 将原矩阵放置在左上和右下角
    expanded_matrix[:rows, :cols] = original_matrix
    expanded_matrix[-rows:, -cols:] = original_matrix

    return expanded_matrix


def generate_Fs_bar(nx, ny, nz, x1, y1, z1, x2, y2, z2, x3, y3, x4, y4, x5, y5):
    global x_global_vector, y_global_vector, z_global_vector
    
    # 创建局部坐标系
    z_local_vector = np.array([nx, ny, nz])
    x_local_vector = np.cross(z_global_vector, z_local_vector)
    y_local_vector = np.cross(z_local_vector, x_local_vector)

    # 局部坐标系和全局坐标系的变换矩阵
    phi_matrix = np.vstack((x_local_vector, y_local_vector, z_local_vector))

    # 扩展为6*6矩阵，平动+旋转
    phi_bar = expand_matrix(phi_matrix,6)

    # 计算D35 D36 D37
    denominator_1 = x5*y3 - x5*y4 + x4*y5 - x3*y5 + x3*y4 -x4*y3
    D35 = (x4*y5 - x5*y4)/ denominator_1
    D36 = (x5*y3 - x3*y5)/ denominator_1
    D37 = (x3*y4 - x4*y3)/ denominator_1

    # 计算D45 D46 D47
    denominator_2 = x4*y5 -x5*y4
    D45 = (x5-x4) * D35 / denominator_2
    D46 = ((x5-x4) * D36 - x5) / denominator_2
    D47 = ((x5-x4) * D37 + x4) / denominator_2

    # 计算D55 D56 D57
    denominator_3 = y4*x3 -y3*x4
    D55 = ((y3-y4) * D35 + y4) / denominator_3
    D56 = ((y3-y4) * D36 - y3) / denominator_3
    D57 = (y3-y4) * D37 / denominator_3

    # P1与P2的距离 L
    L = math.sqrt((x2-x1)*(x2-x1) + (y2-y1)*(y2-y1) + (z2-z1)*(z2-z1))

    # 计算λ：P1P2与x轴的夹角
    # 向量OP和x轴单位向量
    OP2 = np.array([x2, y2, z2])

    # 计算点积和向量模
    dot_product = np.dot(OP2, x_local_vector)
    magnitude_OP2 = np.linalg.norm(OP2)

    # 计算夹角的余弦值
    cos_theta = dot_product / magnitude_OP2

    # 计算夹角（弧度制）
    lamda = np.arccos(cos_theta)
    # 利用定位器的坐标构建Fs 和Fs_bar
    Fs = np.zeros((6, 7))

    # 设置(0, 0)和(1, 1)的值为1
    Fs[0, 0] = 1
    Fs[1, 1] = 1

    # 将计算得到的值放入矩阵中指定位置(2, 4)到(4, 6)
    Fs[2, 4] = D35
    Fs[2, 5] = D36
    Fs[2, 6] = D37
    Fs[3, 4] = D45
    Fs[3, 5] = D46
    Fs[3, 6] = D47
    Fs[4, 4] = D55
    Fs[4, 5] = D56
    Fs[4, 6] = D57

    Fs[5, 0] = np.sin(lamda)/L
    Fs[5, 1] = -1 * np.cos(lamda)/L
    Fs[5, 2] = -1 * np.sin(lamda)/L
    Fs[5, 3] = np.cos(lamda)/L

    Fs_bar = np.dot(np.linalg.inv(phi_bar) , Fs)

    return Fs_bar


def generate_Qi(x_1, x_2, x_3, x_4, x_5, y_1, y_2, y_3, y_4, y_5, z_1, z_2, z_3, z_4, z_5):
    # 创建列表以存储坐标值
    x_values = [x_1, x_2, x_3, x_4, x_5]
    y_values = [y_1, y_2, y_3, y_4, y_5]
    z_values = [z_1, z_2, z_3, z_4, z_5]

    # 创建一个空列表以存储矩阵
    matrices = []

    # 创建循环来生成矩阵
    for i, (xi, yi, zi) in enumerate(zip(x_values, y_values, z_values), 1):
        # 创建3x3的矩阵，并初始化为0
        matrix = np.zeros((3, 3))

        # 设定特定位置的值
        matrix[0, 1] = zi
        matrix[0, 2] = -1 * yi
        matrix[1, 0] = -1 * zi
        matrix[1, 2] = xi
        matrix[2, 0] = yi
        matrix[2, 1] = -1 * xi

        matrices.append(matrix)
        
    # 创建一个空的15x30矩阵
    Qi = np.zeros((15, 30))

    # 在对角线位置放置扩展后的矩阵
    for i, matrix in enumerate(matrices):
        row_start = i * 3
        row_end = row_start + 3
        col_start = i * 6
        col_end = col_start + 6

        # 生成单位矩阵，并使用堆叠函数组合成3x6的矩阵
        unit_matrix = np.eye(3)
        extended_matrix = np.hstack((unit_matrix, matrix))

        # 将扩展后的矩阵放置在对角线位置
        Qi[row_start:row_end, col_start:col_end] = extended_matrix
    return Qi


# 创建对接的J
def generate_J(x_Omega, y_Omega, z_Omega):
    J1 = np.zeros((6,25))
    J1[1,1], J1[2,2], J1[4,4], J1[5,5],J1[1,7], J1[2,8], J1[4,10], J1[5,11], J1[1,13], J1[2,14], J1[4,16], J1[5,17] = 1, 1, 1, 1, 1, 1, 1, 1, -1, -1, -1, -1
    J1[1,3], J1[1,5], J1[2,3], J1[2,4] = -1*z_Omega, x_Omega, y_Omega, -1*x_Omega
    J1[:,18:25] = Fs_bar2 
    
    return J1

def generate_MPs1_i_Pr(x_Omega, y_Omega, z_Omega):
    # 构建 M(Ps1(i),Pr)
    MPs1_i_Pr = np.empty((6, 6))
    # 创建一个3x3的单位矩阵
    top_left = np.eye(3)
    bottom_right = np.eye(3)
    # 创建一个3x3的零矩阵
    bottom_left = np.zeros((3, 3))
    # 指定的坐标值
    specified_values = np.array([
        [0, z_Omega, -1*y_Omega],
        [-1*z_Omega, 0, x_Omega],
        [y_Omega, -1*x_Omega, 0]
    ])
    # 将左上、右下和右上角的值填入矩阵
    MPs1_i_Pr[:3, :3] = top_left
    MPs1_i_Pr[:3, 3:] = specified_values
    MPs1_i_Pr[3:, :3] = bottom_left
    MPs1_i_Pr[3:, 3:] = bottom_right
    
    return MPs1_i_Pr

########################程序部分################################

##### 第2个舱段空装 #####
Fs_bar1 = generate_Fs_bar(nx, ny, nz, x1, y1, z1, x2, y2, z2, x3, y3, x4, y4, x5, y5)

W0_bar = np.zeros((6,13))
W0_bar[:,:7] = Fs_bar1

# 矩阵U1：站1中与局部误差源的偏差
U1_matrix = np.random.uniform(0, 0.001, (13,1))

# 矩阵B1
B1_matrix = np.zeros((18, 13))

# 将原始6x7的矩阵放置在新矩阵的上部
B1_matrix[:6, :] = W0_bar

# 舱段2空装后的6自由度误差
X1 = np.dot(B1_matrix,U1_matrix)

##### 舱段2与舱段1装配 #####
# 创建矩阵F2
F2 = np.zeros((7, 15), dtype=int)

# 在指定位置放置1
positions = [(0, 0), (1, 1), (2, 3), (3, 4), (4, 8), (5, 11), (6, 14)]

for pos in positions:
    row, col = pos
    F2[row, col] = 1

Q2 = generate_Qi(x1_root_1,x2_root_1,x3_root_1,x4_root_1,x5_root_1, 
                y1_root_1,y2_root_1,y3_root_1,y4_root_1,y5_root_1,
                z1_root_1,z2_root_1,z3_root_1,z4_root_1,z5_root_1)

# 创建矩阵S2
unit_matrix = np.eye(6)

# 堆叠5个单位矩阵以创建30x6的矩阵
S2 = np.vstack([unit_matrix for _ in range(5)])

# 计算G2
G2 = F2@Q2@S2

#计算舱段1的Fs_bar2
Fs_bar2 = generate_Fs_bar(nx_1, ny_1, nz_1, x1_1, y1_1, z1_1, x2_1, y2_1, z2_1, x3_1, y3_1, x4_1, y4_1, x5_1, y5_1)

# 构造F1(2)
# 将左边的单位矩阵和右边的零矩阵拼接在一起
F1_2 = np.zeros((6,18))
F1_2[:,:6] = np.eye(6)

MPs1_2_Pr = generate_MPs1_i_Pr(x_Omega_1, y_Omega_1, z_Omega_1)

# 构建Ms(2)(2)
Ms_2 = (np.eye(6) - MPs1_2_Pr) @ Fs_bar2 @ G2 @ F1_2

F4 = np.zeros((25, 6))
F4[:6,:] = np.eye(6)

# 创建J1
J1 = generate_J(x_Omega_1, y_Omega_1, z_Omega_1)

# 构建H(i-1)
H1 = J1 @ F4

# 构建F3(i)
F3_2 = np.eye(6)

# 构建R(i-1)
R1 = H1 @ F3_2 @ Ms_2

# 矩阵A：由于重定位所引入误差的转换矩阵，表征装配过程中的舱段在不同工序之间转换的重定位误差，决定于装配过程中的工序与工序之间的定位基准的变换。
# 在第一工位，由于前一工位引起的零件偏差误差为零。因此，当i=1时，A(0)=0 （6nx6n）
# 当i>1时，A(i-1)表示参考点RP的误差累积和定位方案的变化。

MMs_2 = np.zeros((18, 18))
# 将原始6x18的矩阵放置在新矩阵的左上角
MMs_2[:6, :18] = Ms_2

RR1 = np.zeros((18,18))
RR1[6:12,:18] = R1

A1_matrix = MMs_2 + RR1

W0_bar = np.zeros((6, 32))
# 将原始6x7的矩阵放置在新矩阵的上部
W0_bar[:6, :7] = Fs_bar2

W0_wavy = F3_2 @ MPs1_2_Pr @ W0_bar

U0 = np.zeros((19,32))
U0[:,7:26] = np.eye(19)
W1_bar = np.vstack((W0_wavy,U0))
# W1_bar = np.vstack((W0_bar,U0))

# 创建lamda2
lamda2 = np.zeros((12,32))
lamda2[:6,:] = J1 @ W1_bar

B2_matrix = np.vstack((MPs1_2_Pr @ W0_bar , lamda2))
# 矩阵U2：站2中与局部误差源的偏差
U2_matrix = np.random.uniform(0, 0.001, (32,1))


# 计算舱段2与舱段1装配后的误差：X2
X2 = A1_matrix @ X1 + B2_matrix @ U2_matrix

##### 舱段2与舱段3装配 #####
Q3 = generate_Qi(x1_root_3,x2_root_3,x3_root_3,x4_root_3,x5_root_3, 
                y1_root_3,y2_root_3,y3_root_3,y4_root_3,y5_root_3,
                z1_root_3,z2_root_3,z3_root_3,z4_root_3,z5_root_3)

# 创建矩阵S3(30x12)
S3 = np.zeros((30, 12))

S3[0:6,6:12] = np.eye(6)
S3[6:12,6:12] = np.eye(6)
S3[12:18,0:6] = np.eye(6)
S3[18:24,0:6] = np.eye(6)
S3[24:30,6:12] = np.eye(6)

# 计算G3
G3 = F2@Q3@S3

#计算舱段3的Fs_bar3
Fs_bar3 = generate_Fs_bar(nx_3, ny_3, nz_3, x1_3, y1_3, z1_3, x2_3, y2_3, z2_3, x3_3, y3_3, x4_3, y4_3, x5_3, y5_3)

# 构造F1_3
# 将左边的单位矩阵和右边的零矩阵拼接在一起
F1_3 = np.zeros((12,18))
F1_3[:,:12] = np.eye(12)

# 创建J2
J2 = generate_J(x_Omega2, y_Omega2, z_Omega2)

# 构建 M(Ps1(3),Pr)
MPs1_3_Pr_top = generate_MPs1_i_Pr(x_Omega2, y_Omega2, z_Omega2)
MPs1_3_Pr_bottom = generate_MPs1_i_Pr(x_Omega3, y_Omega3, z_Omega3)
MPs1_3_Pr = np.vstack((MPs1_3_Pr_top,MPs1_3_Pr_bottom))

II = np.vstack((np.eye(6),np.eye(6)))
# 构建Ms(3)(3)
Ms_3 = (II - MPs1_3_Pr) @ Fs_bar3 @ G3 @ F1_3

# 构建H(i-1)
H2 = J2 @ F4

# 构建F3(i)
F3_3 = np.zeros((6,12))
F3_3[:,6:] = np.eye(6)

# 构建R(i-1)
R2 = H2 @ F3_3 @ Ms_3

# 矩阵A：由于重定位所引入误差的转换矩阵，表征装配过程中的舱段在不同工序之间转换的重定位误差，决定于装配过程中的工序与工序之间的定位基准的变换。
# 在第一工位，由于前一工位引起的零件偏差误差为零。因此，当i=1时，A(0)=0 （6nx6n）
# 当i>1时，A(i-1)表示参考点RP的误差累积和定位方案的变化。

MMs_3 = np.zeros((18, 18))
# 将12x18的矩阵放置在新矩阵的左上角
MMs_3[:12, :18] = Ms_3

RR2 = np.zeros((18,18))
RR2[6:12,:18] = R2

A2_matrix = MMs_3 + RR2

# 此处同名，但是数值变了
W0_bar = np.zeros((6, 32))
# 将6x7的矩阵放置在新矩阵的上部
W0_bar[:6, :7] = Fs_bar3

W0_wavy = F3_3 @ MPs1_3_Pr @ W0_bar

U0 = np.zeros((19,32))
U0[:,7:26] = np.eye(19)
W1_bar = np.vstack((W0_wavy,U0))
# W1_bar = np.vstack((W0_bar,U0))

# 创建lamda3
# lamda3 = np.zeros((12,32))
lamda3 = J2 @ W1_bar


B3_matrix = np.vstack((MPs1_3_Pr @ W0_bar , lamda3))

# 矩阵U3：站3中与局部误差源的偏差
U3_matrix = np.random.uniform(0, 0.001, (32,1))

# 舱段2与舱段3装配后的误差
X3 = A2_matrix @ X2 + B3_matrix @ U3_matrix

print("舱段2、1、3的偏差如下：")
print(X3)