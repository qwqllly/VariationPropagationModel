# VariationPropagationModel

## 文件更新说明
### 详细版代码
自己根据论文推导编写的误差计算代码。有2个版本。其中，版本1梳理了全部参数，版本2进行了简化，只保留骨架部分。

重读论文，与小伙伴讨论后，更新了结构参数Fs的推导过程。坦白说，由于缺少相关力学知识，我们也不知道哪一个是更合理的，所以在此给出了2个版本。

### python脚本代码
由于文件 variation propagation详细版.ipynb 是按照舱段装配流程边写边试、逐行调试的，没有函数封装，复用性低，并且重复代码难以维护。本人将其整理成了函数形式的 Python 脚本，将重复代码进行了函数封装，可读性下降但复用性提高。对比阅读后可以快速上手复用。

### 运行结果
X3_data是运行程序计算出的误差数值，分别是前舱段、中舱段、后舱段 质心处的具体误差，单位与程序原始设定保持一致。

Y_data是自主选取的测量点的计算后的误差数值。

### 图片含义
3个飞机舱段的示意图

装配工序及示意图

重复实验后得到的三个舱段的误差可视化结果（概率分布形式）

## 已发表小论文
https://kns.cnki.net/kcms2/article/abstract?v=bTgd32KJj6tVa-ljaqQ-nF8T2tS799Hp5QzLC6fC2px3PhvUTbZdQQ5zDVUo9RG2aQv4UUgd6xKel1lfPjWOfUtWjEs95P9sL9It5Fax_-xLrH-8_qbBrYcwrZ4YJgRA5Z9raAYsSJ2H2PLnwFcqTU44qfmMxuUDlj_St4yeoKQAPUdTYv6R_A==&uniplatform=NZKPT

## 已发表硕士毕业论文
（250708更新）目前还未在知网查询到，以上内容是本人硕论的第一个实验（由于实验室要求，隐去了所用的实测数据，请谅解）。
