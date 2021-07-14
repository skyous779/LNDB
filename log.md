# 进度
## 2021.07.14
尝试了用权重取样和根据label值进行取样，但计算机内存溢出（我的笔记本8g,google colab分配有13g），可能取样方法比普通的随机取样更占内存,去服务器尝试用32g看看行不行。

1. 一张ct拥有n个patch,每个patch通过取样函数进行取样，m个patch组成一个queue,一个queue里又含有多个batch,batch_size只一次训练放入patch的个数。
2. tio.Queue()方法不知道能不能直接取样，因为给出的历程都是在一个subject(里面只有一个label和org)中取样，建议翻阅源码查看。

