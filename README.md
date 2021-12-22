# Optical Music Recognition with CRNN in Jittor

本仓库为Convolutional Recurrent Neural Network (CRNN) 的[Jittor](https://cg.cs.tsinghua.edu.cn/jittor/)实现版本，用于乐谱识别(Optical Music Recognition)任务。模型和训练部分基于[Pytorch版本](https://github.com/meijieru/crnn.pytorch)修改。

## 目录架构

`model.py`为CRNN模型，`utils.py`定义其它类和方法。`dataset.py`提供两种加载数据集的方法，其中`OMRDataset`直接从文件加载数据，效率较低；使用`lmdbDataset`可加速数据读取，lmdb数据集可使用`create_dataset.py`制作。分别在`train.py`与`test.py`中使用数据集训练与测试。`demo.py`提供了一个模型推断的示例。

## 运行示例

`example/model.pkl`是一个在带扰动数据集上训练的模型。使用命令

```
python demo.py --imagePath example/000051652-1_2_1.png --modelPath example/model.pkl
```

可识别示例图片。

示例图片：![000051652-1_2_1](example/000051652-1_2_1.png)

期望输出：

```
[['clef-C1', 'keySignature-EbM', 'timeSignature-2/4', '-', '-', '-', '-', '-', '-', '-', '-', '-', '-', '-', '-', '-', '-', '-', '-', '-', '-', 'multirest-23', '-', '-', '-', '-', '-', '-', '-', '-', '-', 'barline', 'barline', '-', 'rest-quarter', '-', '-', '-', '-', '-', '-', 'rest-eighth', '-', '-', '-', 'note-Bb4_eighth', '-', '-', '-', 'barline', 'barline', '-', 'note-Bb4_quarter.', '-', '-', '-', '-', '-', '-', '-', '-', 'note-G4_eighth', '-', '-', '-', 'barline', 'barline', '-', 'note-Eb5_quarter.', '-', '-', '-', '-', '-', '-', '-', 'note-D5_eighth', '-', '-', '-', '-', 'barline', '-', '-', 'note-C5_eighth', '-', '-', '-', 'note-C5_eighth', '-', '-', '-', 'rest-quarter', '-', '-', '-', '-', '-', '-', 'barline']] => ['clef-C1 keySignature-EbM timeSignature-2/4 multirest-23 barline rest-quarter rest-eighth note-Bb4_eighth barline note-Bb4_quarter. note-G4_eighth barline note-Eb5_quarter. note-D5_eighth barline note-C5_eighth note-C5_eighth rest-quarter barline']
```

## 训练模型

使用[PrIMuS数据集](https://grfia.dlsi.ua.es/primus/)进行训练。解压后将`Corpus`目录置于`data`目录下，同时在`train.txt, val.txt, test.txt `内设置训练、验证、测试集。我们提供两种读取数据方式。

* 文件读取：速度较慢
  * 将`train.py`的第59-68行替换为第70-83行，直接运行`python train.py`
* lmdb数据集读取：速度较快，需制造数据集
  * 在`create_dataset.py`的第80行设置数据集存储路径、第82行设置所需制造数据集的数据项名称列表，运行得到数据集。
  * 运行`python train.py --trainRoot {train path} --valRoot {val path}`，其中将训练、验证数据集选项替换为对应路径。

## 测试

在`test.py`中设置模型与数据集并运行以得到测试结果。