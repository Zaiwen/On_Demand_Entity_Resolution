# **FastER: Fast On-Demand Entity Resolution in Property Graphs**

**FastER** 是我们开发的新开源框架，旨在按需执行高效的实体解析，特别适用于属性图数据集。所有代码均使用 **Python 3** 编写，支持灵活的扩展和多种数据集。

![FastER Overview](FastER%20overview.svg)

---

## **要求**
- Python 3.x
- Neo4j 数据库（支持本地服务器版本）

---

## **数据集**

FastER 支持多种图形和关系 ER（实体解析）基准数据集，具体如下：

### **1. 图形 ER 基准数据集**
FastER 支持以下常用的图形数据集，可直接用于实验：

- **WWC、ArXiv、CiteSeer、GDS 数据集**  
  数据集链接：  
  - [WWC、ArXiv、CiteSeer、GDS 数据集 - linqs.org](https://linqs.org/datasets/)  
  - [WWC、ArXiv、CiteSeer、GDS 数据集 - Neo4j Sandbox](https://neo4j.com/sandbox/)

### **2. 关系 ER 基准数据集**
FastER 还支持以下经典的关系 ER 基准数据集：

- [Fodors-Zagat, DBLP-ACM, Amazon-Google 数据集](https://github.com/anhaidgroup/deepmatcher/blob/master/Datasets.md)
- **其他数据集**：
  - **SIGMOD20**（简称 `alaska_camera`）：  
    从 [SIGMOD 2020 官网](http://www.inf.uniroma3.it/db/sigmod2020contest/task.html) 下载数据集，并将文件夹 `2013_camera_specs` 放入项目的 `data_raw` 文件夹中。
  - **Altosight**（简称 `altosight`）：  
    Altosight 数据集目前尚未上线，未来将提供支持。


## **如何使用 FastER 在关系数据集上实验**
由于 **FastER** 始终需要图数据集作为输入，用户需要先将关系数据集转换为图数据集。为此，我们提供了相关的预处理脚本：
- **`preprocessing.py`** 和 **`txt2csv.py`**：用于关系数据到图数据的转换。  
如果需要详细了解如何进行转换，建议阅读并使用这些脚本。项目中已包含转换后的示例数据集，可直接使用。

---

## **规则挖掘**
FastER 中的规则挖掘基于以下两篇研究论文的定义与过程：
1. **Discovering Graph Differential Dependencies**  
2. **Certus: An Effective Entity Resolution Approach with Graph Differential Dependencies (GDDs)**  

用户可以通过阅读上述论文深入了解规则定义与挖掘方法。

---

## **运行示例**
以下是运行 FastER 的步骤，基于默认数据集 **Arxiv**：

1. **安装 Neo4j**：  
   安装本地 Neo4j 数据库服务器（任何版本均兼容）。

2. **导入数据集**：  
   运行 `neo4j.ipynb` 脚本，将 Arxiv 数据集导入 Neo4j 数据库。

3. **执行实体解析**：  
   使用以下命令运行 `main1.py` 文件：
   ```bash
   python main1.py
