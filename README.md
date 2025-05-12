# **FastER: Fast On-Demand Entity Resolution in Property Graphs**

**FastER** is our newly developed open-source framework designed to efficiently perform on-demand entity resolution, specifically tailored for property graph datasets. All code is written in **Python 3**, supporting flexible extensions and various datasets.

![FastER Overview](FastER%20overview.svg)

---

## **Requirements**
- Python 3.x
- Neo4j Database (supports local server versions)

---

## **Installation**
You can install the external libraries listed in `requirements.txt` using the following command:

```bash
pip install -r requirements.txt
```


## **Datasets**

FastER supports a variety of benchmark datasets for both graph and relational entity resolution (ER), as detailed below:

### **1. Graph ER Benchmark Datasets**
FastER supports the following commonly used graph datasets, ready for experiments:

- **WWC, ArXiv, CiteSeer, GDS Datasets**  
  Dataset Links:  
  - [WWC, ArXiv, CiteSeer, GDS Datasets - linqs.org](https://linqs.org/datasets/)  
  - [WWC, ArXiv, CiteSeer, GDS Datasets - Neo4j Sandbox](https://neo4j.com/sandbox/)

### **2. Relational ER Benchmark Datasets**
FastER also supports the following classic relational ER benchmark datasets:

- [Fodors-Zagat, DBLP-ACM, Amazon-Google Datasets](https://github.com/anhaidgroup/deepmatcher/blob/master/Datasets.md)
- **Other Datasets**:
  - **SIGMOD20** (referred to as `alaska_camera`):  
    Download the dataset from [SIGMOD 2020 Official Website](http://www.inf.uniroma3.it/db/sigmod2020contest/task.html) and place the folder `2013_camera_specs` in the project directory `data_raw`.
  - **Altosight** (referred to as `altosight`):  
    The Altosight dataset is currently unavailable but will be supported in the future.

### <span style="color:red;">**3. Synthetic Datasets(Graph Datasets)**</span>  
WWC and GDS datasets: These datasets are respectively derived from the Women’s World Cup (WWC) dataset and the Graph Data Science (GDS) dataset, and are augmented with 10% duplicate nodes to introduce noise.GDS-* models connections between global airports, including nodes representing airports and their connections. WWC-* includes nodes representing persons, teams, matches, and other related entities.



## Data Conversion for Entity Resolution 
Q:How to Use FastER on Relational Datasets or Run graph datasets on relational datasets-based baselines?

A:https://anonymous.4open.science/r/data-conversion-E737      
This project provides tools for converting between relational data types and graph data types for entity resolution (ER). The goal is to enable efficient integration and processing of data from various sources, facilitating graph-based ER tasks.



## **Rule Mining**
The rule mining feature in FastER is based on the definitions and processes described in the following papers:
1. **Discovering Graph Differential Dependencies**  
2. **Certus: An Effective Entity Resolution Approach with Graph Differential Dependencies (GDDs)**  

For a deeper understanding of rule definitions and mining processes, users are encouraged to read these papers.

---

## **Parameter Configuration**

All parameters have been preconfigured in the codebase. No manual adjustment is required when switching datasets, as the impact of configuration changes on the overall results is negligible.


## **Running example**
Below are the steps to run FastER using the default **Arxiv** dataset:

1. **Install Neo4j**:  
   Install the Neo4j database (any compatible local server version will work).

2. **Import Dataset**:  
   Run the `neo4j.ipynb` script to import the Arxiv dataset into the Neo4j database.

3. **Perform Entity Resolution**:  
   Use the following command to run the `main1.py` script for entity resolution:

   ```bash
   python main1.py


## **Recent work on automatic rule mechanisms**
This is a recent effort on automatic rule mechanisms. For more details, please refer to: [LLM-Enhanced Entity Matching Using Graph Differential Dependencies](https://anonymous.4open.science/r/LLM-Enhanced-Entity-Matching-Using-Graph-Differential-Dependencies-F03B)

