import sys

import configparser
from py2neo import Graph
config = configparser.ConfigParser()
config.read("config.ini")
neo4j_profile = config["neo4j"]["profile"]
neo4j_user = config["neo4j"]["username"]
neo4j_password = config["neo4j"]["password"]

graph = Graph(neo4j_profile, auth=(neo4j_user, neo4j_password))

query1 = """
MATCH (a1:Author)
WHERE  
  a1.name contains 'robbert'
RETURN a1
"""


airport_ids = set()

data1 = graph.run(query1).data()
for record in data1:
    airport_ids.add(record['a1']['id'])




for airport_id in airport_ids:
    print(airport_id)

def generate_subgroundtruth(paper_ids, input_file, output_file):

    with open(input_file, 'r') as infile:
        lines = infile.readlines()


    subgroundtruth_pairs = set()

    current_pair = []
    for line in lines:

        line = line.strip()
        if line:
            current_pair.append(line)
        else:

            if len(current_pair) == 2:
                id1, id2 = current_pair
                # print(id1, id2)

                if id1 in paper_ids and id2 in paper_ids:

                    pair = tuple(sorted((id1, id2)))
                    subgroundtruth_pairs.add(pair)

            current_pair = []

    if len(current_pair) == 2:
        id1, id2 = current_pair
        if id1 in paper_ids and id2 in paper_ids:
            pair = tuple(sorted((id1, id2)))
            subgroundtruth_pairs.add(pair)


    with open(output_file, 'w') as outfile:
        for pair in sorted(subgroundtruth_pairs):
            outfile.write(f"{pair[0]} {pair[1]}\n")


input_file = 'dataset/network/arxiv_ground_turth.txt'
output_file = 'dataset/subgroundtruth.txt'

generate_subgroundtruth(airport_ids, input_file, output_file)

print("The subgroundtruth.txt file is generated.")
