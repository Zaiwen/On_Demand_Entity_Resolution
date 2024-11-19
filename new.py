import sys
import os
from py2neo import Graph, Node

def jaccard_similarity(str1, str2):
    set1 = set(str1.split())
    set2 = set(str2.split())
    intersection = set1.intersection(set2)
    union = set1.union(set2)
    return len(intersection) / len(union) if len(union) > 0 else 0.0

def similar(a, b):
    return jaccard_similarity(a, b)

def connect_to_neo4j(uri, user, password):
    return Graph(uri, auth=(user, password))

def execute_query(graph, base_query, batch_size=50):
    offset = 0
    batch_number = 0

    while True:
        query = f"""
        {base_query}
        SKIP {offset} LIMIT {batch_size}
        """
        cursor = graph.run(query)
        data_batch = []

        while cursor.forward():
            data_batch.append(cursor.current)

        if not data_batch:
            break

        batch_number += 1
        yield data_batch

        offset += batch_size

def process_query_results(data_batch, output_dir, iteration):
    os.makedirs(output_dir, exist_ok=True)
    filtered_output_file = os.path.join(output_dir, "filtered_author_paper_pairs.txt")

    pair_count = 0
    unique_pairs = set()
    matching_pairs = []
    matched_data = []

    for record_index, record in enumerate(data_batch, start=1):
        author1 = record['a1']
        author2 = record['a2']
        paper1 = record['p1']
        paper2 = record['p2']

        name_similarity = similar(author1['name'], author2['name'])
        # paper_title_similarity = similar(paper1['title'], paper2['title'])

        pair_id = tuple(sorted((author1['id'], author2['id'])))

        if pair_id not in unique_pairs:
            unique_pairs.add(pair_id)

            output_str = {
                "author_1": {"id": author1['id'], "name": author1['name'], "code": author1['code']},
                "author_2": {"id": author2['id'], "name": author2['name'], "code": author2['code']},
                "name_similarity": name_similarity
            }
            matched_data.append(output_str)
            pair_count += 1

        if pair_id not in matching_pairs:
            matching_pairs.append(pair_id)

    with open(filtered_output_file, 'a', encoding='utf-8') as f:
        f.write(f"Iteration {iteration} matching pairs:\n")
        for pair_id in matching_pairs:
            author1, author2, paper1, paper2 = None, None, None, None
            for record in data_batch:
                if tuple(sorted((record['a1']['id'], record['a2']['id']))) == pair_id:
                    author1 = record['a1']
                    author2 = record['a2']
                    paper1 = record['p1']
                    paper2 = record['p2']
                    break

            if author1 and author2 and paper1 and paper2:
                name_similarity = similar(author1['name'], author2['name'])
                # paper_title_similarity = similar(paper1['title'], paper2['title'])

                output_str = (
                    f"Author 1: (ID: {author1['id']}, Name: {author1['name']}, Code: {author1['code']})\n"
                    f"Author 2: (ID: {author2['id']}, Name: {author2['name']}, Code: {author2['code']})\n"
                    f"Paper 1: {paper1['title']}, Paper 2: {paper2['title']}\n"
                    f"Name Similarity: {name_similarity:.2f}\n"
                    "--------------------------------------------\n"
                )
                f.write(output_str)

        f.write("\n")

    return pair_count, matched_data, filtered_output_file
