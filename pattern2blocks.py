import re
import os
from collections import defaultdict

def parse_filtered_pattern(file_path):
    author_similarities = defaultdict(set)


    with open(file_path, 'r', encoding='utf-8') as file:
        content = file.read()


    pattern = re.compile(
        r'Author 1: \(ID: (\d+), Name: (.*?), Code: (.*?)\)\n'      # Matches Author 1's ID, Name, and Code
        r'Author 2: \(ID: (\d+), Name: (.*?), Code: (.*?)\)\n'      # Matches Author 2's ID, Name, and Code
        r'Paper 1: (.*?), Paper 2: (.*?)\n'                         # Matches Paper 1 and Paper 2 titles
        r'Name Similarity: (\d\.\d+)\n'                             # Matches Name Similarity
    )

    matches = pattern.findall(content)

    for match in matches:
        id1, author1_name, code1, id2, author2_name, code2, paper1_title, paper2_title, name_similarity = match
        id1, id2 = int(id1), int(id2)


        if float(name_similarity) > 0:
            author_similarities[id1].add(id2)
            author_similarities[id2].add(id1)

    return author_similarities


def generate_blocks(author_similarities):
    blocks = []
    visited = set()

    def dfs(entity, block):
        if entity not in visited:
            visited.add(entity)
            block.add(entity)
            for neighbor in author_similarities[entity]:
                dfs(neighbor, block)

    for entity in author_similarities:
        if entity not in visited:
            block = set()
            dfs(entity, block)
            blocks.append(block)

    return blocks


def main():

    input_file = 'output_file/filtered_author_paper_pairs.txt'
    output_file = 'output_file/author_blocks.txt'

    if not os.path.exists(input_file):
        print(f"Input file {input_file} not found.")
        return

    author_similarities = parse_filtered_pattern(input_file)

    blocks = generate_blocks(author_similarities)

    with open(output_file, 'w', encoding='utf-8') as file:
        for block in sorted(blocks, key=lambda x: min(x)):
            sorted_block = sorted(block)
            file.write(f"Block of similar authors: {', '.join(map(str, sorted_block))}\n")
            for entity in sorted_block:
                similar_entities = sorted(author_similarities[entity])
                similar_entities_str = " ".join(map(str, similar_entities))
                file.write(f"target author: {entity}\n")
                file.write(f"similar authors: {similar_entities_str}\n\n")

    print(f"Author similarity blocks have been written to {output_file}")

if __name__ == "__main__":
    main()
