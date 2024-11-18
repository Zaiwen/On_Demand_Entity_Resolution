from new import jaccard_similarity, similar, connect_to_neo4j, execute_query, process_query_results
from pattern2blocks import parse_filtered_pattern, generate_blocks
from PPS import (
    parse_input_from_file,
    initialization_phase,
    emission_phase_for_pps,
    load_ground_truth,
    save_matched_results,
    wScheme,
)
import time
import os
from py2neo import Graph
import configparser

def clear_previous_results(output_dir):
    filtered_output_file = os.path.join(output_dir, "filtered_pattern.txt")
    blocks_file = os.path.join(output_dir, "blocks.txt")

    if os.path.exists(filtered_output_file):
        os.remove(filtered_output_file)

    if os.path.exists(blocks_file):
        os.remove(blocks_file)


def process_and_save_results(filtered_output_file, iteration, matched_pairs, start_time):
    input_file = filtered_output_file
    output_file_blocks = 'output_file/blocks.txt'
    ground_truth_path = 'dataset/network/arxiv_ground_turth.txt'
    results_path = 'output_file/results.txt'
    time_log_path = 'output_file/time.txt'

    print(f"Processing iteration {iteration}, reading from {input_file}...")
    ground_truth = load_ground_truth(ground_truth_path)

    entity_similarities = parse_filtered_pattern(input_file)
    blocks = generate_blocks(entity_similarities)

    with open(output_file_blocks, 'a', encoding='utf-8') as file:
        file.write(f"Iteration {iteration} blocks:\n")
        all_entities = set()
        for block in blocks:
            all_entities.update(block)

        for entity in sorted(all_entities):
            similar_entities = sorted(entity_similarities[entity])
            similar_entities_str = " ".join(map(str, similar_entities))
            file.write(f"target entity: {entity}\n")
            file.write(f"similar entities: {similar_entities_str}\n\n")

    P = parse_input_from_file(output_file_blocks)
    ComparisonList, SortedProfileList, ProfileIndex = initialization_phase(P, wScheme)

    saved_results = set()
    new_matches = 0
    total_extractions = 0
    count = 0
    last_pi = None

    while SortedProfileList or ComparisonList:
        next_best_comparison, count = emission_phase_for_pps(ComparisonList, SortedProfileList, P, ProfileIndex,
                                                             wScheme, total_extractions, count)

        if not next_best_comparison:
            break

        pi, pj, weight = next_best_comparison


        if (pi, pj) not in matched_pairs and (pj, pi) not in matched_pairs:

            pi_pj_found_via_transitivity = False
            for pk in matched_pairs:
                if (pi, pk) in matched_pairs and (pj, pk) in matched_pairs:
                    pi_pj_found_via_transitivity = True
                    break

            #Embed any matching function here

            # ---------------------------------------------------------
            if (pi, pj) in ground_truth or (pj, pi) in ground_truth:
                save_matched_results(results_path, (pi, pj), saved_results)

                # print(f"Matched and saved: ({pi}, {pj})")
            #-------------------------------------------------------------

                if pi != last_pi:
                    elapsed_time = time.time() - start_time
                    with open(time_log_path, 'a') as time_file:
                        time_file.write(f"{elapsed_time:.4f}\n")


                last_pi = pi

                new_matches += 1
            matched_pairs.add((pi, pj))

    return new_matches


def main():
    start_time = time.time()

    config = configparser.ConfigParser()
    config.read("config.ini")
    neo4j_profile = config["neo4j"]["profile"]
    neo4j_user = config["neo4j"]["username"]
    neo4j_password = config["neo4j"]["password"]

    print("connecting Neo4j ...")

    graph = Graph(neo4j_profile, auth=(neo4j_user, neo4j_password))

    print("Successfully connected to the Neo4j database!")

    query1 = """
MATCH (a1:Author)-[:WROTE]->(p1:Paper),
      (a2:Author)-[:WROTE]->(p2:Paper)
WHERE a1 <> a2  
    AND a1.name contains 'robbert' AND a2.name contains 'robbert'
  AND apoc.text.levenshteinDistance(a1.name, a2.name) <=2
RETURN a1, a2, p1, p2
   """

    total_pairs = 0
    output_dir = "output_file"
    os.makedirs(output_dir, exist_ok=True)

    iteration = 1
    last_new_matches = None
    matched_pairs = set()

    print("Executing query 1...")
    data_stream = execute_query(graph, query1)

    query_has_new_matches = False
    for data_batch in data_stream:
        clear_previous_results(output_dir)

        pair_count, matched_data, filtered_output_file = process_query_results(data_batch, output_dir, iteration)
        total_pairs += pair_count

        if pair_count > 0:
            new_matches = process_and_save_results(filtered_output_file, iteration, matched_pairs,
                                                   start_time)
            print(f"Iteration {iteration}, new matches: {new_matches}")
            end_time = time.time()
            execution_time = end_time - start_time
            print(f"execution_time：{execution_time:.4f} s")
            iteration += 1

            if last_new_matches is not None and new_matches < last_new_matches / 6:
                print("New matches have dropped significantly")
                # break

            if new_matches > 0:
                query_has_new_matches = True
                last_new_matches = new_matches
            # else:
            # break

    if not query_has_new_matches:
        print("No new matches found, stopping execution.")

    end_time = time.time()
    execution_time = end_time - start_time
    print(f"A total of {total_pairs} pairs were matched")
    print(f"time：{execution_time:.4f} s")


if __name__ == "__main__":
    main()


