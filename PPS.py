def parse_input_from_file(file_path):
    with open(file_path, 'r') as file:
        lines = file.readlines()

    profile_collection = {}
    current_entity = None

    for line in lines:
        line = line.strip()
        if line.startswith("target entity:"):
            current_entity = int(line.split(':')[1].strip())
            profile_collection[current_entity] = []
        elif line.startswith("similar entities:"):
            similar_entities = line.split(':')[1].strip()
            if similar_entities:
                profile_collection[current_entity] = list(map(int, similar_entities.split()))

    return profile_collection


def buildRedundancyPositiveBlocks(profile_collection):
    return profile_collection


def buildProfileIndex(blocks):
    profile_index = {}
    for block_id, profiles in blocks.items():
        for profile in profiles:
            if profile not in profile_index:
                profile_index[profile] = []
            profile_index[profile].append(block_id)
    return profile_index


def getComparison(i, j, weight):
    return (i, j, weight)


def wScheme(profile1, profile2, block):
    return 1  # 示例权重计算，使用档案ID的乘积


# 初始化阶段
def initialization_phase(P, wScheme):
    B = buildRedundancyPositiveBlocks(P)
    ProfileIndex = buildProfileIndex(B)
    SortedProfileList = []
    topComparisonsSet = set()

    for pi in P:
        weights = {}
        distinctNeighbors = set()

        for bk in ProfileIndex.get(pi, []):
            for pj in B[bk]:
                if pj != pi:
                    weights[pj] = weights.get(pj, 0) + wScheme(pj, pi, bk)
                    distinctNeighbors.add(pj)

        topComparison = None
        duplicationLikelihood = 0

        for j in distinctNeighbors:
            duplicationLikelihood += weights[j]
            if topComparison is None or weights[j] > topComparison[2]:
                topComparison = getComparison(pi, j, weights[j])

        if topComparison:
            topComparisonsSet.add(topComparison)

        duplicationLikelihood /= max(1, len(distinctNeighbors))
        SortedProfileList.append((pi, duplicationLikelihood))

    ComparisonList = list(topComparisonsSet)
    ComparisonList.sort(key=lambda x: x[2], reverse=True)
    SortedProfileList.sort(key=lambda x: x[1], reverse=True)

    # 打印前10个ComparisonList
    print("Top 10 comparisons:")
    for comparison in ComparisonList[:10]:
        print(comparison)

    # 打印SortedProfileList
    print("SortedProfileList:")
    for profile in SortedProfileList[:10]:
        print(profile)

    return ComparisonList, SortedProfileList, ProfileIndex


# Emission Phase for PPS
def emission_phase_for_pps(ComparisonList, SortedProfileList, ProfileIndex, wScheme, Kmax=10):
    checkedEntities = set()
    B = buildRedundancyPositiveBlocks(P)  # Added this line to ensure B is available

    if not ComparisonList:
        if SortedProfileList:
            pi, _ = SortedProfileList.pop(0)
            checkedEntities.add(pi)
            weights = {}
            distinctNeighbors = set()
            SortedStack = []

            for bk in ProfileIndex.get(pi, []):
                for pj in B[bk]:
                    if pj != pi and pj not in checkedEntities:
                        weights[pj] = weights.get(pj, 0) + wScheme(pj, pi, bk)
                        distinctNeighbors.add(pj)

            for j in distinctNeighbors:
                comparison = getComparison(pi, j, weights[j])
                SortedStack.append(comparison)
                if len(SortedStack) > Kmax:
                    SortedStack.pop(0)  # 保持SortedStack大小为Kmax

            ComparisonList.extend(SortedStack)
            ComparisonList.sort(key=lambda x: x[2], reverse=True)


    # 跳过权重小于2的比较
    while ComparisonList and ComparisonList[0][2] < 2:
        ComparisonList.pop(0)

    # print(f"\nCurrent SortedProfileList:")
    # for profile in SortedProfileList[:10]:
    #     print(profile)
    #
    # print(f"Current ComparisonList:")
    # for comparison in ComparisonList[:10]:
    #     print(comparison)


    return ComparisonList.pop(0) if ComparisonList else None


# 文件路径
file_path = "output_file/blocks.txt"

# 将解析的文件数据转换成P
P = parse_input_from_file(file_path)

# 执行初始化阶段算法并输出结果
ComparisonList, SortedProfileList, ProfileIndex = initialization_phase(P, wScheme)
# 一开始设置为空
ComparisonList = []

# 持续获取并输出下一最佳比较，直到ComparisonList为空或达到限制
while SortedProfileList or ComparisonList:
    next_best_comparison = emission_phase_for_pps(ComparisonList, SortedProfileList, ProfileIndex, wScheme)
    print(f"The next best comparison: {next_best_comparison}")
