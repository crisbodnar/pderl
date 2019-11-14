from core.mod_neuro_evo import SSNE
import numpy as np


def test_sort_groups_by_fitness():
    genome = list(range(5))
    fitness = np.arange(5)

    sorted_groups = SSNE.sort_groups_by_fitness(genome, fitness)

    # Check that the list is sorted correctly
    for i, group in enumerate(sorted_groups):
        print(group)
        if i < len(sorted_groups) - 1:
            assert (sorted_groups[i][2] >= sorted_groups[i + 1][2])

    # Check the reverse group is not in the list
    for group in sorted_groups:
        assert ((group[1], group[0], group[2]) not in sorted_groups)

    # Check the fitness summs are correctly computed
    for group in sorted_groups:
        assert (group[0] + group[1] == group[2])

    # Check all possible combinations are generated
    assert (len(sorted_groups) == (len(genome) * (len(genome) - 1) // 2))

    # Check they are added in the right order
    for group in sorted_groups:
        assert group[0] > group[1]
