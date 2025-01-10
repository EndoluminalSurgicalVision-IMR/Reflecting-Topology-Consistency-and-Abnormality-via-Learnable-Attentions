import numpy as np
import json
import re
import numpy as np
from typing import Dict, Tuple


def consistency_calculate(
        pred_sub: np.ndarray,
        descendants_matrix: np.ndarray,
        class2anno: Dict[str, Tuple[str, str]],
) -> Tuple[float, float]:
    """
    Compute the consistency of the predicted subtree annotations.

    Args:
        pred_sub (np.ndarray): A 1D array of predicted class indices for nodes.
        descendants_matrix (np.ndarray): A binary matrix where `descendants_matrix[i, j] == 1`
            indicates node j is a descendant of node i.
        class2anno (Dict[str, Tuple[str, str]]): A mapping from class indices to annotation parts
            in the form (part1, part2).

    Returns:
        Tuple[float, float]: A tuple containing two consistency scores:
            - The first score measures segmental consistency across subtrees.
            - The second score measures subsegmental consistency across subtrees.
            Both scores range from 0 (no consistency) to 1 (full consistency).
    """
    # Generate annotation part1 and part2 for the predicted classes
    anno_parts = np.array([class2anno[str(pred)] for pred in pred_sub])  # 将所有注释分为两部分
    anno_part1, anno_part2 = anno_parts[:, 0], anno_parts[:, 1]
    num_nodes = pred_sub.shape[0]

    # Initialize arrays to track subtree consistency
    consistent_subtree_sub = np.zeros(num_nodes, dtype=np.int32)
    consistent_subtree_seg = np.zeros(num_nodes, dtype=np.int32)
    is_subtree = np.zeros(num_nodes, dtype=np.int32)

    for i in range(num_nodes):
        # Skip "main bronchus" nodes
        if anno_part1[i] == "main bronchus":
            continue

        # Get descendants of the current node
        descendants = np.where(descendants_matrix[i, :] == 1)[0]

        if len(descendants) > 0:  # Node is not a leaf
            is_subtree[i] = 1
            anno_part1_i = anno_part1[i]
            anno_part2_i = anno_part2[i]

            # Check if all descendants share the same segments(part1)
            if np.all([anno_part1[d] == anno_part1_i for d in descendants]):
                consistent_subtree_seg[i] = 1
                # Case 1: Suffix matches exactly
                if anno_part2_i in ["a", "b", "c"] and np.all([anno_part2[d] == anno_part2_i for d in descendants]):
                    consistent_subtree_sub[i] = 1

                # Case 2: Composite suffix matches
                elif anno_part2_i in ["a+b", "a+c", "b+c"]:
                    allowed_suffixes = {anno_part2_i[0], anno_part2_i[2]}
                    if np.all([anno_part2[d] in allowed_suffixes for d in descendants]):
                        consistent_subtree_sub[i] = 1

                # Case 3: No suffix (empty string)
                elif anno_part2_i == "":
                    consistent_subtree_sub[i] = 1
    total_subtrees = np.sum(is_subtree)
    if total_subtrees == 0:
        consistency_seg = consistency_sub = 0
    else:
        consistency_seg = np.sum(consistent_subtree_seg) / total_subtrees
        consistency_sub = np.sum(consistent_subtree_sub) / total_subtrees

    return consistency_seg, consistency_sub



def calculate_topology_distance(pred: np.ndarray, gt: np.ndarray, spd: np.ndarray) -> float:
    """
    Calculate the average topology distance (td). For each node i, td[i] represents the minimum
    shortest path distance from node i to a node where the ground truth (gt) matches the predicted label.

    Args:
        pred (np.ndarray): A 1D array where pred[i] is the predicted label for node i.
        gt (np.ndarray): A 1D array where gt[i] is the ground truth label for node i.
        spd (np.ndarray): A 2D matrix where spd[i, j] represents the shortest path distance
            between node i and node j.

    Returns:
        float: The average topology distance across all nodes.
    """
    # Number of nodes
    num_nodes = spd.shape[0]

    # Initialize the topology distance array with a high default value (30)
    td = np.full(num_nodes, 30, dtype=np.float32)

    for i in range(num_nodes):
        # Find nodes where the ground truth matches the predicted label
        matching_nodes = np.where(gt == pred[i])[0]

        if matching_nodes.size > 0:
            # Update td[i] with the minimum shortest path distance to matching nodes
            td[i] = np.min(spd[i, matching_nodes])

    # Return the average topology distance
    return td.mean()
