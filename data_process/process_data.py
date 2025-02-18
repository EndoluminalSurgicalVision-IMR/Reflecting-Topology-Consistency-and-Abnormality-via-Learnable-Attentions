from tree_parse_skelbased import tree_parse
from feature_cuda import  feature_extraction_cuda
import nibabel

 ##### INPUT PATHS ####################################################
# Input file paths (nii.gz format)
label_path = ""  # Path to binary airway segmentation mask
skel_path = ""   # Path to airway skeleton
lobe_path = ""   # Path to lung lobe segmentation (from lung_mask package)

spacing, parent_map, children_map, generation, trachea, tree_parsing_skel, tree_parsing = tree_parse(label_path,
                                                                                                     skel_path)
segmentation = nibabel.load(lobe_path).get_fdata()

feature, edge, edge_feature, node_idx= (
    feature_extraction_cuda(tree_parsing_skel, spacing, segmentation, parent_map, children_map, generation, trachea))

"""
Final outputs to save:
# - tree_parsing_skel
# - tree_parsing
# - feature
# - edge
# - edge_feature
# - node_idx

Recommended naming convention:
- skel_parse.nii.gz
- parse.nii.gz
- x.npy
- edge.npy
- edge_feature.npy
- node_idx.npy
"""
