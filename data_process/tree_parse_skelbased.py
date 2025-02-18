import numpy as np
from scipy import ndimage
import skimage.measure as measure
import nibabel
from itertools import combinations

import pycuda.driver as drv
from pycuda.compiler import SourceModule
def large_connected_domain(label):
    cd, num = measure.label(label, return_num=True, connectivity=1)
    volume = np.zeros([num])
    for k in range(num):
        volume[k] = ((cd == (k + 1)).astype(np.uint8)).sum()
    volume_sort = np.argsort(volume)
    label = (cd == (volume_sort[-1] + 1)).astype(np.uint8)

    label = ndimage.binary_fill_holes(label)
    label = label.astype(np.uint8)
    return label

def skeleton_parsing(skeleton):
    # separate the skeleton
    neighbor_filter = ndimage.generate_binary_structure(3, 3)
    skeleton_filtered = ndimage.convolve(skeleton, neighbor_filter) * skeleton
    skeleton_parse = skeleton.copy()
    skeleton_parse[skeleton_filtered > 3] = 0

    nd_labels, num = ndimage.label(skeleton_parse,structure=neighbor_filter)
    sizes = np.bincount(nd_labels.ravel())
    mask = sizes > 1
    mask[0] = False
    skeleton_parse = mask[nd_labels]
    cd, num = ndimage.label(skeleton_parse, structure=neighbor_filter)
    return skeleton_parse, cd, num


def tree_parsing1(skeleton_parse, label, cd):
    # parse the airway tree
    edt, inds = ndimage.distance_transform_edt(1 - skeleton_parse, return_indices=True)
    tree_parsing = cd[inds[0, ...], inds[1, ...], inds[2, ...]] * label
    return tree_parsing

def loc_trachea(tree_parsing, num):
    # find the trachea
    volume = np.zeros([num])
    for k in range(num):
        volume[k] = ((tree_parsing == (k + 1)).astype(np.uint8)).sum()
    volume_sort = np.argsort(volume)
    trachea = (volume_sort[-1] + 1)
    return trachea

def loc_trachea_vectorized(tree_parsing, num):
    # Compute volumes for all regions at once using histogram
    volumes = np.bincount(tree_parsing.flatten(), minlength=num+1)[1:num+1]  # +1 to handle zero-index and slice off zero volume
    # Find the index of the region with the maximum volume
    trachea = np.argmax(volumes) + 1
    return trachea


def adjacent_map_cuda(skeleton,skeleton_parse,cd):
    mod = SourceModule("""
        __global__ void binary_dilation_and_adjacency_parallel(
            int *bifurcation_labeled, int *cd, int *ad_matric, int num, int num_cd, int sx, int sy, int sz
        ) {
            int label = blockIdx.x * blockDim.x + threadIdx.x + 1; // label is 1-indexed

            if (label > num) return;

            for (int z = 0; z < sz; z++) {
                for (int y = 0; y < sy; y++) {
                    for (int x = 0; x < sx; x++) {
                        // int idx = x + y * sx + z * sx * sy;  before
                        int idx = z + y * sz + x * sy * sz; //after
                        int cd_cur = (bifurcation_labeled[idx] == label) ? 1 : 0;
                        if (cd_cur == 1) {
                            //atomicOr(&cd[idx], 1);
                            for (int dz = -1; dz <= 1; dz++) {
                                for (int dy = -1; dy <= 1; dy++) {
                                    for (int dx = -1; dx <= 1; dx++) {
                                        int nx = x + dx;
                                        int ny = y + dy;
                                        int nz = z + dz;
        
                                        if (nx >= 0 && nx < sx && ny >= 0 && ny < sy && nz >= 0 && nz < sz) {
                                            int n_idx = nz + ny * sz + nx * sy * sz;
                                            int neighbor_value = cd[n_idx];
                                            if (neighbor_value > 0) {
                                                atomicOr(&ad_matric[(label - 1) * num_cd + (neighbor_value - 1)], 1);
                                            }
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
    """)
    bifurcation = skeleton - skeleton_parse
    structure = np.ones((3, 3, 3), dtype=bool)

    bifurcation_labeled, num_features = ndimage.label(bifurcation, structure=structure)

    sx, sy, sz = bifurcation_labeled.shape
    num = num_features
    num_cd = int(cd.max())

    # Initialize outputs
    ad_matric = np.zeros((num, num_cd), dtype=np.int32)

    # Allocate device memory
    bifurcation_labeled_gpu = drv.mem_alloc(bifurcation_labeled.nbytes)
    ad_matric_gpu = drv.mem_alloc(ad_matric.nbytes)
    #cd = np.zeros_like(bifurcation)
    cd_gpu = drv.mem_alloc(cd.nbytes)

    # Copy data to GPU
    drv.memcpy_htod(bifurcation_labeled_gpu, bifurcation_labeled)
    drv.memcpy_htod(ad_matric_gpu, ad_matric)
    drv.memcpy_htod(cd_gpu, cd)

    # Define grid and block sizes
    block_size = 256
    grid_size = (num + block_size - 1) // block_size

    # Get the kernel function
    binary_dilation_and_adjacency_parallel = mod.get_function("binary_dilation_and_adjacency_parallel")

    # Execute the kernel
    binary_dilation_and_adjacency_parallel(
        bifurcation_labeled_gpu, cd_gpu, ad_matric_gpu, np.int32(num), np.int32(num_cd),np.int32(sx), np.int32(sy), np.int32(sz),
        block=(block_size, 1, 1), grid=(grid_size, 1)
    )

    # Copy results back to host
    drv.memcpy_dtoh(ad_matric, ad_matric_gpu)
    drv.memcpy_dtoh(cd, cd_gpu)

    # Free GPU resources
    bifurcation_labeled_gpu.free()
    ad_matric_gpu.free()
    cd_gpu.free()

    adjent_map = np.zeros((num_cd, num_cd), dtype=np.int32)
    for i in range(num):
        values = np.where(ad_matric[i] == 1)[0]
        pairs = list(combinations(values, 2))
        for pair in pairs:
            adjent_map[pair[0],pair[1]] = 1
            adjent_map[pair[1], pair[0]] = 1
    return adjent_map

def parent_children_map(ad_matric, trachea, num):
    # build the parent map and children map
    parent_map = np.zeros((num, num), dtype=np.uint8)
    children_map = np.zeros((num, num), dtype=np.uint8)
    generation = np.zeros((num), dtype=np.uint8)
    processed = np.zeros((num), dtype=np.uint8)

    processing = [trachea - 1]
    parent_map[trachea - 1, trachea - 1] = 1
    while len(processing) > 0:
        iteration = processing
        # print("items in this iteration: ", iteration)
        processed[processing] = 1
        processing = []
        while len(iteration) > 0:
            cur = iteration.pop()
            children = np.where(ad_matric[cur, :] > 0)[0]
            for i in range(len(children)):
                cur_child = children[i]
                if parent_map[cur_child, :].sum() == 0:
                    parent_map[cur_child, cur] = 1
                    children_map[cur, cur_child] = 1
                    generation[cur_child] = generation[cur] + 1
                    processing.append(cur_child)
                else:
                    if generation[cur] + 1 == generation[cur_child]:
                        parent_map[cur_child, cur] = 1
                        children_map[cur, cur_child] = 1
    return parent_map, children_map, generation


def find_bb_3D(segmentation):
    if len(segmentation.shape) != 3:
        print("The dimension of input is not 3!")
    pos = np.where(segmentation)
    return pos[0].min(), pos[0].max() + 1, pos[1].min(), pos[1].max() + 1, pos[2].min(), pos[2].max() + 1


def tree_parse(path_label: str, path_skel: str) -> tuple:
    # According to bifurcation points, the binary segmentation of airway can be branched.
    # Each branch is assigned a unique value in the volume.
    #
    # Returns:
    # - spacing(tuple): Voxel spacing (dx, dy, dz) from nii header
    # - parent_map(ndarray): NxN adjacency matrix, parent_map[i, j] = 1 indicates
    #               branch j+1 is parent of branch i+1 (0-based index)
    # - children_map (ndarray): NxN adjacency matrix for child relationships
    # - generation (ndarray): (N,) array indicating branch generation in the airway tree.
    # - trachea (int): indicating which branch correspond to the trachea.
    # - tree_parsing_skel: The branched airway.
    # - tree_parsing: The branched skeleton.


    label_nii = nibabel.load(path_label)
    skel_nii = nibabel.load(path_skel)
    spacing = skel_nii.header['pixdim'][1:4]

    label = np.asanyarray(label_nii.dataobj)
    label = label.astype(np.uint8)
    #label = large_connected_domain(label)


    skeleton = np.asanyarray(skel_nii.dataobj)
    skeleton = skeleton.astype(np.float32)

    skeleton_parse, cd, num = skeleton_parsing(skeleton)
    trachea = loc_trachea_vectorized(cd, num)
    #print("trachea_loc",trachea)
    tree_parsing = tree_parsing1(skeleton_parse, label,cd)


    ad_matric = adjacent_map_cuda(skeleton,skeleton_parse,cd)

    parent_map, children_map, generation = parent_children_map(ad_matric, trachea, num)
    return spacing, parent_map, children_map, generation, trachea, cd, tree_parsing

if __name__ == '__main__':
    path_label, path_skel, idx, save_path = ""
    tree_parse(path_label, path_skel,idx,save_path)




