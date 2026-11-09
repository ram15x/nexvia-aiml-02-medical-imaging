import numpy as np

from segment_spleen import keep_largest_component


def test_keeps_only_the_biggest_blob():
    mask = np.zeros((20, 20, 20), dtype=bool)
    mask[2:10, 2:10, 2:10] = True       # 512 voxels
    mask[15:17, 15:17, 15:17] = True    # 8 voxels, disconnected
    cleaned, blobs = keep_largest_component(mask)
    assert blobs == 2
    assert int(cleaned.sum()) == 512
    assert not cleaned[15:17, 15:17, 15:17].any()


def test_single_blob_and_empty_masks_are_unchanged():
    one = np.zeros((10, 10, 10), dtype=bool)
    one[1:4, 1:4, 1:4] = True
    cleaned, blobs = keep_largest_component(one)
    assert blobs == 1 and np.array_equal(cleaned, one)
    empty = np.zeros((5, 5, 5), dtype=bool)
    cleaned, blobs = keep_largest_component(empty)
    assert blobs == 0 and not cleaned.any()
