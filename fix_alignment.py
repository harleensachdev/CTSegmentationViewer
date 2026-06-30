
import argparse
import nibabel as nib
import numpy as np
from scipy.ndimage import map_coordinates


def resample_seg_to_ct_grid(ct_img: nib.Nifti1Image, seg_img: nib.Nifti1Image) -> nib.Nifti1Image:

    ct_shape = np.array(ct_img.shape, dtype=float)
    seg_shape = np.array(seg_img.shape, dtype=float)

    if seg_shape.shape[0] != 3 or len(ct_img.shape) != 3:
        raise ValueError("3D volumes were expected")

    scale = seg_shape / ct_shape  

    seg_data = seg_img.get_fdata()

    # CT GRID COORDINATE MESH
    gx, gy, gz = np.meshgrid(
        np.arange(ct_img.shape[0]),
        np.arange(ct_img.shape[1]),
        np.arange(ct_img.shape[2]),
        indexing="ij",
    )
    coords = np.stack([gx * scale[0], gy * scale[1], gz * scale[2]], axis=0)

    resampled = map_coordinates(seg_data, coords, order=0, mode="constant", cval=0)
    resampled = resampled.astype(np.uint8)

    out_img = nib.Nifti1Image(resampled, affine=ct_img.affine, header=ct_img.header)
    out_img.header.set_data_dtype(np.uint8)
    return out_img


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ct", required=True, help="Path to original ct.nii.gz")
    parser.add_argument("--seg", required=True, help="Path to combined_labels.nii.gz from SuPreM")
    parser.add_argument("--out", required=True, help="Output path for the aligned segmentation")
    args = parser.parse_args()

    ct_img = nib.load(args.ct)
    seg_img = nib.load(args.seg)
    
    out_img = resample_seg_to_ct_grid(ct_img, seg_img)
    nib.save(out_img, args.out)
if __name__ == "__main__":
    main()