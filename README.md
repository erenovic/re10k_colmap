# Re10K Dataset COLMAP Reconstruction using Known Parameters

This directory contains the COLMAP reconstruction method for the Re10K dataset using known parameters. The known parameters are the camera intrinsics and the camera poses.

## Instructions
```bash
python sparse_reconstruct_re10k.py --scene_id all --data_path ../dataset_subsets/re10k_subset/test
```

For sparse reconstruction, you can use either `all` or specific `scene_id` for Re10K scenes. The `data_path` should point to the directory containing the Re10K dataset.

For the dataset, this repository uses the compressed format with `.torch` extension. A subset of the dataset can be downloaded from [here](https://drive.google.com/drive/folders/1joiezNCyQK2BvWMnfwHJpm2V77c7iYGe?usp=sharing).

For further information on dataset, please refer to [Re10K documentation](https://google.github.io/realestate10k/).

### Dataset structure

The dataset should be structured as follows:
```
re10k_subset
├── test
│   ├── 000000.torch
│   ├── 000001.torch
│   ├── 000002.torch
│   └── index.json
└── train
    ├── 000000.torch
    ├── 000001.torch
    ├── 000002.torch
    └── index.json
```

Note again that the actual dataset is huge and this tree only contains the subset provided by [pixelSplat](https://davidcharatan.com/pixelsplat/)