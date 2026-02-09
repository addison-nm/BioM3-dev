import h5py

N = 1000
pfam_infpath = f"data/Stage2_MMD_pfam_embedding_subset_{N}.hdf5"
swissprot_infpath = f"data/Stage2_MMD_swissprot_embedding_subset_{N}.hdf5"



for infpath in [pfam_infpath, swissprot_infpath]:
    with h5py.File(infpath, "r") as fin:
        in_grp = fin["MMD_data"]

        for name in in_grp.keys():
            dset = in_grp[name]

            print(f"Dataset: {name}, shape={dset.shape}")
            print(dset[0:10])

        


# with h5py.File(swissprot_infpath, "r") as f:
#     print(list(f.keys()))  # see datasets
#     dset = f["MMD_data"]  # e.g. shape (1000000, 10)
#     # print(dset.shape)
