import h5py

pfam_infpath = "data/Stage2_MMD_pfam_embedding_last_ckpt_all.hdf5"
swissprot_infpath = "data/Stage2_MMD_swissprot_embedding_last_ckpt_all.hdf5"

N = 8000

pfam_outfpath = f"data/Stage2_MMD_pfam_embedding_subset_{N}.hdf5"
swissprot_outfpath = f"data/Stage2_MMD_swissprot_embedding_subset_{N}.hdf5"


for infpath, outfpath in zip(
    [pfam_infpath, swissprot_infpath], [pfam_outfpath, swissprot_outfpath]
):
    with h5py.File(infpath, "r") as fin, h5py.File(outfpath, "w") as fout:
        in_grp = fin["MMD_data"]

        # create group in output file
        out_grp = fout.create_group("MMD_data")

        for name in in_grp.keys():
            dset = in_grp[name]

            print(f"Copying dataset: {name}, shape={dset.shape}")

            subset = dset[:N]   # first 1000 rows

            out_grp.create_dataset(
                name,
                data=subset,
                dtype=dset.dtype,
                compression="gzip"  # optional but recommended
            )
            
        


# with h5py.File(swissprot_infpath, "r") as f:
#     print(list(f.keys()))  # see datasets
#     dset = f["MMD_data"]  # e.g. shape (1000000, 10)
#     # print(dset.shape)
