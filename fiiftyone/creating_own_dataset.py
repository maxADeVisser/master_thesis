import fiftyone as fo

sample = fo.Sample(
    filepath="/Users/newuser/Documents/ITU/master_thesis/data/lung_data/manifest-1725363397135/LIDC-IDRI/LIDC-IDRI-0001/01-01-2000-NA-NA-30178/3000566.000000-NA-03192/1-001.dcm"
)

dataset = fo.Dataset(name="test")
dataset.add_sample(sample)

session = fo.launch_app(dataset)
