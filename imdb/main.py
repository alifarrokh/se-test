from imdb import ImdbDataset, download_ds

# download_ds()
print(ImdbDataset(ds_split="train")[0])