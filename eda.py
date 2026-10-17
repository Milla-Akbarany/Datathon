import pandas as pd

for d in ["A", "B", "C"]:
    path = f"Dataset/dataset_{d}/"
    train = pd.read_csv(path + "train.csv")
    test = pd.read_csv(path + "test.csv")
    print("=" * 60)
    print(f"DATASET {d}: train {train.shape}, test {test.shape}")
    print(train.head())
    print("\nTipe kolom:")
    print(train.dtypes.value_counts())
    print("\nKolom non-numerik:", train.select_dtypes(exclude="number").columns.tolist())
    print("Total missing di train:", int(train.isna().sum().sum()))
    print("Kolom train yang tidak ada di test:", set(train.columns) - set(test.columns))
    if "target" in train.columns:
        print("\nDistribusi kelas:")
        print(train["target"].value_counts(normalize=True).sort_index())