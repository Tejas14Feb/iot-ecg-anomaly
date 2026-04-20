import wfdb
import numpy as np
import pandas as pd
import os

# Download a subset of PTB-XL records
# Full dataset is large so we grab 500 records to keep it manageable

print("Starting PTB-XL data download...")

save_dir = "data/ptbxl_records"
os.makedirs(save_dir, exist_ok=True)

records = []
labels = []

# PTB-XL records are numbered 00001 to 21837
# We grab the first 500 for speed
target = 500
count = 0
record_num = 1

while count < target and record_num <= 21837:
    try:
        rid = str(record_num).zfill(5)
        folder = str((record_num - 1) // 1000 + 1).zfill(2) + "000"
        path = "ptb-xl/records100/" + folder + "/" + rid + "_lr"

        record = wfdb.rdrecord(path, pn_dir="ptb-xl/1.0.3/")
        annotation = wfdb.rdheader(path, pn_dir="ptb-xl/1.0.3/")

        signal = record.p_signal  # shape: (1000, 12) at 100Hz

        # Use lead II (index 1) which is standard for cardiac monitoring
        lead2 = signal[:, 1]

        # Label: check comments for NORM
        comments = " ".join(annotation.comments).upper()
        if "NORM" in comments:
            label = 0  # normal
        else:
            label = 1  # abnormal

        records.append(lead2)
        labels.append(label)
        count = count + 1

        if count % 50 == 0:
            print("Downloaded", count, "records so far...")

    except Exception as e:
        pass  # skip records that fail to download

    record_num = record_num + 1

records = np.array(records)
labels = np.array(labels)

np.save("data/ecg_signals.npy", records)
np.save("data/ecg_labels.npy", labels)

normal_count = int(np.sum(labels == 0))
abnormal_count = int(np.sum(labels == 1))

print("Done! Saved", len(records), "records")
print("Normal:", normal_count, "  Abnormal:", abnormal_count)
