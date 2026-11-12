# Backblaze Drive Stats dataset and failure labels

Sources:
[1] Backblaze Drive Stats data page. https://www.backblaze.com/cloud-storage/resources/hard-drive-test-data
[2] Backblaze Drive Stats dataset card (Hugging Face). https://huggingface.co/datasets/backblaze/Drive_Stats

## What one row means
Every day Backblaze takes a snapshot of each operational drive, including basic drive information and the SMART statistics it reports; one drive-day is one row, and each day is one CSV file named YYYY-MM-DD.csv [1].

## The failure flag
The failure column contains 0 if the drive is OK and 1 if that is the last day the drive was operational before failing [2]. A drive that disappears from the data without a failure row was removed for other reasons; its outcome after removal is unknown, so this system treats such windows as censored rather than as survivors.

## Schema changes
The columns date, serial_number, model, capacity_bytes and failure never change; the set of SMART attributes can change from quarter to quarter, so loaders must handle different column sets [2]. Absent attributes are stored as missing values, not zeros.

## Terms of use
Backblaze asks users to cite Backblaze as the source of the data, to accept sole responsibility for how they use it, and not to sell the data itself [1][2].
