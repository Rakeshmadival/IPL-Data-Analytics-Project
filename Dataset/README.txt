DATASET SETUP
=============

This project requires two CSV files:

1. matches.csv
   - One row represents one IPL match.
   - Expected shape in the current notebook: 1,095 rows x 20 columns.

2. deliveries.csv
   - One row represents one ball-by-ball delivery.
   - Expected shape in the current notebook: 260,920 rows x 17 columns.

The files are linked by:
    matches.id = deliveries.match_id

IMPORTANT:
The raw CSV files are not included in this repository. Download them from the same source used for the project and check that the source permits the intended use and redistribution.

Record the exact source details here before publishing:
- Dataset name:
- Original download URL:
- Dataset publisher/uploader:
- Licence or terms of use:
- Download date:

Do not guess or invent the source or licence.

GOOGLE COLAB SETUP
------------------
Place both files in this Google Drive folder:

MyDrive/IPL_Data_Analytics/Data/raw/
    matches.csv
    deliveries.csv

The current notebook uses this path:
    /content/drive/MyDrive/IPL_Data_Analytics/Data/raw

If you use a different folder, update DATA_PATH in the notebook's dataset-loading cell.

LOCAL SETUP
-----------
Place both CSV files in one local folder and update DATA_PATH in the notebook to that folder.

EXPECTED CLEANED OUTPUTS
------------------------
The notebook creates cleaned datasets and a saved model during execution. These files can be regenerated and do not need to be committed to Git.
