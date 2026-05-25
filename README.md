# CariSurg-portfolio
This notebook demonstrates basic data exploration and cleaning techniques using Python and Pandas on an emergency triage dataset. The primary focus of this session was to set up the Colab environment and clean the Gender column.

Tasks Completed:
Google Drive Mounting & Python Version Check: Ensured the Colab environment was correctly set up and connected to Google Drive.
Dataset Loading: Loaded the EmergencyTriageDataset_Reduced_Dirty.csv into a Pandas DataFrame.
Data Inspection: Performed initial checks on the dataset's shape, column types, and unique values within the Gender column.
Gender Column Cleaning: Standardized the Gender column by mapping various representations ('Male', 'MALE', '1', 'Female', 'FEMALE', '0') to a numerical format (1 for Male, 0 for Female).
Column Refinement: Replaced the original Gender column with the newly cleaned version.
