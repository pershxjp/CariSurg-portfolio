# CariSurg-portfolio
# Day 1

This notebook demonstrates basic data exploration and cleaning techniques using Python and Pandas on an emergency triage dataset. The primary focus of this session was to set up the Colab environment and clean the Gender column.

Tasks Completed:
Google Drive Mounting & Python Version Check: Ensured the Colab environment was correctly set up and connected to Google Drive.
Dataset Loading: Loaded the EmergencyTriageDataset_Reduced_Dirty.csv into a Pandas DataFrame.
Data Inspection: Performed initial checks on the dataset's shape, column types, and unique values within the Gender column.
Gender Column Cleaning: Standardized the Gender column by mapping various representations ('Male', 'MALE', '1', 'Female', 'FEMALE', '0') to a numerical format (1 for Male, 0 for Female).
Column Refinement: Replaced the original Gender column with the newly cleaned version.

# Day 2
Advanced cleaning techniques on the emergency triage dataset — understanding clinical meaning before deciding how to clean.
What's covered

Pandas inspection commands + clinical reference ranges
Type conversion, outlier detection, and imputation (mean/median/mode)
Unit inconsistency handling (e.g. °C vs °F in Temp)

Breakout Challenge
Each group cleans one assigned column (DBP, pulse, RR, Fio2, or MAP) and presents their approach.

# Day 3

Turning the cleaned triage dataset into clinically meaningful plots. The rule: every plot must answer a specific clinical question — not just look nice.

What's covered
Bar plots, histograms, scatter plots, and box plots with matplotlib
Adding clinical reference lines (e.g. GCS ≤ 8, bradycardia/tachycardia thresholds)
Saving and sharing plots
