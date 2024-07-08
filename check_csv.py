import os

import pandas as pd

# Initialize a dictionary to store min and max values for each column
from numpy import mean

columns_values = {}

if __name__ == "__main__":
    # Check all csv files in the directory and keep track of min and max in all columns
    # We will use this to normalize the data
    path_csv_files = "data/train/v2-nov-dec/nodes/"

    # For each CSV file
    for filename in os.listdir(path_csv_files):
        if filename.endswith(".csv"):
            print("Reading file: ", filename)
            df = pd.read_csv(path_csv_files + filename)

            # Keep track of min, max and avg values across all dataframes
            # Iterate through columns in the DataFrame
            for column in df.columns:
                # Update min, avg, max values for each column
                col_min = df[column].min()
                col_max = df[column].max()
                # col_avg = 0
                # if column != 'ts':
                    # col_avg = df[column].mean()

                if column not in columns_values:
                    columns_values[column] = {'min': col_min, 'max': col_max} # 'avg': col_avg}
                else:
                    columns_values[column]['min'] = min(columns_values[column]['min'], col_min)
                    columns_values[column]['max'] = max(columns_values[column]['max'], col_max)
                    # columns_values[column]['avg'] = mean(columns_values[column]['avg'], col_avg)

            # check if any column has NaN values
            print("Columns with NaN values: ", df.columns[df.isna().any()].tolist())
            print("\n")

    # Print the min and max values for each column
    for column, values in columns_values.items():
        print(f"Column: {column}")
        print(f"  Min Value: {values['min']}")
        print(f"  Max Value: {values['max']}")
        # print(f"  Avg Value: {values['avg']}")

        # print("Min values: ", df.min())
        # print("Max values: ", df.max())
        print("\n")
