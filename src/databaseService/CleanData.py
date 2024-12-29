import pandas as pd
import numpy as np

class CleanData():
    def __init__():
        pass
    
    @staticmethod
    def financial_fiscalDateIncongruence(fin: pd.DataFrame, daysDiscrep: int = 10) -> pd.DataFrame:
        """Combine rows of a financial DataFrame that have fiscalDateEnding within daysDiscrep of each other.

        Args:
            fin (pd.DataFrame): A financial DataFrame with a 'fiscalDateEnding' column.
            daysDiscrep (int, optional): The maximum number of days that two fiscalDateEnding values can differ by to be combined. Defaults to 10.

        Returns:
            pd.DataFrame: The financial DataFrame with rows combined where fiscalDateEnding values are within daysDiscrep of each other.
        """
        # We'll iterate through the rows to find groups of rows whose fiscalDateEnding are within days of e
        combined_rows = []
        skip_indices = set()
        i = 0
        while i < len(fin):
            if i in skip_indices:
                i += 1
                continue
            
            current_row = fin.iloc[i]
            # Look ahead to see if the next row is within days
            if i + 1 < len(fin):
                next_row = fin.iloc[i+1]
                diff_days = (next_row['fiscalDateEnding'] - current_row['fiscalDateEnding']).days

                if diff_days <= daysDiscrep:
                    # We have at least two rows within days. Combine them.
                    # The combination logic: for each column, if current is null and next is not, fill curren
                    combined_dict = {}
                    for col in fin.columns:
                        val_current = current_row[col]
                        val_next = next_row[col]
                        # Prefer the non-null value (if both non-null, keep the first or choose as you like)
                        if pd.isna(val_current) and not pd.isna(val_next):
                            combined_dict[col] = val_next
                        else:
                            combined_dict[col] = val_current

                    combined_rows.append(combined_dict)
                    skip_indices.add(i+1)  # We used the next row to fill in current, so skip it in future
                    i += 2
                else:
                    # No close next row, just keep the current one as is
                    combined_rows.append(current_row.to_dict())
                    i += 1
            else:
                # Last row, no pairs possible
                combined_rows.append(current_row.to_dict())
                i += 1

        # Convert the combined list of dictionaries back to a DataFrame
        res_df = pd.DataFrame(combined_rows)
        
        res_df.columns = fin.columns
        
        return res_df
    
    @staticmethod
    def financial_dropDuplicateYears(fin: pd.DataFrame) -> pd.DataFrame:
        # Drop duplicate years in fin_ann, keep first entry
        if "fiscalDateEnding" in fin.columns:
            fin["year"] = fin["fiscalDateEnding"].dt.year
            fin = fin.drop_duplicates(subset="year", keep="first").drop(columns="year")
        
        return fin
    
    @staticmethod
    def financial_dropLastRow(fin: pd.DataFrame, factor_nulls: float = 0.5) -> pd.DataFrame:
        #if the last row has more null than half of the entries, drop it
        if fin.iloc[-1].isnull().sum() > int(len(fin.columns)*factor_nulls):
            fin = fin.iloc[:-1]
        return fin
            
    
    @staticmethod
    def fill_NAN_to_BusinessDays(s: pd.Series):
        # PRE: s has index as dates. They are sorted.
        # POST: Completes the dates to business days
        #   Adds to nan values a normal random number with mean of the neighbouring prices and sigma half their difference.

        # Ensure the index is a DateTimeIndex
        s.index = pd.to_datetime(s.index)

        # Generate a date range covering all business days between the earliest and latest dates
        date_range = pd.bdate_range(start=s.index.min(), end=s.index.max())

        # Reindex the series to include all business days
        s = s.reindex(date_range)

        # Initialize a list to store indices of missing values
        missing_indices = s[s.isna()].index

        # Iterate over the missing dates to fill them
        for date in missing_indices:
            # Get the position of the current missing date
            idx = s.index.get_loc(date)

            # Find previous valid price
            prev_idx = idx - 1
            while prev_idx >= 0 and pd.isna(s.iloc[prev_idx]):
                prev_idx -= 1

            # Find next valid price
            next_idx = idx + 1
            while next_idx < len(s) and pd.isna(s.iloc[next_idx]):
                next_idx += 1

            # If both previous and next prices are found
            if prev_idx >= 0 and next_idx < len(s):
                prev_price = s.iloc[prev_idx]
                next_price = s.iloc[next_idx]
                mean = (prev_price + next_price) / 2
                sigma = abs(next_price - prev_price) / 2

                # Generate a random value from the normal distribution
                random_value = np.random.normal(mean, sigma)

                # Assign the random value to the missing date
                s.iloc[idx] = random_value if random_value>0 else mean
            else:
                # If only one neighbor is available, fill with that price
                if prev_idx >= 0:
                    s.iloc[idx] = s.iloc[prev_idx]
                elif next_idx < len(s):
                    s.iloc[idx] = s.iloc[next_idx]
                else:
                    # If neither neighbor is available, leave as 0
                    s.iloc[idx] = 0