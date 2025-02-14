import pandas as pd
import numpy as np
from pandas.api.types import is_numeric_dtype

class CleanData():
    def __init__():
        pass
    
    @staticmethod
    def financial_fiscalDateIncongruence(fin: pd.DataFrame, daysDiscrep: int = 30) -> pd.DataFrame:
        """Combine rows of a financial DataFrame that have fiscalDateEnding within daysDiscrep of each other.

        Args:
            fin (pd.DataFrame): A financial DataFrame with a 'fiscalDateEnding' column.
            daysDiscrep (int, optional): The maximum number of days that two fiscalDateEnding values can differ by to be combined. Defaults to 10.

        Returns:
            pd.DataFrame: The financial DataFrame with rows combined where fiscalDateEnding values are within daysDiscrep of each other.
        """
        def combine_block(block: pd.DataFrame) -> dict:
            out = {}
            for col in fin.columns:
                vals = block[col].dropna()
                if len(vals) == 0:
                    out[col] = None
                elif is_numeric_dtype(block[col]):
                    out[col] = vals.mean()  # average numeric
                else:
                    out[col] = vals.iloc[0]  # first non-null for non-numeric
            return out
    
        groups, start = [], 0
        for i in range(len(fin) - 1):
            diff = (fin.loc[i+1, 'fiscalDateEnding'] - fin.loc[i, 'fiscalDateEnding']).days
            if diff > daysDiscrep:
                groups.append(combine_block(fin.iloc[start:i+1]))
                start = i+1
        # last group
        groups.append(combine_block(fin.iloc[start:]))
    
        return pd.DataFrame(groups, columns=fin.columns)
    
    @staticmethod
    def financial_dropDuplicateYears(fin: pd.DataFrame) -> pd.DataFrame:
        # Drop duplicate years in fin, combine duplicates by:
        #  - taking the mean for numeric columns
        #  - keeping the first (chronologically) row's values for other columns
        if "fiscalDateEnding" in fin.columns:
            # Create a 'year' column from the datetime
            fin["year"] = fin["fiscalDateEnding"].dt.year
            
            # Sort by 'fiscalDateEnding' so "first entry" is chronologically first
            fin = fin.sort_values("fiscalDateEnding")
            
            # Identify numeric columns (for which we'll take the mean)
            numeric_cols = fin.select_dtypes(include=[np.number]).columns
            
            # Group by 'year' and aggregate
            def aggregate_group(group: pd.DataFrame) -> pd.Series:
                # For numeric columns -> mean; for others -> take the first row's value
                return pd.Series({
                    col: group[col].mean(skipna=True) if col in numeric_cols 
                         else group[col].iloc[0] 
                    for col in group.columns
                })
            
            # Apply the aggregation and reset index
            fin_agg = fin.groupby("year", as_index=False).apply(aggregate_group)
            fin_agg.reset_index(drop=True, inplace=True)
            
            return fin_agg
        else:
            # If 'fiscalDateEnding' not found, just return the original DataFrame
            return fin
    
    @staticmethod
    def financial_lastRow_fillWithCompanyOverview_AV(fin: pd.DataFrame, compOverview: pd.DataFrame) -> pd.DataFrame:
        if not len(compOverview) == 1:
            return fin
        
        totalRevenue = pd.to_numeric(compOverview['RevenueTTM'].iloc[0], errors='coerce')
        grossProfit = pd.to_numeric(compOverview['GrossProfitTTM'].iloc[0], errors='coerce')
        EBITDA = pd.to_numeric(compOverview['EBITDA'].iloc[0], errors='coerce')

        if np.isnan(fin.loc[len(fin)-1, 'totalRevenue']):
            fin.loc[len(fin)-1, 'totalRevenue'] = totalRevenue
        if np.isnan(fin.loc[len(fin)-1, 'grossProfit']):
            fin.loc[len(fin)-1, 'grossProfit'] = grossProfit
        if np.isnan(fin.loc[len(fin)-1, 'ebitda']):
            fin.loc[len(fin)-1, 'ebitda'] = EBITDA

        return fin
    
    @staticmethod
    def financial_lastRow_removeIfOutOfFiscal(fin: pd.DataFrame) -> pd.DataFrame:
        # Remove last row if the date difference between last to second is not equal to the difference between second to third and third to fourth
        if len(fin) > 3:
            m1 = fin.iloc[-1]['fiscalDateEnding'].month
            m2 = fin.iloc[-2]['fiscalDateEnding'].month
            d1 = fin.iloc[-1]['fiscalDateEnding'].day
            d2 = fin.iloc[-2]['fiscalDateEnding'].day
            if m1 != m2 and d1 != d2:
                fin = fin.iloc[:-1]

        return fin
    
    @staticmethod
    def financial_lastRow_rmIfNanInKeyValues_AV(fin: pd.DataFrame) -> pd.DataFrame:
        isMissing = False
        isMissing = isMissing | np.isnan(fin.loc[len(fin)-1, 'totalRevenue'])
        isMissing = isMissing | np.isnan(fin.loc[len(fin)-1, 'grossProfit'])
        isMissing = isMissing | np.isnan(fin.loc[len(fin)-1, 'totalAssets'])
        
        if isMissing:
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