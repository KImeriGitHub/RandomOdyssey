import matplotlib.pyplot as plt
import pandas as pd
from src.common.Portfolio import Portfolio
from src.common.AssetData import AssetData
from typing import List, Dict
import plotly.graph_objs as go
from plotly.subplots import make_subplots

class ResultAnalyzer:
    def __init__(self, portfolio: Portfolio):
        self.portfolio = portfolio

    def plot_portfolio_value(self):
        df = pd.DataFrame(self.portfolio.valueOverTime, columns=["Timestamp", "Value"]).set_index("Timestamp")
        df.plot()
        plt.title('Portfolio Value Over Time')
        plt.xlabel('Date')
        plt.ylabel('Portfolio Value')
        plt.show()

    def plot_positions_per_asset(self, assets: Dict[str, AssetData]):
        # Build a DataFrame from positionsOverTime
        positions_list = []
        for timestamp, positions_dict in self.portfolio.positionsOverTime:
            positions_series = pd.Series(positions_dict, name=pd.to_datetime(timestamp))
            positions_list.append(positions_series)

        positions_df = pd.DataFrame(positions_list).sort_index().fillna(0)

        # Get all dates from positions and share prices
        all_shareprice_dates = set()
        for asset in assets.values():
            all_shareprice_dates.update(asset.shareprice.index)

        # Combine and sort all dates
        all_dates = positions_df.index.union(pd.Index(all_shareprice_dates)).drop_duplicates().sort_values()

        # Reindex positions_df to include all dates and forward-fill positions
        positions_df = positions_df.reindex(all_dates).fillna(method='ffill').fillna(0)

        num_assets = len(positions_df.columns)
        fig, axes = plt.subplots(num_assets, 1, figsize=(12, num_assets * 3), sharex=False)

        if num_assets == 1:
            axes = [axes]  # Ensure axes is iterable

        for ax, asset in zip(axes, positions_df.columns):
            asset_positions = positions_df[asset]

            # Get the share price data
            shareprice = assets[asset].shareprice['Close']
            shareprice = shareprice.reindex(all_dates).fillna(method='ffill')

            df = pd.DataFrame({
                'SharePrice': shareprice,
                'Position': asset_positions,
            }).dropna(subset=['SharePrice'])

            ax.plot(df.index, df['SharePrice'], label=asset)

            # Highlight periods where the position is held
            ax.fill_between(df.index, df['SharePrice'].min(), df['SharePrice'],
                            where=(df['Position'] > 0),
                            color='red', alpha=0.3)

            ax.set_title(f"{asset} Share Price and Position Held")
            ax.legend()

            # Adjust x-axis limits to focus on holding periods plus 50% extra on both sides
            positions_held = df[df['Position'] > 0]
            if not positions_held.empty:
                # X-axis adjustments (same as before)
                min_date = positions_held.index.min()
                max_date = positions_held.index.max()
                position_duration = max_date - min_date

                # Calculate the extra time to extend on both sides
                left_extra = position_duration * 0.5
                right_extra = position_duration * 0.5

                # Calculate new start and end dates
                start_date = min_date - left_extra
                end_date = max_date + right_extra

                # Ensure the dates are within available data
                start_date = max(df.index.min(), start_date)
                end_date = min(df.index.max(), end_date)

                ax.set_xlim([start_date, end_date])

                # Y-axis adjustments
                min_price = positions_held['SharePrice'].min()
                max_price = positions_held['SharePrice'].max()
                price_range = max_price - min_price
                padding = price_range * 0.25  # 25% padding

                # Calculate new y-axis limits
                y_min = max(0, min_price - padding)
                y_max = max_price + padding

                # Ensure y-axis limits are within the share price data range
                overall_min_price = df['SharePrice'].min()
                overall_max_price = df['SharePrice'].max()
                y_min = max(overall_min_price, y_min)
                y_max = min(overall_max_price, y_max)

                ax.set_ylim([y_min, y_max])
            else:
                # Asset was never held; set x-limits and y-limits to the full range
                ax.set_xlim([df.index.min(), df.index.max()])
                ax.set_ylim([df['SharePrice'].min(), df['SharePrice'].max()])

        plt.tight_layout()
        plt.show()

    def plot_positions_per_asset_shared(self, assets: Dict[str, AssetData]):
        # Build a DataFrame from positionsOverTime
        positions_list = []
        for timestamp, positions_dict in self.portfolio.positionsOverTime:
            positions_series = pd.Series(positions_dict, name=pd.to_datetime(timestamp))
            positions_list.append(positions_series)

        positions_df = pd.DataFrame(positions_list).sort_index().fillna(0)

        # Get all dates from positions and share prices
        all_shareprice_dates = set()
        for asset in assets.values():
            all_shareprice_dates.update(asset.shareprice.index)

        # Combine and sort all dates
        all_dates = positions_df.index.union(pd.Index(all_shareprice_dates)).drop_duplicates().sort_values()

        # Reindex positions_df to include all dates and forward-fill positions
        positions_df = positions_df.reindex(all_dates).fillna(method='ffill').fillna(0)

        num_assets = len(positions_df.columns)

        # Determine number of rows and columns for subplots
        n_cols = 3  # Adjust this value as needed
        n_rows = (num_assets + n_cols - 1) // n_cols  # Ceiling division

        fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, n_rows * 4), sharex=True)
        axes = axes.flatten()

        # Calculate the global holding period across all assets
        all_positions_held = pd.DataFrame()
        for asset in positions_df.columns:
            asset_positions = positions_df[asset]
            df_asset = pd.DataFrame({
                'Position': asset_positions,
            })
            positions_held = df_asset[df_asset['Position'] > 0]
            all_positions_held = pd.concat([all_positions_held, positions_held])

        if not all_positions_held.empty:
            global_min_date = all_positions_held.index.min()
            global_max_date = all_positions_held.index.max()
            global_duration = global_max_date - global_min_date

            # Calculate the extra time to extend on both sides (15% padding)
            left_extra = global_duration * 0.15
            right_extra = global_duration * 0.15

            # Calculate new global start and end dates
            global_start_date = global_min_date - left_extra
            global_end_date = global_max_date + right_extra

            # Ensure the dates are within available data
            global_start_date = max(all_dates.min(), global_start_date)
            global_end_date = min(all_dates.max(), global_end_date)
        else:
            # If no positions were held, use the full date range
            global_start_date = all_dates.min()
            global_end_date = all_dates.max()

        # Iterate over each asset to plot
        for i, asset in enumerate(positions_df.columns):
            ax = axes[i]
            asset_positions = positions_df[asset]

            # Get the share price data
            shareprice = assets[asset].shareprice['Close']
            shareprice = shareprice.reindex(all_dates).fillna(method='ffill')

            df = pd.DataFrame({
                'SharePrice': shareprice,
                'Position': asset_positions,
            }).dropna(subset=['SharePrice'])

            ax.plot(df.index, df['SharePrice'], label=asset)

            # Highlight periods where the position is held
            ax.fill_between(df.index, df['SharePrice'].min(), df['SharePrice'],
                            where=(df['Position'] > 0),
                            color='red', alpha=0.3)

            ax.set_title(f"{asset} Share Price and Position Held")
            ax.legend()

            # Set the shared x-axis limits
            ax.set_xlim([global_start_date, global_end_date])

            # Y-axis adjustments per asset
            positions_held = df[df['Position'] > 0]
            if not positions_held.empty:
                min_price = positions_held['SharePrice'].min()
                max_price = positions_held['SharePrice'].max()
                price_range = max_price - min_price
                padding = price_range * 0.25  # 25% padding (can adjust if needed)

                # Calculate new y-axis limits
                y_min = max(0, min_price - padding)
                y_max = max_price + padding

                # Ensure y-axis limits are within the share price data range
                overall_min_price = df['SharePrice'].min()
                overall_max_price = df['SharePrice'].max()
                y_min = max(overall_min_price, y_min)
                y_max = min(overall_max_price, y_max)

                ax.set_ylim([y_min, y_max])
            else:
                # Asset was never held; set y-limits to the full range
                ax.set_ylim([df['SharePrice'].min(), df['SharePrice'].max()])

        # Hide any unused subplots
        for j in range(i + 1, len(axes)):
            fig.delaxes(axes[j])

        plt.tight_layout()
        plt.show()

    def plot_positions_per_asset_separate(self, assets: Dict[str, AssetData]):
        # Build a DataFrame from positionsOverTime
        positions_list = []
        for timestamp, positions_dict in self.portfolio.positionsOverTime:
            positions_series = pd.Series(positions_dict, name=pd.to_datetime(timestamp))
            positions_list.append(positions_series)

        positions_df = pd.DataFrame(positions_list).sort_index().fillna(0)

        # Get all dates from positions and share prices
        all_shareprice_dates = set()
        for asset in assets.values():
            all_shareprice_dates.update(asset.shareprice.index)

        # Combine and sort all dates
        all_dates = positions_df.index.union(pd.Index(all_shareprice_dates)).drop_duplicates().sort_values()

        # Reindex positions_df to include all dates and forward-fill positions
        positions_df = positions_df.reindex(all_dates).fillna(method='ffill').fillna(0)

        # Calculate the global holding period across all assets
        all_positions_held = pd.DataFrame()
        for asset in positions_df.columns:
            asset_positions = positions_df[asset]
            df_asset = pd.DataFrame({
                'Position': asset_positions,
            })
            positions_held = df_asset[df_asset['Position'] > 0]
            all_positions_held = pd.concat([all_positions_held, positions_held])

        if not all_positions_held.empty:
            global_min_date = all_positions_held.index.min()
            global_max_date = all_positions_held.index.max()
            global_duration = global_max_date - global_min_date

            # Calculate the extra time to extend on both sides (15% padding)
            left_extra = global_duration * 0.15
            right_extra = global_duration * 0.15

            # Calculate new global start and end dates
            global_start_date = global_min_date - left_extra
            global_end_date = global_max_date + right_extra

            # Ensure the dates are within available data
            global_start_date = max(all_dates.min(), global_start_date)
            global_end_date = min(all_dates.max(), global_end_date)
        else:
            # If no positions were held, use the full date range
            global_start_date = all_dates.min()
            global_end_date = all_dates.max()

        # Iterate over each asset to plot
        for asset in positions_df.columns:
            # Create a new figure and axis for each asset
            fig, ax = plt.subplots(figsize=(10, 6))

            asset_positions = positions_df[asset]

            # Get the share price data
            shareprice = assets[asset].shareprice['Close']
            shareprice = shareprice.reindex(all_dates).fillna(method='ffill')

            df = pd.DataFrame({
                'SharePrice': shareprice,
                'Position': asset_positions,
            }).dropna(subset=['SharePrice'])

            ax.plot(df.index, df['SharePrice'], label=asset)

            # Highlight periods where the position is held
            ax.fill_between(df.index, df['SharePrice'].min(), df['SharePrice'],
                            where=(df['Position'] > 0),
                            color='red', alpha=0.3)

            ax.set_title(f"{asset} Share Price and Position Held")
            ax.legend()

            # Set the x-axis limits
            ax.set_xlim([global_start_date, global_end_date])

            # Y-axis adjustments per asset
            positions_held = df[df['Position'] > 0]
            if not positions_held.empty:
                min_price = positions_held['SharePrice'].min()
                max_price = positions_held['SharePrice'].max()
                price_range = max_price - min_price
                padding = price_range * 0.25  # 25% padding (can adjust if needed)

                # Calculate new y-axis limits
                y_min = max(0, min_price - padding)
                y_max = max_price + padding

                # Ensure y-axis limits are within the share price data range
                overall_min_price = df['SharePrice'].min()
                overall_max_price = df['SharePrice'].max()
                y_min = max(overall_min_price, y_min)
                y_max = min(overall_max_price, y_max)

                ax.set_ylim([y_min, y_max])
            else:
                # Asset was never held; set y-limits to the full range
                ax.set_ylim([df['SharePrice'].min(), df['SharePrice'].max()])

            plt.tight_layout()
        
        plt.show()

    def plot_positions_per_asset_separate_test(self, assets: Dict[str, AssetData]):
        # Build a DataFrame from positionsOverTime
        positions_list = []
        for timestamp, positions_dict in self.portfolio.positionsOverTime:
            positions_series = pd.Series(positions_dict, name=pd.to_datetime(timestamp))
            positions_list.append(positions_series)

        positions_df = pd.DataFrame(positions_list).sort_index().fillna(0)

        # Get all dates from positions and share prices
        all_shareprice_dates = set()
        for asset in assets.values():
            all_shareprice_dates.update(asset.shareprice.index)

        # Combine and sort all dates
        all_dates = positions_df.index.union(pd.Index(all_shareprice_dates)).drop_duplicates().sort_values()

        # Reindex positions_df to include all dates and forward-fill positions
        positions_df = positions_df.reindex(all_dates).fillna(method='ffill').fillna(0)

        # Iterate over each asset to plot
        for asset in positions_df.columns:
            # Create a new figure and axis for each asset
            fig, ax = plt.subplots(figsize=(10, 6))

            asset_positions = positions_df[asset]

            # Get the share price data
            shareprice = assets[asset].shareprice['Close']
            shareprice = shareprice.reindex(all_dates).fillna(method='ffill')

            df = pd.DataFrame({
                'SharePrice': shareprice,
                'Position': asset_positions,
            }).dropna(subset=['SharePrice'])

            ax.plot(df.index, df['SharePrice'], label=asset)

            # Highlight periods where the position is held
            ax.fill_between(df.index, df['SharePrice'].min(), df['SharePrice'],
                            where=(df['Position'] > 0),
                            color='red', alpha=0.3)

            ax.set_title(f"{asset} Share Price and Position Held")
            ax.legend()

            # Now calculate per asset start and end dates
            positions_held = df[df['Position'] > 0]
            if not positions_held.empty:
                asset_min_date = positions_held.index.min()
                asset_max_date = positions_held.index.max()
                asset_duration = asset_max_date - asset_min_date

                # Calculate the extra time to extend on both sides
                six_months = pd.Timedelta(days=182.5)
                left_extra = max(asset_duration * 0.15, six_months)
                right_extra = asset_duration * 0.15  # Right padding remains at 15%

                # Calculate new start and end dates for this asset
                asset_start_date = asset_min_date - left_extra
                asset_end_date = asset_max_date + right_extra

                # Ensure the dates are within available data
                asset_start_date = max(df.index.min(), asset_start_date)
                asset_end_date = min(df.index.max(), asset_end_date)
            else:
                # If asset was never held, use the full date range
                asset_start_date = df.index.min()
                asset_end_date = df.index.max()

            # Set the x-axis limits per asset
            ax.set_xlim([asset_start_date, asset_end_date])

            # Y-axis adjustments per asset
            if not positions_held.empty:
                min_price = positions_held['SharePrice'].min()
                max_price = positions_held['SharePrice'].max()
                price_range = max_price - min_price
                padding = price_range * 0.25  # 25% padding (can adjust if needed)

                # Calculate new y-axis limits
                y_min = max(0, min_price - padding)
                y_max = max_price + padding

                # Ensure y-axis limits are within the share price data range
                overall_min_price = df['SharePrice'].min()
                overall_max_price = df['SharePrice'].max()
                y_min = max(overall_min_price, y_min)
                y_max = min(overall_max_price, y_max)

                ax.set_ylim([y_min, y_max])
            else:
                # Asset was never held; set y-limits to the full range
                ax.set_ylim([df['SharePrice'].min(), df['SharePrice'].max()])

            plt.tight_layout()

        plt.show()

