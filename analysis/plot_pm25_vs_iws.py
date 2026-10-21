"""
Plot PM2.5 against the cumulated wind speed (Iws) at three time scales.

Creates, under output_plots/:
    - monthly_averages/: monthly averages for each year
    - daily_averages/: daily averages for each month of each year
    - hourly_values/: hourly values for each day
"""

import os

import matplotlib.pyplot as plt
import pandas as pd


def _load_data(
    file_path: str
) -> pd.DataFrame:
    """
    Load the dataset and drop the rows without pm2.5 or Iws.

    Args:
        file_path (str): Path to the original hourly CSV file.

    Returns:
        pd.DataFrame: The loaded dataset.
    """
    df = pd.read_csv(file_path)
    df = df.dropna(subset=['pm2.5', 'Iws'])
    return df


def _create_directory(
    path: str
) -> None:
    """
    Create a directory if it does not exist.

    Args:
        path (str): Path of the directory.

    Returns:
        None
    """
    if not os.path.exists(path):
        os.makedirs(path)


def _plot_monthly_averages(
    df: pd.DataFrame,
    output_dir: str
) -> None:
    """
    Plot the monthly averages of pm2.5 and Iws for each year, with Iws
    on a secondary y-axis.

    Args:
        df (pd.DataFrame): Hourly dataset.
        output_dir (str): Directory where the plots are saved.

    Returns:
        None
    """
    _create_directory(output_dir)
    for year in df['year'].unique():
        yearly_data = df[df['year'] == year]
        monthly_avg = yearly_data.groupby('month')[['pm2.5', 'Iws']].mean()

        fig, ax1 = plt.subplots()

        # pm2.5 on primary y-axis
        ax1.plot(
            monthly_avg.index, monthly_avg['pm2.5'], label='pm2.5',
            marker='o', color='blue'
        )
        ax1.set_xlabel('Month')
        ax1.set_ylabel('pm2.5', color='blue')
        ax1.tick_params(axis='y', labelcolor='blue')

        # Iws on secondary y-axis
        ax2 = ax1.twinx()
        ax2.plot(
            monthly_avg.index, monthly_avg['Iws'], label='Iws',
            marker='o', color='green'
        )
        ax2.set_ylabel('Iws', color='green')
        ax2.tick_params(axis='y', labelcolor='green')

        fig.suptitle(f'Monthly Averages for {year}')
        fig.legend(loc="upper right")
        plt.grid()
        plt.savefig(os.path.join(output_dir, f'monthly_averages_{year}.png'))
        plt.close()


def _plot_daily_averages(
    df: pd.DataFrame,
    output_dir: str
) -> None:
    """
    Plot the daily averages of pm2.5 and Iws for each month of each
    year, with Iws on a secondary y-axis.

    Args:
        df (pd.DataFrame): Hourly dataset.
        output_dir (str): Directory where the plots are saved, in one
            subfolder per year and month.

    Returns:
        None
    """
    for year in df['year'].unique():
        yearly_data = df[df['year'] == year]
        for month in yearly_data['month'].unique():
            monthly_data = yearly_data[yearly_data['month'] == month]
            daily_avg = monthly_data.groupby('day')[['pm2.5', 'Iws']].mean()

            month_dir = os.path.join(output_dir, str(year), f'month_{month}')
            _create_directory(month_dir)

            fig, ax1 = plt.subplots()

            ax1.plot(
                daily_avg.index, daily_avg['pm2.5'], label='pm2.5',
                marker='o', color='blue'
            )
            ax1.set_xlabel('Day')
            ax1.set_ylabel('pm2.5', color='blue')
            ax1.tick_params(axis='y', labelcolor='blue')

            ax2 = ax1.twinx()
            ax2.plot(
                daily_avg.index, daily_avg['Iws'], label='Iws',
                marker='o', color='green'
            )
            ax2.set_ylabel('Iws', color='green')
            ax2.tick_params(axis='y', labelcolor='green')

            fig.suptitle(f'Daily Averages for {year}-{month:02d}')
            fig.legend(loc="upper right")
            plt.grid()
            plt.savefig(os.path.join(
                month_dir, f'daily_averages_{year}_{month:02d}.png'
            ))
            plt.close()


def _plot_hourly_values(
    df: pd.DataFrame,
    output_dir: str
) -> None:
    """
    Plot the hourly values of pm2.5 and Iws for each day, with Iws on a
    secondary y-axis.

    Args:
        df (pd.DataFrame): Hourly dataset.
        output_dir (str): Directory where the plots are saved, in one
            subfolder per year, month and day.

    Returns:
        None
    """
    for year in df['year'].unique():
        yearly_data = df[df['year'] == year]
        for month in yearly_data['month'].unique():
            monthly_data = yearly_data[yearly_data['month'] == month]
            for day in monthly_data['day'].unique():
                daily_data = monthly_data[monthly_data['day'] == day]
                hourly_values = (
                    daily_data.groupby('hour')[['pm2.5', 'Iws']].mean()
                )

                day_dir = os.path.join(
                    output_dir, str(year), f'month_{month}', f'day_{day}'
                )
                _create_directory(day_dir)

                fig, ax1 = plt.subplots()

                ax1.plot(
                    hourly_values.index, hourly_values['pm2.5'],
                    label='pm2.5', marker='o', color='blue'
                )
                ax1.set_xlabel('Hour')
                ax1.set_ylabel('pm2.5', color='blue')
                ax1.tick_params(axis='y', labelcolor='blue')

                ax2 = ax1.twinx()
                ax2.plot(
                    hourly_values.index, hourly_values['Iws'],
                    label='Iws', marker='o', color='green'
                )
                ax2.set_ylabel('Iws', color='green')
                ax2.tick_params(axis='y', labelcolor='green')

                fig.suptitle(
                    f'Hourly Values for {year}-{month:02d}-{day:02d}'
                )
                fig.legend(loc="upper right")
                plt.grid()
                plt.savefig(os.path.join(
                    day_dir,
                    f'hourly_values_{year}_{month:02d}_{day:02d}.png'
                ))
                plt.close()


def main() -> None:
    """
    Create the monthly, daily and hourly plots.

    Returns:
        None
    """
    input_file = "PRSA_data_2010.1.1-2014.12.31.csv"
    output_dir = "output_plots"

    data = _load_data(input_file)

    _plot_monthly_averages(data, os.path.join(output_dir, "monthly_averages"))
    _plot_daily_averages(data, os.path.join(output_dir, "daily_averages"))
    _plot_hourly_values(data, os.path.join(output_dir, "hourly_values"))


if __name__ == "__main__":
    main()
