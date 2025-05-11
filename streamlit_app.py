import datetime
import streamlit as st
import polars as pl
import numpy as np
import plotly.graph_objects as go
import statsforecast.models as sfm
from statsforecast import StatsForecast

# ----------

# Fourier transform helper function
def cylical_encoding(df: pl.DataFrame, period_terms: dict[str, list[int]]) -> pl.DataFrame:
    for key, (val, terms) in period_terms.items():
        for term in range(1, terms + 1):
            df = df.with_columns(
                pl.col(key).mul(2 * np.pi * term / val).sin().alias(f'{key}_sin{term}'),
                pl.col(key).mul(2 * np.pi * term / val).cos().alias(f'{key}_cos{term}'),
            )

        df = df.drop(key)

    return df

# ----------

st.title('Rossman Store Sales Forecast')

# Combine train and validation data as one
# Since we can customize the period later
df = pl.concat([
    pl.read_csv('data/store_b_clean/train.csv'),
    pl.read_csv('data/store_b_clean/val.csv')
]).with_columns(pl.col('Date').cast(pl.Date))

st.header('Data Preview', divider = True)

use_fourier = st.toggle('Use Fourier transform', value = True)

if use_fourier:
    # Transform the time columns
    df = cylical_encoding(df, {
        'Month': [12, 1], 'DayOfWeek': [7, 1],
        'DayOfMonth': [31, 2], 'DayOfYear': [366, 3]
    })

    # Rearrange the holiday columns to be placed last
    df = df.select(pl.exclude('^.*Holiday.*$'), pl.col('^.*Holiday.*$'))

column_config = {
    'Date': st.column_config.DateColumn(format = 'YYYY/MM/DD'),
    'ds': st.column_config.DateColumn(format = 'YYYY/MM/DD')
}

# Show data preview
st.dataframe(df, column_config = column_config)

st.text(
    'The "Open" and holiday columns represent the number of '
    'stores with that status at that specific date.'
)

# ----------

st.header('Observation Period', divider = True)

st.info(
    'Model forecast will be based on the observation period. '
    'Give the model at least 365 days to observe so it can '
    'make better prediction!'
)

start_date = st.date_input(
    'Select start date:',
    value = df['Date'].min(),
    min_value = df['Date'].min(),
    # Leave at least 30 days for forecast period
    max_value = df['Date'].max() - datetime.timedelta(30),
)

end_date = st.date_input(
    'Select end date:',
    min_value = start_date,
    max_value = df['Date'].max() - datetime.timedelta(30)
)

if (end_date - start_date).days + 1 < 365:
    st.error('Observation period must be at least 365 days!')
    st.stop()
else:
    st.text(f'Observation period is {(end_date - start_date).days + 1} days.')

# ----------

st.header('Forecast Period (Horizon)', divider = True)

st.info(
    'Since forecast result is based on past data, it will '
    'become less accurate the longer you set the forecast '
    'period. 30 days seems reasonable, though you can use '
    'more if you want to.'
)

fstart_date = st.date_input(
    'Start date (a day after observation end date):',
    value = end_date + datetime.timedelta(1),
    disabled = True
)

fend_date = st.date_input(
    'Select end date:',
    # 29 days + the starting day = 30 days
    value = fstart_date + datetime.timedelta(29),
    # End period can be the same day as start period
    # If you only want to forecast for 1 day
    min_value = fstart_date,
    max_value = df['Date'].max()
)

if fend_date == df['Date'].max():
    st.warning(
        f'Maximum date on the dataset is {df['Date'].max().strftime('%Y/%m/%d')}. '
        'Try reducing the observation end date if you want '
        'to forecast more than 30 days. Otherwise, just '
        'ignore this warning.'
    )

st.text(f'Forecast period is {(fend_date - fstart_date).days + 1} days.')

# ----------

st.header('Forecast Result', divider = True)

sel_models = st.segmented_control(
    'Select model(s):',
    options = [
        'SARIMAX (pre-tuned)',
        'MSTL with ETS (auto-tune)'
    ],
    selection_mode = 'multi'
)

roll_forecast = st.toggle('Use rolling forecast (slow)')

if roll_forecast:
    st.warning(
        'Rolling forecast will predict 1 day at a time, and '
        'refit the model each time a new day is added. This '
        'may result better forecast at the cost of time. '
        f'You will be refitting {(fend_date - fstart_date).days + 1} '
        f'times x {len(sel_models)} model(s)!'
    )

only_forecast = st.toggle('Don\'t show observation period in result', value = True)

get_button = st.button(
    'Get Forecast',
    type = 'primary',
    disabled = not sel_models
)

models = []

if 'SARIMAX (pre-tuned)' in sel_models:
    models.append(
        sfm.ARIMA(
            order = (1, 0, 0),
            seasonal_order = (0, 1, 7),
            season_length = 7
        )
    )

if 'MSTL with ETS (auto-tune)' in sel_models:
    models.append(
        sfm.MSTL(season_length = [7, 365])
    )

if get_button:
    # Nixtla format
    df = df.rename({'Date': 'ds', 'Sales': 'y'})
    df = df.with_columns(pl.lit(1).alias('unique_id'))

    fit_df = df.filter(
        pl.col('ds') >= start_date,
        pl.col('ds') <= end_date
    )

    test_df = df.filter(
        pl.col('ds') >= fstart_date,
        pl.col('ds') <= fend_date
    )

    if roll_forecast:
        # Predict N times with 1 day difference
        n_predicts = (fend_date - fstart_date).days + 1
        horizon = 1
    else:
        horizon = (fend_date - fstart_date).days + 1
        n_predicts = 1

    sf = StatsForecast(models = models, freq = '1d', n_jobs = 2)
    results = []

    with st.spinner('Please wait...', show_time = True):
        for i in range(n_predicts):
            res_df = sf.forecast(
                h = horizon,
                df = fit_df,
                X_df = test_df.drop('y').head(horizon)
            )

            results.append(res_df)

            # Modify input on-the-go for rolling forecast
            fit_df = pl.concat([fit_df, test_df.head(horizon)], how = 'vertical')
            test_df = test_df.tail(len(test_df) - horizon)

    # Combine all forecast results (for rolling forecast)
    res_df = pl.concat(results, how = 'vertical')

    if only_forecast:
        # Filter observation period from the graph
        fit_df = fit_df.filter(pl.col('ds').is_in(res_df['ds']))

    fig = go.Figure()
    fig.add_scatter(x = fit_df['ds'], y = fit_df['y'], name = 'True value')

    if 'ARIMA' in res_df.columns:
        fig.add_scatter(x = res_df['ds'], y = res_df['ARIMA'], name = 'SARIMAX')
    if 'MSTL' in res_df.columns:
        fig.add_scatter(x = res_df['ds'], y = res_df['MSTL'], name = 'MSTL (ETS)')

    fig.update_layout(title = 'Result Comparison')
    st.plotly_chart(fig)