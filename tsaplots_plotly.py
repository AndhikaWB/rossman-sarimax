import numpy as np
import plotly.graph_objects as go
import statsmodels.tsa.stattools as stattools


def plot_acf(
    series,
    nlags,
    alpha=0.05,
    zero=False,
    pacf=False,
    fig=None,
    row=None,
    col=None,
    **kwargs,
):
    func = stattools.pacf if pacf else stattools.acf
    corr_array = func(series, nlags=nlags, alpha=alpha, **kwargs)

    # Exclude zero lag
    if not zero:
        corr_array = (corr_array[0][1:], corr_array[1][1:])
        fig_range = np.arange(len(corr_array[0])) + 1
    else:
        fig_range = np.arange(len(corr_array[0]))

    # Confidence interval range
    lower_y = corr_array[1][:, 0] - corr_array[0]
    upper_y = corr_array[1][:, 1] - corr_array[0]

    if not fig:
        fig = go.Figure()
        fig_kwargs = dict()
    else:
        assert row and col, "Row and col should be specified"
        fig_kwargs = dict(row=row, col=col)

    # Add the vertical lines
    for x in range(len(corr_array[0])):
        fig.add_scatter(
            x=[x, x] if zero else [x + 1, x + 1],
            y=[0, corr_array[0][x]],
            mode="lines",
            line_color="#3f3f3f",
            **fig_kwargs,
        )

    # Add the dots above vertical lines
    fig.add_scatter(
        x=fig_range,
        y=corr_array[0],
        mode="markers",
        marker_color="#1f77b4",
        **fig_kwargs,
    )

    # Dummy line (required for below)
    fig.add_scatter(
        x=fig_range,
        y=upper_y,
        mode="lines",
        line_color="rgba(255, 255, 255, 0)",
        **fig_kwargs,
    )

    # Fill between 2 confidence ranges
    fig.add_scatter(
        x=fig_range,
        y=lower_y,
        mode="lines",
        line_color="rgba(255, 255, 255, 0)",
        fillcolor="rgba(32, 146, 230, 0.3)",
        fill="tonexty",
        **fig_kwargs,
    )

    return fig


def plot_pacf(
    series, nlags, alpha=0.05, zero=False, fig=None, row=None, col=None, **kwargs
):
    # Use ywm to match the default plot_pacf behavior
    kwargs = {**dict(method="ywm"), **kwargs}

    return plot_acf(
        series,
        nlags,
        alpha=alpha,
        zero=zero,
        pacf=True,
        fig=fig,
        row=row,
        col=col,
        **kwargs,
    )
