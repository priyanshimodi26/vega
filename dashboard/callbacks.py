import json
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from dash import Input, Output, html
from scipy import stats

# ── Colour palette ─────────────────────────────────────────────────
NAVY  = "#1B2A4A"
TEAL  = "#0F6E56"
SLATE = "#3D4F6B"


def register_callbacks(app, DF, NARRATIVES, TICKERS,
                       RISK_CATEGORIES, RISK_LABELS):

    # ── Update date dropdown when ticker changes ───────────────────
    @app.callback(
        Output("date-dropdown", "options"),
        Output("date-dropdown", "value"),
        Input("ticker-dropdown", "value"),
    )
    def update_date_dropdown(ticker):
        dates = DF[DF["ticker"] == ticker]["earnings_date"].sort_values()
        options = [{"label": str(d.date()), "value": str(d.date())}
                   for d in dates]
        return options, options[-1]["value"] if options else None

    # ── Panel 1: Sentiment timeline ────────────────────────────────
    @app.callback(
        Output("sentiment-chart", "figure"),
        Input("ticker-dropdown", "value"),
    )
    def update_sentiment_chart(ticker):
        d = DF[DF["ticker"] == ticker].sort_values("earnings_date")

        fig = go.Figure()

        fig.add_trace(go.Scatter(
            x=d["earnings_date"], y=d["prepared_positive"] * 100,
            name="Prep Positive %", mode="lines+markers",
            line=dict(color=TEAL, width=2),
            marker=dict(size=6),
        ))
        fig.add_trace(go.Scatter(
            x=d["earnings_date"], y=d["qa_uncertainty"] * 100,
            name="QA Uncertainty %", mode="lines+markers",
            line=dict(color="#E06B2D", width=2, dash="dot"),
            marker=dict(size=6),
        ))
        fig.add_trace(go.Scatter(
            x=d["earnings_date"], y=d["ar_3d"] * 100,
            name="ar_3d %", mode="lines+markers",
            line=dict(color=NAVY, width=1.5, dash="dash"),
            marker=dict(size=5),
            yaxis="y2",
        ))

        # Vertical lines for earnings dates
        for dt in d["earnings_date"]:
            fig.add_vline(x=dt, line_width=0.5,
                          line_dash="dot", line_color="#C8D2E0")

        fig.update_layout(
            yaxis=dict(title="Score (%)", ticksuffix="%"),
            yaxis2=dict(title="ar_3d (%)", overlaying="y",
                        side="right", ticksuffix="%",
                        showgrid=False),
            legend=dict(orientation="h", y=-0.2),
            margin=dict(l=40, r=40, t=20, b=60),
            plot_bgcolor="white",
            paper_bgcolor="white",
            hovermode="x unified",
        )
        return fig

    # ── Panel 2: Risk heatmap ──────────────────────────────────────
    @app.callback(
        Output("risk-heatmap", "figure"),
        Input("ticker-dropdown", "value"),
    )
    def update_risk_heatmap(ticker):
        d = DF[DF["ticker"] == ticker].sort_values("earnings_date")
        dates = [str(dt.date()) for dt in d["earnings_date"]]
        labels = [RISK_LABELS[c] for c in RISK_CATEGORIES]
        z = d[RISK_CATEGORIES].T.values.tolist()

        fig = go.Figure(go.Heatmap(
            z=z,
            x=dates,
            y=labels,
            colorscale=[
                [0.0, "#F0F4FA"],
                [0.4, "#A8C7E8"],
                [0.7, "#3A7FC1"],
                [1.0, "#1B2A4A"],
            ],
            showscale=True,
            hovertemplate="Date: %{x}<br>Risk: %{y}<br>Score: %{z:.3f}<extra></extra>",
        ))

        fig.update_layout(
            margin=dict(l=140, r=20, t=20, b=60),
            plot_bgcolor="white",
            paper_bgcolor="white",
            xaxis=dict(tickangle=-45, tickfont=dict(size=10)),
            yaxis=dict(tickfont=dict(size=10)),
        )
        return fig

    # ── Panel 3: Signal vs returns scatter ─────────────────────────
    @app.callback(
        Output("scatter-chart", "figure"),
        Input("signal-dropdown", "value"),
    )
    def update_scatter(signal):
        clean = DF[["ticker", signal, "ar_3d"]].dropna()

        fig = go.Figure()

        # One trace per ticker
        colors = px.colors.qualitative.Set2
        for i, ticker in enumerate(TICKERS):
            td = clean[clean["ticker"] == ticker]
            if td.empty:
                continue
            fig.add_trace(go.Scatter(
                x=td[signal], y=td["ar_3d"] * 100,
                mode="markers",
                name=ticker,
                marker=dict(size=7, color=colors[i % len(colors)],
                            opacity=0.8),
                hovertemplate=(
                    f"<b>{ticker}</b><br>"
                    f"{signal}: %{{x:.3f}}<br>"
                    "ar_3d: %{y:.2f}%<extra></extra>"
                ),
            ))

        # Regression line
        x_vals = clean[signal].values
        y_vals = clean["ar_3d"].values * 100
        if len(x_vals) > 5:
            slope, intercept, r, p, _ = stats.linregress(x_vals, y_vals)
            x_line = np.linspace(x_vals.min(), x_vals.max(), 100)
            y_line = slope * x_line + intercept
            fig.add_trace(go.Scatter(
                x=x_line, y=y_line,
                mode="lines", name=f"OLS (R={r:.3f})",
                line=dict(color=NAVY, width=2, dash="dash"),
                showlegend=True,
            ))
            fig.add_annotation(
                x=0.02, y=0.97, xref="paper", yref="paper",
                text=f"R={r:.3f}  p={p:.3f}",
                showarrow=False,
                font=dict(size=11, color=NAVY),
                bgcolor="white",
                bordercolor=NAVY,
                borderwidth=1,
            )

        fig.update_layout(
            xaxis_title=signal,
            yaxis_title="3-Day Abnormal Return (%)",
            yaxis_ticksuffix="%",
            legend=dict(orientation="v", x=1.01),
            margin=dict(l=50, r=120, t=20, b=50),
            plot_bgcolor="white",
            paper_bgcolor="white",
            hovermode="closest",
        )
        return fig

    # ── Panel 4: Analyst note ──────────────────────────────────────
    @app.callback(
        Output("analyst-note", "children"),
        Input("ticker-dropdown", "value"),
        Input("date-dropdown", "value"),
    )
    def update_analyst_note(ticker, date):
        if not ticker or not date:
            return "Select a company and date to view the analyst note."

        key = f"{ticker}_{date}"
        narrative = NARRATIVES.get(key)

        if narrative:
            return narrative
        else:
            return (
                f"No analyst note available for {ticker} on {date}.\n\n"
                "Run: python models/narrative_gen.py to generate notes "
                "(subject to Gemini API daily quota)."
            )