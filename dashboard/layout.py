from dash import dcc, html

# ── Colour palette ────────────────────────────────────────────────
NAVY  = "#1B2A4A"
TEAL  = "#0F6E56"
SLATE = "#3D4F6B"
LIGHT = "#F4F6FA"
WHITE = "#FFFFFF"
BORDER = "#C8D2E0"


def create_layout(tickers: list[str]) -> html.Div:
    return html.Div([

        # ── Header ────────────────────────────────────────────────
        html.Div([
            html.H1("VEGA", style={
                "color": WHITE, "margin": "0", "fontSize": "2.2rem",
                "fontWeight": "bold", "letterSpacing": "3px"
            }),
            html.P("Volatility & Earnings Guidance Analyzer", style={
                "color": "#A8BDD4", "margin": "4px 0 0 0", "fontSize": "0.9rem"
            }),
        ], style={
            "background": NAVY, "padding": "20px 32px",
            "borderBottom": f"3px solid {TEAL}"
        }),

        # ── Controls bar ──────────────────────────────────────────
        html.Div([
            html.Div([
                html.Label("Company", style={"fontWeight": "600",
                           "color": SLATE, "marginBottom": "6px",
                           "display": "block", "fontSize": "0.85rem"}),
                dcc.Dropdown(
                    id="ticker-dropdown",
                    options=[{"label": t, "value": t} for t in tickers],
                    value=tickers[0],
                    clearable=False,
                    style={"width": "200px"}
                ),
            ], style={"marginRight": "32px"}),

            html.Div([
                html.Label("Signal (scatter x-axis)", style={
                    "fontWeight": "600", "color": SLATE,
                    "marginBottom": "6px", "display": "block",
                    "fontSize": "0.85rem"
                }),
                dcc.Dropdown(
                    id="signal-dropdown",
                    options=[
                        {"label": "QA Uncertainty",        "value": "qa_uncertainty"},
                        {"label": "Prepared Positive",     "value": "prepared_positive"},
                        {"label": "Specific FLS Ratio",    "value": "specific_fls_ratio"},
                        {"label": "LM Uncertainty",        "value": "lm_uncertainty"},
                        {"label": "LM Extreme Positive",   "value": "lm_extreme_positive"},
                        {"label": "LM Weak Modal",         "value": "lm_weak_modal"},
                    ],
                    value="qa_uncertainty",
                    clearable=False,
                    style={"width": "220px"}
                ),
            ]),
        ], style={
            "display": "flex", "alignItems": "flex-end",
            "padding": "20px 32px", "background": WHITE,
            "borderBottom": f"1px solid {BORDER}",
            "flexWrap": "wrap", "gap": "16px"
        }),

        # ── Main content ──────────────────────────────────────────
        html.Div([

            # ── Row 1: Sentiment timeline + Risk heatmap ──────────
            html.Div([

                # Panel 1 — Sentiment timeline
                html.Div([
                    html.H3("Sentiment Timeline", style={
                        "color": NAVY, "marginBottom": "12px",
                        "fontSize": "1rem", "fontWeight": "600"
                    }),
                    dcc.Graph(id="sentiment-chart",
                              style={"height": "320px"}),
                ], style={
                    "flex": "1", "background": WHITE,
                    "borderRadius": "8px", "padding": "20px",
                    "boxShadow": "0 1px 4px rgba(0,0,0,0.08)",
                    "marginRight": "16px"
                }),

                # Panel 2 — Risk heatmap
                html.Div([
                    html.H3("Risk Signal Heatmap", style={
                        "color": NAVY, "marginBottom": "12px",
                        "fontSize": "1rem", "fontWeight": "600"
                    }),
                    dcc.Graph(id="risk-heatmap",
                              style={"height": "320px"}),
                ], style={
                    "flex": "1", "background": WHITE,
                    "borderRadius": "8px", "padding": "20px",
                    "boxShadow": "0 1px 4px rgba(0,0,0,0.08)",
                }),

            ], style={"display": "flex", "marginBottom": "16px"}),

            # ── Row 2: Signal vs returns scatter + Analyst note ───
            html.Div([

                # Panel 3 — Signal vs returns scatter
                html.Div([
                    html.H3("Signal vs Abnormal Returns", style={
                        "color": NAVY, "marginBottom": "12px",
                        "fontSize": "1rem", "fontWeight": "600"
                    }),
                    dcc.Graph(id="scatter-chart",
                              style={"height": "340px"}),
                ], style={
                    "flex": "1", "background": WHITE,
                    "borderRadius": "8px", "padding": "20px",
                    "boxShadow": "0 1px 4px rgba(0,0,0,0.08)",
                    "marginRight": "16px"
                }),

                # Panel 4 — Analyst note
                html.Div([
                    html.H3("AI Analyst Note", style={
                        "color": NAVY, "marginBottom": "8px",
                        "fontSize": "1rem", "fontWeight": "600"
                    }),
                    html.Div([
                        html.Label("Select earnings date:", style={
                            "fontSize": "0.8rem", "color": SLATE,
                            "marginBottom": "6px", "display": "block"
                        }),
                        dcc.Dropdown(
                            id="date-dropdown",
                            clearable=False,
                            style={"width": "100%", "marginBottom": "12px"}
                        ),
                    ]),
                    html.Div(id="analyst-note", style={
                        "background": LIGHT,
                        "borderRadius": "6px",
                        "padding": "14px",
                        "fontSize": "0.85rem",
                        "lineHeight": "1.6",
                        "color": "#2D3748",
                        "minHeight": "240px",
                        "whiteSpace": "pre-wrap",
                        "overflowY": "auto",
                        "maxHeight": "280px",
                        "border": f"1px solid {BORDER}"
                    }),
                ], style={
                    "flex": "1", "background": WHITE,
                    "borderRadius": "8px", "padding": "20px",
                    "boxShadow": "0 1px 4px rgba(0,0,0,0.08)",
                }),

            ], style={"display": "flex"}),

        ], style={"padding": "20px 32px", "background": LIGHT,
                  "minHeight": "calc(100vh - 160px)"}),

        # ── Footer ────────────────────────────────────────────────
        html.Div([
            html.P("VEGA · Built by Priyanshi Modi · BITS Pilani Goa · 2026", style={
                "color": "#A8BDD4", "margin": "0", "fontSize": "0.8rem"
            }),
        ], style={
            "background": NAVY, "padding": "14px 32px",
            "textAlign": "center", "marginTop": "auto"
        }),

    ], style={"fontFamily": "Arial, sans-serif", "minHeight": "100vh"})