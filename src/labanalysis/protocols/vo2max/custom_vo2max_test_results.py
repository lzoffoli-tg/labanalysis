"""Custom VO2max test results implementation."""

import numpy as np
import pandas as pd
import plotly.colors as p_colors
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from ...equations.cardio import Bike, Run
from ..test_results import TestResults
from ...constants import RANK_5COLORS


class CustomVO2MaxTestResults(TestResults):
    """
    Results container for custom VO2max test analysis.

    Parameters
    ----------
    test : CustomVO2MaxTest
        Processed custom VO2max test data to analyze.

    Attributes
    ----------
    summary : pd.DataFrame
        Comprehensive table of cardiorespiratory metrics including:
        - Predicted VO2max (ml/min and ml/kg/min)
        - Anaerobic Threshold
        - Running speed zones
        - Walking speed/grade zones
        - Cycling power zones
        - Fitness level classification (if normative data provided)
    analytics : pd.DataFrame
        an empty dataframe that is returned for compatibility with the parent class TestResults.
        The analytics dataframe is not used in this implementation.
    figures : dict of str -> go.Figure
        Dictionary of interactive Plotly figures:
        - 'vo2max_interpretation': VO2max ranking (if normative data is provided)
        - 'running_zones': speeds according to vo2max
        - 'walking_zones': grades according to vo2max and provided speed
        - 'cycling_zones': power according to vo2max
    """

    def __init__(self, test: "CustomVO2MaxTest"):
        self._summary = pd.DataFrame()
        self._analytics = pd.DataFrame()
        self._figures = {}
        self._generate_results(test)

    def _get_normative_data(self, test: "CustomVO2MaxTest"):
        participant = test.participant
        norm = test.normative_data.copy()

        # extract the percentile and fitness level from the normative data
        if norm is None or participant.gender is None or participant.age is None:
            return None

        # filter the normative data for the participant's gender and age
        norm = norm.loc[
            norm.gender.map(lambda x: x[0].upper()) == participant.gender[0].upper()
        ]
        ages = norm.age.unique()
        idxs = np.argsort(np.abs(ages - participant.age))
        age = ages[idxs[0]]
        norm = norm.loc[norm.age == age]

        return pd.DataFrame(norm)

    def _get_summary(self, test: "CustomVO2MaxTest"):

        # get the data
        vo2max = test.vo2max
        norm = self._get_normative_data(test)
        participant = test.participant

        # extract the percentile and fitness level from the normative data
        if norm is None:
            percentile = None
            fitness_level = None
        else:

            # get the closer percentile and fitness level
            closest_idx = (norm.vo2max - vo2max).abs().idxmin()
            percentile = norm.loc[closest_idx, "percentile"]
            fitness_level = norm.loc[closest_idx, "interpretation"]

        # get the vo2max percentages according to standard fatloss, endurance,
        # threshold and maximal training zones
        vo2max_percentages = {
            "fat-oxydation": vo2max * 0.6,
            "aerobic-endurance": vo2max * 0.7,
            "anaerobic-threshold": vo2max * 0.8,
            "vo2max": vo2max,
        }

        # convert those percentages to running speeds, walking grades and cycling powers
        running_speeds = pd.DataFrame(
            pd.Series(
                {k: Run().predict_speed(v, 0)[0] for k, v in vo2max_percentages.items()}
            )
        ).T
        running_speeds.insert(0, "Mode", "Running Speed (km/h)")

        walking_grades = pd.DataFrame(
            pd.Series(
                {
                    k: Run().predict_grade(v, test.walking_speed)[0]
                    for k, v in vo2max_percentages.items()
                }
            )
        ).T
        walking_grades.insert(
            0, "Mode", f"Walking Grade at {test.walking_speed} km/h (%)"
        )

        if participant.weight is None or participant.gender is None:
            cycling_powers = pd.DataFrame()
        else:
            cycling_powers = pd.DataFrame(
                pd.Series(
                    {
                        k: Bike().predict_power(
                            v,
                            participant.weight,
                            (
                                "Male"
                                if participant.gender[0].upper() == "M"
                                else "Female"
                            ),
                        )[0]
                        for k, v in vo2max_percentages.items()
                    }
                )
            ).T
            cycling_powers.insert(0, "Mode", "Cycling Power (W)")

        # create the summary dataframe
        index = pd.DataFrame(
            pd.Series(
                {
                    "VO2max (ml/kg/min)": vo2max,
                    "Percentile": percentile,
                    "Fitness Level": fitness_level.capitalize(),
                }
            )
        ).T
        out = pd.concat(
            [
                pd.concat([index, i], axis=1)
                for i in [running_speeds, walking_grades, cycling_powers]
            ],
            ignore_index=True,
        )

        out = out.groupby(index.columns.tolist() + ["Mode"]).mean()

        return out

    def _get_analytics(self, test: "CustomVO2MaxTest"):
        return pd.DataFrame()

    def _get_vo2max_interpretation_figure(self, test: "CustomVO2MaxTest"):

        # extract the percentile and fitness level from the normative data
        norm = self._get_normative_data(test)
        if norm is None:
            return None

        # generate the figure
        fig = go.Figure()
        for lbl, df in norm.groupby("interpretation"):

            # aggiungo le barre
            x0 = 1 if lbl == "Poor" else df.percentile.min()
            x1 = df.percentile.max()
            fig.add_shape(
                type="rect",
                x0=x0,
                x1=x1,
                y0=0,
                y1=1,
                line_width=0,
                fillcolor=RANK_5COLORS[lbl],
                opacity=0.3,
                label=dict(
                    text=f"<b>{lbl}<b>",
                    textposition="middle center",
                    font=dict(color=RANK_5COLORS[lbl], size=12),
                ),
            )

            # valore di intersezione in VO2
            if lbl != "Excellent":
                vo2 = df["vo2max"].max()
                fig.add_annotation(
                    x=x1,
                    y=0.0,
                    text=f"{vo2:0.1f}<br>ml/kg/min",
                    showarrow=False,
                    font=dict(size=12, color=RANK_5COLORS[lbl]),
                    valign="top",
                    align="center",
                    xanchor="center",
                    yanchor="top",
                )

        # VO2Max
        vo2max = test.vo2max
        perc = norm.percentile.to_numpy()[
            np.argmin(abs(norm["vo2max"].to_numpy() - vo2max))
        ]
        fig.add_annotation(
            x=perc,
            y=1.0,
            text=f"<b>VO<sub>2</sub>Max<br>{vo2max:0.1f}<br>ml/kg/min<b>",
            showarrow=True,
            arrowhead=2,
            arrowsize=1.3,
            arrowwidth=2,
            ax=0,
            ay=-35,
        )

        fig.update_layout(
            xaxis=dict(
                range=[1, 99],
                showgrid=False,
                showticklabels=False,
                zeroline=False,
            ),
            yaxis=dict(
                range=[-1, 1.5],
                showgrid=False,
                showticklabels=False,
                zeroline=False,
            ),
            height=200,
            margin=dict(l=20, r=20, t=100, b=20),
            title="VO<sub>2</sub>Max Interpretation",
            template="plotly_white",
        )

        return fig

    def _get_figures(self, test: "CustomVO2MaxTest"):
        out: dict[str, go.Figure] = {}

        # vo2max interpretationt
        vo2max_interpretation = self._get_vo2max_interpretation_figure(test)
        if vo2max_interpretation is not None:
            out["vo2max"] = vo2max_interpretation

        return out


__all__ = ["CustomVO2MaxTestResults"]
