"""agility test module"""

#! IMPORTS

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from ..constants import MINIMUM_CONTACT_FORCE_N, RANK_5COLORS, SIDE_COLORS
from ..records.timeseries import Point3D
from ..records.records import ForcePlatform
from ..signalprocessing import butterworth_filt, fillna
from ..utils import hex_to_rgba
from ..records.pipelines import get_default_processing_pipeline
from ..records.agility import ChangeOfDirectionExercise
from .protocols import Participant, TestProtocol, TestResults

#! CONSTANTS


__all__ = ["ShuttleTest", "ShuttleTestResults"]


class ShuttleTest(TestProtocol):

    def __init__(
        self,
        participant: Participant,
        change_of_direction_exercises: list[ChangeOfDirectionExercise],
        normative_data: pd.DataFrame = pd.DataFrame(),
    ):
        super().__init__(
            participant,
            normative_data,
        )
        self.set_change_of_direction_exercise(change_of_direction_exercises)

    def set_change_of_direction_exercise(
        self, records: list[ChangeOfDirectionExercise]
    ):
        if (not isinstance(records, list)) or (
            not all(isinstance(record, ChangeOfDirectionExercise) for record in records)
        ):
            raise ValueError(
                "recorda must be a list of ChangeOfDirectionExercise instances."
            )
        self._change_of_direction_exercises = records

    @property
    def change_of_direction_exercises(self):
        return self._change_of_direction_exercises

    def copy(self):
        return ShuttleTest(
            participant=self.participant.copy(),
            normative_data=self.normative_data,
            change_of_direction_exercises=[
                i.copy() for i in self.change_of_direction_exercises
            ],
        )

    @classmethod
    def from_files(
        cls,
        filenames: list[str],
        participant: Participant,
        normative_data: pd.DataFrame = pd.DataFrame(),
        left_foot_ground_reaction_force: str | None = "left_foot",
        right_foot_ground_reaction_force: str | None = "right_foot",
        s2: str | None = "s2",
    ):
        if not isinstance(filenames, list):
            raise ValueError("filename must be a list")
        return cls(
            participant=participant,
            normative_data=normative_data,
            change_of_direction_exercises=[
                ChangeOfDirectionExercise.from_tdf(
                    file=filename,
                    left_foot_ground_reaction_force=left_foot_ground_reaction_force,
                    right_foot_ground_reaction_force=right_foot_ground_reaction_force,
                    s2=s2,
                )
                for filename in filenames
            ],
        )

    def get_results(self):
        return ShuttleTestResults(self.processed_data)

    @property
    def processed_data(self):
        out = self.copy()
        pipeline = self.processing_pipeline
        for i in range(len(out.change_of_direction_exercises)):
            pipeline(out.change_of_direction_exercises[i], inplace=True)
        return out

    @property
    def processing_pipeline(self):
        pipeline = get_default_processing_pipeline()

        def get_point3d_processing_func(point: Point3D):
            point.strip(inplace=True)
            point.fillna(inplace=True)
            fsamp = float(1 / np.mean(np.diff(point.index)))
            point.apply(
                butterworth_filt,
                fcut=6,
                fsamp=fsamp,
                order=4,
                ftype="lowpass",
                phase_corrected=True,
                inplace=True,
            )

        def get_forceplatform_processing_func(fp: ForcePlatform):

            # ensure force below minimum contact are set to NaN
            vals = fp.force.copy().to_numpy()
            module = fp.force.copy().module.to_numpy().flatten()  # type: ignore
            idxs = module < MINIMUM_CONTACT_FORCE_N
            for i in ["origin", "force", "torque"]:
                vals = fp[i].copy().to_numpy()
                vals[idxs, :] = np.nan
                fp[i][:, :] = vals

            # strip nans from the ends
            fp.strip(inplace=True)

            # fill remaining force nans with zeros
            fp.force[:, :] = fillna(fp.force.to_numpy(), value=0, inplace=False)

            # fill remaining position nans via cubic spline
            fp.origin[:, :] = fillna(fp.origin.to_numpy(), inplace=False)

            # lowpass filter both origin and force
            fsamp = float(1 / np.mean(np.diff(fp.index)))
            filt_fun = lambda x: butterworth_filt(
                x,
                fcut=10,
                fsamp=fsamp,  # type: ignore
                order=4,
                ftype="lowpass",
                phase_corrected=True,
            )
            fp.origin.apply(filt_fun, axis=0, inplace=True)
            fp.force.apply(filt_fun, axis=0, inplace=True)

            # update moments
            fp.update_moments(inplace=True)

            # set moments corresponding to the very low vertical force to zero
            module = fp.force.copy().module.to_numpy().flatten()  # type: ignore
            idxs = module < MINIMUM_CONTACT_FORCE_N
            vals = fp.torque.copy().to_numpy()
            vals[idxs, :] = 0
            fp.torque[:, :] = vals

        pipeline["Point3D"] = [get_point3d_processing_func]
        pipeline["ForcePlatform"] = [get_forceplatform_processing_func]

        return pipeline


class ShuttleTestResults(TestResults):

    def __init__(self, test: ShuttleTest):
        if not isinstance(test, ShuttleTest):
            raise ValueError("Test must be a ShuttleTest instance")
        super().__init__(test, include_emg=False)

    def _get_summary(self, test: ShuttleTest):
        cont = {}
        out = {
            "side": [],
            "limb": [],
            "type": [],
            "n": [],
            "Contact Time (s)": [],
            "Loading Time (s)": [],
            "Loading Time (%)": [],
            "Propulsion Time (s)": [],
            "Propulsion Time (%)": [],
            "Max Velocity (m/s)": [],
        }
        for exercise in test.change_of_direction_exercises:
            if exercise.side not in cont:
                cont[exercise.side] = 0
            cont[exercise.side] += 1
            contact_time = exercise.contact_time
            loading_time = exercise.loading_time
            propulsion_time = exercise.propulsion_time
            out["Contact Time (s)"].append(contact_time)
            out["Loading Time (s)"].append(loading_time)
            out["Loading Time (%)"].append(100 * loading_time / (contact_time))
            out["Propulsion Time (s)"].append(propulsion_time)
            out["Propulsion Time (%)"].append(100 * propulsion_time / contact_time)
            out["Max Velocity (m/s)"].append(
                float(exercise.velocity[exercise.anteroposterior_axis].max())
            )
            side = exercise.side
            out["side"].append(side if side == "bilateral" else "unilateral")
            out["n"].append(cont[side])
            out["limb"].append(side)
            out["type"].append(exercise.__class__.__name__)

        df = pd.DataFrame(out)
        df = df.melt(
            id_vars=["type", "side", "limb", "n"],
            var_name="parameter",
            value_name="value",
        )
        df = df.pivot_table(
            index=["parameter", "type", "side", "n"],
            columns="limb",
            values="value",
        )
        df = pd.concat([df.index.to_frame(), df], axis=1)
        df.reset_index(drop=True, inplace=True)
        return df

    def _get_analytics(self, test: ShuttleTest):
        cont = {}
        out = []
        for exercise in test.change_of_direction_exercises:
            if exercise.side not in cont:
                cont[exercise.side] = 0
            cont[exercise.side] += 1
            new = exercise.to_dataframe()
            new.loc[new.index, "side"] = exercise.side
            new.loc[new.index, "n"] = cont[exercise.side]
            out.append(new)

        return pd.concat(out, ignore_index=True)

    def _get_figures(self, test: ShuttleTest):
        out: dict[str, go.Figure] = {}
        out["times"] = self._get_times_figure(test)
        out["velocities"] = self._get_velocities_figure(test)
        return out

    def _get_times_figure(self, test: ShuttleTest):
        # retrieve the time data
        contact_time_data, contact_time_norms = self._get_data_and_norms(
            "Contact Time (s)",
            test,
        )
        loading_time_data, loading_time_norms = self._get_data_and_norms(
            "Loading Time (s)",
            test,
        )
        prop_time_data, prop_time_norms = self._get_data_and_norms(
            "Propulsion Time (s)",
            test,
        )

        # aggregate data and norms
        performance_data = {}
        performance_norms = {}
        for t, s in contact_time_data:
            performance_data[(t, s)] = {}
            performance_data[(t, s)]["Loading Time (s)"] = loading_time_data[(t, s)]
            performance_data[(t, s)]["Propulsion Time (s)"] = prop_time_data[(t, s)]
            performance_data[(t, s)]["Contact Time (s)"] = contact_time_data[(t, s)]
            performance_norms[(t, s)] = {}
            if len(contact_time_norms) > 0:
                performance_norms[(t, s)]["Contact Time (s)"] = contact_time_norms[
                    (t, s)
                ]
            if len(loading_time_norms) > 0:
                performance_norms[(t, s)]["Loading Time (s)"] = loading_time_norms[
                    (t, s)
                ]
            if len(prop_time_norms) > 0:
                performance_norms[(t, s)]["Propulsion Time (s)"] = prop_time_norms[
                    (t, s)
                ]

        # generate the figure
        figures: dict[str, go.Figure] = {}
        for t, s in performance_data:
            p_data = performance_data.get((t, s))
            p_norms = performance_norms.get((t, s))
            if p_data is not None:
                fig = self._get_time_figure(
                    p_data,
                    p_norms,  # type: ignore
                )
                title = f"{t}-{s}".capitalize()
                # fig.update_layout(title=title)
                figures[title] = fig

        return figures

    def _get_data_and_norms(
        self,
        metric: str,
        test: ShuttleTest,
        bilateral_is_unique: bool = True,
        ranks: dict[str, str] = RANK_5COLORS,
        symmetric_ranks: bool = False,
        reversed_ranks: bool = False,
    ):

        # retrieve the data of the required metric from summary
        metric_df = self.summary.copy()
        params: list[str] = metric_df.parameter.to_list()
        idx = [i for i, v in enumerate(params) if v.endswith(metric)]
        metric_df = metric_df.iloc[idx]
        metric_df = metric_df.melt(
            id_vars=["type", "side", "parameter", "n"],
            var_name="limb",
            value_name="value",
        )
        if bilateral_is_unique:
            idx = (metric_df.side != "bilateral") | (metric_df.limb == "left")
            metric_df = metric_df.loc[idx]
            new_limbs = metric_df[["side", "limb"]].apply(
                lambda x: x.side if x.side == "bilateral" else x.limb,
                axis=1,
            )
            metric_df.loc[metric_df.index, "limb"] = new_limbs
        metric_df.reset_index(drop=True, inplace=True)

        # get the data sorted according to the subplots to be rendered
        data: dict[tuple[str, str], dict[str, dict[str, list[float]]]] = {}
        for (t, s), dfr in metric_df.groupby(["type", "side"]):
            key = (str(t), str(s))
            val: dict[str, dict[str, list[float]]] = {}
            for side, dfs in dfr.groupby("limb"):
                k = str(side)
                v = dfs.sort_values("n").value.to_numpy().flatten().tolist()
                val[k] = v
            data[key] = val

        # get the normative data sorted according to the subplots to be rendered
        norms: dict[
            tuple[str, str], tuple[list[float], list[float], list[str], list[str]]
        ] = {}
        combs = metric_df[["type", "side"]].drop_duplicates().values.tolist()
        if not test.normative_data.empty:
            gender = test.participant.gender
            if gender is None:
                raise ValueError("Normative Data require gender being specified.")
            gender = gender.lower()[0]
            norm = test.normative_data.copy()
            params: list[str] = norm.parameter.to_list()
            idx = [i for i, v in enumerate(params) if v.endswith(metric)]
            norm = norm.iloc[idx]
            types = norm["type"].str.lower().tolist()
            types = [t.lower().rsplit(" (", 1)[0] for t in types]
            sides = norm["side"].str.lower().tolist()
            genders = [i.lower()[0] for i in norm["gender"]]
            for t, s in combs:
                types_idx = [t.lower().rsplit(" (", 1)[0] in v for v in types]
                types_idx = np.array(types_idx)
                sides_idx = np.array([s in v for v in sides])
                gender_idx = np.array([gender == v for v in genders])
                mask = types_idx & sides_idx & gender_idx
                tnorm = norm.loc[mask]
                if tnorm.shape[0] > 1:
                    msg = "Multiple normative values found for jump elevation."
                    raise ValueError(msg)
                if not tnorm.empty:
                    avg = float(tnorm["mean"].to_numpy()[0])
                    std = float(tnorm["std"].to_numpy()[0])
                    rank_clrs = list(ranks.values())
                    rank_lbls = list(ranks.keys())
                    if reversed_ranks:
                        rank_clrs = rank_clrs[::-1]
                        rank_lbls = rank_lbls[::-1]
                    n_vals = len(ranks)
                    if symmetric_ranks:
                        rank_clrs = rank_clrs[::-1] + rank_clrs
                        rank_lbls = rank_lbls[::-1] + rank_lbls
                        rank_vals = np.arange(n_vals + 1)
                    else:
                        if n_vals % 2 == 1:
                            rank_vals = np.arange((n_vals + 1) // 2) + 1
                        else:
                            rank_vals = np.arange((n_vals + 2) // 2)
                    rank_vals = np.concatenate([rank_vals, -rank_vals]) * std + avg
                    rank_vals = np.unique(rank_vals)[::-1]
                    rank_lows = rank_vals[1:].copy().tolist()
                    rank_tops = rank_vals[:-1].copy().tolist()
                    norms[(t, s)] = (rank_lows, rank_tops, rank_lbls, rank_clrs)

        return data, norms

    def _get_time_figure(
        self,
        performance_data: dict[str, list[float]],
        performance_norms: tuple[list[float], list[float], list[str], list[str]],
    ):

        # generate the figure
        subplot_titles = ["Step Time"]
        fig = make_subplots(
            rows=1,
            cols=1,
            vertical_spacing=0.05,
            subplot_titles=subplot_titles,
            horizontal_spacing=0.01,
        )
        fig.update_layout(
            template="plotly_white",
            legend=dict(title_text="Legend"),
            width=1000,
            height=400,
            bargroupgap=0.25,
        )
        fig.update_xaxes(
            showgrid=False,
            showline=True,
            zeroline=False,
            linewidth=2,
            linecolor="black",
        )
        fig.update_yaxes(
            showgrid=False,
            showline=False,
            zeroline=False,
            showticklabels=False,
        )

        # plot the bars representing the performance value
        yvals = []
        colors_plotted = []
        side_colors = list(SIDE_COLORS.values())
        sides = sorted(list(performance_data["Contact Time (s)"].keys()))
        for p, (param, performances) in enumerate(performance_data.items()):

            # get the normative data if available
            if len(performance_norms) > 0 and performance_norms[param] is not None:
                rank_lows, rank_tops, rank_lbls, rank_clrs = performance_norms[param]
                rank_lows = np.array(rank_lows)
                rank_tops = np.array(rank_tops)
            else:
                rank_lows = rank_tops = rank_lbls = rank_clrs = np.array([])

            # plot each single value
            for k, side in enumerate(sides):
                values = performances[side]
                for j, y in enumerate(values):
                    if np.isnan(y):
                        continue
                    value = int(round(y * 1000))

                    # if normative data are available get the main bar color as
                    # the color of the rank achieved by the actual value.
                    # Otherwise, use the color of the side with which the jump
                    # has been performed.
                    if len(rank_tops) > 0:
                        idx = np.where(rank_tops >= value)[0]
                        idx = idx[-1] if len(idx) > 0 else 0  # (len(rank_clrs) - 1)
                        color = rank_clrs[idx]
                    else:
                        color = side_colors[p]  # type: ignore

                    # update the y-axis range values
                    if param == "Contact Time (s)":
                        yvals += rank_lows.tolist() + rank_tops.tolist() + [value]

                    # plot the bar
                    fig.add_trace(
                        row=1,
                        col=1,
                        trace=go.Bar(
                            x=[k + 1],
                            y=[value],
                            text=[
                                (
                                    f"{value}ms"
                                    if param != "Contact Time (s)"
                                    else f"Rep {j+1}<br>{value} ms"
                                )
                            ],
                            textposition="inside",
                            insidetextanchor=(
                                "middle" if param != "Contact Time (s)" else "start"
                            ),
                            textangle=0,
                            showlegend=(
                                (j == 0)
                                and (k == 0)
                                and len(performance_norms) == 0
                                and (param != "Contact Time (s)")
                            ),
                            marker_color=[
                                (
                                    color
                                    if param != "Contact Time (s)"
                                    else "rgba(0,0,0,0)"
                                )
                            ],
                            marker_line_color=[
                                (
                                    "black"
                                    if param != "Contact Time (s)"
                                    else "rgba(0,0,0,0)"
                                )
                            ],
                            name=param.capitalize(),
                            legendgroup="times",
                            offsetgroup=str(j + 1),
                        ),
                    )

        # set the barmode
        fig.update_layout(barmode="stack")

        # update the yaxes
        yrange = [0, np.max(yvals) * 1.4]
        fig.update_yaxes(row=1, col=1, range=yrange)

        # update the xaxes
        fig.update_xaxes(
            col=1,
            row=1,
            showticklabels=True,
            tickvals=np.arange(len(sides)) + 1,
            tickmode="array",
            ticktext=[str(i).capitalize() for i in sides],
        )

        # plot the norms as colored boxes behind the bars
        zipped = zip(rank_lows, rank_tops, rank_lbls, rank_clrs)
        for rlow, rtop, rlbl, rclr in zipped:
            if rlow == np.min(rank_lows) and rlow > yrange[0]:
                rlow = yrange[0]
            if rtop == np.max(rank_tops) and rtop < yrange[1]:
                rtop = yrange[1]
            fig.add_shape(
                type="rect",
                x0=0,
                x1=len(performance_data) + 1,
                y0=rlow,
                y1=rtop,
                line_width=0,
                fillcolor=hex_to_rgba(rclr, 0.25),
                layer="below",
                name=rlbl.capitalize(),
                legendgroup="Rank",
                legendgrouptitle_text="Rank",
                showlegend=rclr not in colors_plotted,
                col=1,
                row=1,
            )
            if rtop < np.max(rank_tops):
                fig.add_annotation(
                    x=len(performance_data) + 1,
                    y=rtop,
                    text=f"{rtop:0.1f}s",
                    showarrow=False,
                    xanchor="right",
                    yanchor="top",
                    font=dict(color=rclr),
                    valign="top",
                    yshift=0,
                    name=rlbl,
                    col=1,  # type: ignore
                    row=1,  # type: ignore
                )

            # ensure that the legend is plotted once
            colors_plotted.append(rclr)

        return fig

    def _get_velocities_figure(self, test: ShuttleTest):
        # retrieve the velocity data
        performance_data, performance_norms = self._get_data_and_norms(
            "Max Velocity (m/s)",
            test,
        )

        # generate the figure
        figures: dict[str, go.Figure] = {}
        for t, s in performance_data:
            p_data = performance_data.get((t, s))
            p_norms = performance_norms.get((t, s))
            if p_data is not None:
                fig = self._get_velocity_figure(
                    p_data,
                    p_norms,  # type: ignore
                )
                title = f"{t}-{s}".capitalize()
                # fig.update_layout(title=title)
                figures[title] = fig

        return figures

    def _get_velocity_figure(
        self,
        performance_data: dict[str, list[float]],
        performance_norms: tuple[list[float], list[float], list[str], list[str]],
    ):

        # generate the figure
        subplot_titles = ["Max Velocity"]
        fig = make_subplots(
            rows=1,
            cols=1,
            vertical_spacing=0.05,
            subplot_titles=subplot_titles,
            horizontal_spacing=0.01,
        )
        fig.update_layout(
            template="plotly_white",
            legend=dict(title_text="Legend"),
            width=1000,
            height=400,
            bargroupgap=0.25,
        )
        fig.update_xaxes(
            showgrid=False,
            showline=True,
            zeroline=False,
            linewidth=2,
            linecolor="black",
        )
        fig.update_yaxes(
            showgrid=False,
            showline=False,
            zeroline=False,
            showticklabels=False,
        )

        # get the normative data if available
        if performance_norms is not None:
            rank_lows, rank_tops, rank_lbls, rank_clrs = performance_norms
            rank_lows = np.array(rank_lows)
            rank_tops = np.array(rank_tops)
        else:
            rank_lows = rank_tops = rank_lbls = rank_clrs = np.array([])

        # plot the bars representing the performance value
        yvals = []
        colors_plotted = []
        for k, (side, performances) in enumerate(performance_data.items()):
            for j, y in enumerate(performances):
                if np.isnan(y):
                    continue
                value = float(round(y, 1))

                # if normative data are available get the main bar color as
                # the color of the rank achieved by the actual value.
                # Otherwise, use the color of the side with which the jump
                # has been performed.
                if len(rank_tops) > 0:
                    idx = np.where(rank_tops >= value)[0]
                    idx = idx[-1] if len(idx) > 0 else 0  # (len(rank_clrs) - 1)
                    color = rank_clrs[idx]
                else:
                    color = SIDE_COLORS[side]  # type: ignore

                # update the y-axis range values
                yvals += rank_lows.tolist() + rank_tops.tolist() + [value]

                # plot the bar
                fig.add_trace(
                    row=1,
                    col=1,
                    trace=go.Bar(
                        x=[k + 1],
                        y=[value],
                        text=[f"Rep{j+1}<br>{value:0.2f} m/s"],
                        textposition="outside",
                        textangle=0,
                        showlegend=(j == 0)
                        and performance_norms is None
                        and len(performance_data) > 1,
                        marker_color=[color],
                        marker_line_color=["black"],
                        name=side.capitalize(),
                        legendgroup="Limb",
                        offsetgroup=str(j + 1),
                    ),
                )

        # update the yaxes
        yrange = [np.min(yvals) * 0.9, np.max(yvals) * 1.2]
        fig.update_yaxes(row=1, col=1, range=yrange)

        # update the xaxes
        fig.update_xaxes(
                col=1,
                row=1,
                showticklabels=True,
                tickvals=np.arange(len(performance_data)) + 1,
                tickmode="array",
                ticktext=[str(i).capitalize() for i in list(performance_data.keys())],
            )

        # plot the norms as colored boxes behind the bars
        zipped = zip(rank_lows, rank_tops, rank_lbls, rank_clrs)
        for rlow, rtop, rlbl, rclr in zipped:
            if rlow == np.min(rank_lows) and rlow > yrange[0]:
                rlow = yrange[0]
            if rtop == np.max(rank_tops) and rtop < yrange[1]:
                rtop = yrange[1]
            fig.add_shape(
                type="rect",
                x0=0,
                x1=len(performance_data) + 1,
                y0=rlow,
                y1=rtop,
                line_width=0,
                fillcolor=hex_to_rgba(rclr, 0.25),
                layer="below",
                name=rlbl.capitalize(),
                legendgroup="Rank",
                legendgrouptitle_text="Rank",
                showlegend=rclr not in colors_plotted,
                col=1,
                row=1,
            )
            if rtop < np.max(rank_tops):
                fig.add_annotation(
                    x=len(performance_data) + 1,
                    y=rtop,
                    text=f"{rtop:0.1f}m/s",
                    showarrow=False,
                    xanchor="right",
                    yanchor="top",
                    font=dict(color=rclr),
                    valign="top",
                    yshift=0,
                    name=rlbl,
                    col=1,  # type: ignore
                    row=1,  # type: ignore
                )

            # ensure that the legend is plotted once
            colors_plotted.append(rclr)

        return fig
