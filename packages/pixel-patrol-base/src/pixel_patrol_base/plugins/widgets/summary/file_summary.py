from typing import Dict, List, Set

import polars as pl
from dash import html, dcc, Input, Output

from pixel_patrol_base.report.widget_categories import WidgetCategories
from pixel_patrol_base.report.base_widget import BaseReportWidget
from pixel_patrol_base.report.global_controls import (
    prepare_widget_data,
    GLOBAL_CONFIG_STORE_ID,
    FILTERED_INDICES_STORE_ID,
)
from pixel_patrol_base.report.factory import (
    plot_bar,
    show_no_data_message,
)

class FileSummaryWidget(BaseReportWidget):
    NAME: str = "File Data Summary"
    TAB: str = WidgetCategories.SUMMARY.value
    REQUIRES: Set[str] = {"size_bytes", "file_extension"}
    REQUIRES_PATTERNS = None

    CONTENT_ID = "file-summary-content"

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._df: pl.DataFrame | None = None

    @property
    def help_text(self) -> str:
        return (
            "Summarizes file counts, total size, and file types present in each group."
        )

    def get_content_layout(self) -> List:
        return [html.Div(id=self.CONTENT_ID)]

    def register(self, app, df: pl.DataFrame):
        self._df = df

        app.callback(
            Output(self.CONTENT_ID, "children"),
            Input("color-map-store", "data"),
            Input(FILTERED_INDICES_STORE_ID, "data"),
            Input(GLOBAL_CONFIG_STORE_ID, "data"),
        )(self._update_content)


    def _update_content(
        self,
        color_map: Dict[str, str] | None,
        subset_indices: List[int] | None,
        global_config: Dict | None,
    ):

        df_filtered, group_col, _resolved, _warning_msg, group_order = prepare_widget_data(
            self._df,
            subset_indices,
            global_config or {},
            metric_base=None,
        )

        if df_filtered.is_empty():
            return show_no_data_message()

        required_cols = {"size_bytes", "file_extension"}
        missing = [c for c in required_cols if c not in df_filtered.columns]
        if missing:
            return show_no_data_message()

        summary = (
            df_filtered
            .group_by(group_col)
            .agg(
                pl.count().alias("file_count"),
                (pl.col("size_bytes").sum() / (1024 * 1024)).alias("total_size_mb"),
                pl.col("file_extension").unique().sort().alias("file_types"),
            )
            .sort(group_col)
        )

        if summary.height == 0:
            return show_no_data_message()

        intro = self._build_intro(summary, group_col)
        graphs = self._build_plots(summary, group_col, color_map, group_order)
        table = self._build_table(summary, group_col)

        return [
            html.Div(intro, style={"marginBottom": "20px"}),
            html.Div(graphs, style={"marginBottom": "20px"}),
            html.Div(table, style={"marginTop": "10px"}),
        ]


    @staticmethod
    def _build_intro(summary: pl.DataFrame, group_col: str) -> List:
        intro_md: List = [
            html.P(
                f"This summary focuses on file properties across "
                f"{summary.height} group(s) (by '{group_col}')."
            )
        ]

        for row in summary.iter_rows(named=True):
            ft_str = ", ".join(row["file_types"])
            intro_md.append(
                html.P(
                    f"Group '{row[group_col]}' contains "
                    f"{row['file_count']} files "
                    f"({row['total_size_mb']:.3f} MB) with types: {ft_str}."
                )
            )

        return intro_md

    @staticmethod
    def _build_plots(
        summary: pl.DataFrame,
        group_col: str,
        color_map: Dict[str, str],
        group_order
    ) -> List:

        figs: List = []

        fig_count = plot_bar(
            df=summary,
            x=group_col,
            y="file_count",
            order_x=group_order,
            color=group_col,
            color_map=color_map,
            title="File Count per Group",
            labels={group_col: group_col, "file_count": "Number of files"},
            force_category_x=True,
            show_legend=False,
        )
        figs.append(dcc.Graph(figure=fig_count))

        fig_size = plot_bar(
            df=summary,
            x=group_col,
            y="total_size_mb",
            order_x=group_order,
            color=group_col,
            color_map=color_map,
            title="Total Size per Group (MB)",
            labels={group_col: group_col, "total_size_mb": "Size (MB)"},
        )
        figs.append(dcc.Graph(figure=fig_size))

        return figs

    @staticmethod
    def _build_table(summary: pl.DataFrame, group_col: str) -> html.Table:
        header = html.Thead(
            html.Tr(
                [
                    html.Th(f"{group_col.replace('_',' ').title()}"),
                    html.Th("Number of Files"),
                    html.Th("Size (MB)"),
                    html.Th("File Extension"),
                ]
            )
        )

        body_rows: List[html.Tr] = []
        for row in summary.iter_rows(named=True):
            body_rows.append(
                html.Tr(
                    [
                        html.Td(row[group_col]),
                        html.Td(row["file_count"]),
                        html.Td(f"{row['total_size_mb']:.3f}"),
                        html.Td(", ".join(row["file_types"])),
                    ]
                )
            )

        return html.Table(
            [header, html.Tbody(body_rows)],
            style={"width": "100%", "borderCollapse": "collapse"},
        )