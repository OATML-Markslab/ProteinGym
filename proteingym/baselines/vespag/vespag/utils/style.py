import functools as ft
from dataclasses import dataclass
from typing import Union

import polars as pl
import seaborn as sns

# Copyright (c) 2023 Christopher Prohm
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
#


@pl.api.register_dataframe_namespace("sns")
@pl.api.register_lazyframe_namespace("sns")
@dataclass
class SeabornPlotting:
    df: Union[pl.DataFrame, pl.LazyFrame]

    def pipe(self, func, /, **kwargs):
        def maybe_collect(df):
            return df.collect() if isinstance(df, pl.LazyFrame) else df

        exprs = {}
        for key in "x", "y", "hue", "col", "row":
            val = kwargs.get(key)
            if val is None:
                continue

            expr = pl.col(val) if isinstance(val, str) else val

            exprs[expr.meta.output_name()] = expr
            kwargs[key] = expr.meta.output_name()

        return (
            self.df.select(list(exprs.values()))
            .pipe(maybe_collect)
            .to_pandas()
            .pipe(func, **kwargs)
        )

    relplot = ft.partialmethod(pipe, sns.relplot)
    scatterplot = ft.partialmethod(pipe, sns.scatterplot)
    lineplot = ft.partialmethod(pipe, sns.lineplot)
    displot = ft.partialmethod(pipe, sns.displot)
    histplot = ft.partialmethod(pipe, sns.histplot)
    kdeplot = ft.partialmethod(pipe, sns.kdeplot)
    ecdfplot = ft.partialmethod(pipe, sns.ecdfplot)
    rugplot = ft.partialmethod(pipe, sns.rugplot)
    distplot = ft.partialmethod(pipe, sns.distplot)
    catplot = ft.partialmethod(pipe, sns.catplot)
    stripplot = ft.partialmethod(pipe, sns.stripplot)
    swarmplot = ft.partialmethod(pipe, sns.swarmplot)
    boxplot = ft.partialmethod(pipe, sns.boxplot)
    violinplot = ft.partialmethod(pipe, sns.violinplot)
    boxenplot = ft.partialmethod(pipe, sns.boxenplot)
    pointplot = ft.partialmethod(pipe, sns.pointplot)
    barplot = ft.partialmethod(pipe, sns.barplot)
    countplot = ft.partialmethod(pipe, sns.countplot)
    lmplot = ft.partialmethod(pipe, sns.lmplot)
    regplot = ft.partialmethod(pipe, sns.regplot)
    residplot = ft.partialmethod(pipe, sns.residplot)
