import os
from collections import defaultdict

import numpy as np
import pandas as pd
import panel as pn
from bokeh.layouts import gridplot
from bokeh.models import (
    BooleanFilter,
    CategoricalColorMapper,
    CDSView,
    ColumnDataSource,
    CustomJS,
    GroupFilter,
    HoverTool,
)
from bokeh.palettes import d3
from bokeh.plotting import figure
from bokeh.transform import jitter, linear_cmap

# output_notebook()
# pn.extension('vtk')
from minisom import MiniSom
from sade.datasets.loaders import get_image_files_list

workdir = os.path.expanduser(
    "/ASD/ahsan_projects/braintypicality/workdir/cuda_opt/learnable/eval/ckpt_1500002/smin=0.01"
)
score_file = np.load(
    f"{workdir}/abcd-val_abcd-test_lesion_load_20_results.npz", allow_pickle=True
)
abcd_data = np.concatenate(
    [score_file[f] for f in ["eval_score_norms", "inlier_score_norms"]], axis=0
)

score_file = np.load(
    f"{workdir}/ibis-inlier_ibis-hr-inlier_ibis-atypical_results.npz", allow_pickle=True
)
ibis_typical = score_file["eval_score_norms"]
ibis_atypical = np.concatenate(
    [score_file[f] for f in ["inlier_score_norms", "ood_score_norms"]], axis=0
)

score_file = np.load(
    f"{workdir}/ibis-inlier_ibis-hr-inlier_ibis-ds-sa_results.npz", allow_pickle=True
)
ibis_ds = score_file["ood_score_norms"]

score_file = np.load(
    f"{workdir}/ibis-inlier_ibis-hr-inlier_ibis-asd_results.npz", allow_pickle=True
)
ibis_asd = score_file["ood_score_norms"]
region_scores = pd.read_csv(
    f"/ASD/ahsan_projects/braintypicality/workdir/cuda_opt/learnable/eval/heatmaps/region_scores.csv",
    index_col="ID",
)


datset_list = [abcd_data, ibis_typical, ibis_atypical, ibis_ds, ibis_asd]
score_data = np.concatenate(datset_list)
# score_target = np.concatenate([[i]*len(s) for i,s in enumerate(datset_list)], axis=0)
score_target = np.concatenate(
    [
        [0] * len(abcd_data),
        [1] * len(ibis_typical),
        [2] * len(ibis_atypical),
        [3] * len(ibis_ds),
        [4] * len(ibis_asd),
    ]
)

sample_ids = []
for name in [
    "abcd-val",
    "abcd-test",
    "ibis-inlier",
    "ibis-hr-inlier",
    "ibis-atypical",
    "ibis-ds-sa",
    "ibis-asd",
]:
    filenames = get_image_files_list(
        name,
        dataset_dir="/DATA/Users/amahmood/braintyp/processed_v2",
        splits_dir="/codespace/sade/sade/datasets/brains/",
    )
    sample_ids.extend(
        [x["image"].split("/")[-1].replace(".nii.gz", "") for x in filenames]
    )
sample_ids = np.array(sample_ids)

score_label_names = ["ABCD", "LR-Typical", "HR-Inlier", "Down's Syndrome", "ASD +ve"]
colors = ["tab:cyan", "tab:green", "tab:blue", "tab:orange", "tab:red"]

palette = d3["Category10"][len(score_label_names)]
label_color_map = CategoricalColorMapper(factors=score_label_names, palette=palette)

# data normalization
data = (score_data - np.mean(abcd_data, axis=0)) / np.std(abcd_data, axis=0)
inlier_data = data[:380]


num_neurons = 5 * np.sqrt(data.shape[0])
grid_size = int(np.ceil(np.sqrt(num_neurons)))

# Initialization and training
n_neurons = 7
m_neurons = 7
max_iters = 10_000
som = MiniSom(
    n_neurons,
    m_neurons,
    data.shape[1],
    sigma=2,
    learning_rate=0.5,
    neighborhood_function="gaussian",
    random_seed=42,
    topology="rectangular",
)
som.pca_weights_init(inlier_data)

som.train(inlier_data, max_iters, verbose=True)
print(f"Topographic Error: {som.topographic_error(data[:500])}")


# Create data sources for the plots


distance_map = umatrix = som.distance_map()
weights = som.get_weights()
labels_map = som.labels_map(data, [score_label_names[i] for i in score_target])
win_map = som.win_map(data, return_indices=True)

heatmap_data = defaultdict(list)

# TODO: keep track of cell ROI scores
# and only doisplay those above a certain threshold 
for i in range(weights.shape[0]):
    for j in range(weights.shape[1]):
        heatmap_data["x"].append(i)
        heatmap_data["y"].append(j)
        heatmap_data["distance"].append(distance_map[(i, j)])

        sids = sample_ids[win_map[(i, j)]] if len(win_map[(i, j)]) > 0 else []
        heatmap_data["sample_ids"].append(sids)

        counts = labels_map[(i, j)]
        if len(counts) > 0:
            heatmap_data["total"].append(counts.total())
            heatmap_data["counts"].append(dict(counts))
        else:
            heatmap_data["total"].append(0)
            heatmap_data["counts"].append({l: 0 for l in score_label_names})

source_heatmap = ColumnDataSource(heatmap_data)


scatter_data = defaultdict(list)

for idx, x in enumerate(data):
    wx, wy = som.winner(x)
    scatter_data["x"].append(wx)
    scatter_data["y"].append(wy)
    scatter_data["cohort"].append(score_label_names[score_target[idx]])
    scatter_data["ID"].append(sample_ids[idx])
    scatter_data["roi_scores"].append(region_scores.iloc[np.random.randint(200)])

source_scatter = ColumnDataSource(scatter_data)
view = CDSView(filter=~GroupFilter(column_name="cohort", group="ABCD"))

# create a ColumnDataSource for the barplot data
init_bar_data = pd.Series(scatter_data["cohort"]).value_counts()
init_bar_data = dict(cohort=init_bar_data.index.to_list(), count=init_bar_data.values)
bar_source = ColumnDataSource(data=init_bar_data)
# create a figure for the barplot
bar_p = figure(
    title="Barplot",
    x_axis_label="Cohort",
    y_axis_label="Counts",
    x_range=score_label_names,
    height=200,
)
bar_p.vbar(
    x="cohort",
    width=0.5,
    bottom=0,
    top="count",
    source=bar_source,
    color="blue",
)


source_heatmap = ColumnDataSource(heatmap_data)
heatmap_plot = figure(
    title="SOM Heatmap",
    match_aspect=True,
    width=600,
    height=600,
    tools="wheel_zoom,reset,tap,box_select",
)
cmap = linear_cmap("distance", "Reds256", low=0, high=max(heatmap_data["distance"]))
heatmap_renderer = heatmap_plot.rect(
    x="x", y="y", width=1, height=1, source=source_heatmap, fill_color=cmap
)

heatmap_plot.scatter(
    x=jitter("x", 0.2),
    y=jitter("y", 0.2),
    source=source_scatter,
    color={"field": "cohort", "transform": label_color_map},
    size=7,
    alpha=0.6,
    view=view,
    legend_group="cohort",
)

heatmap_hover = HoverTool(
    tooltips=[("Samples", "@total")],
    mode="mouse",
    point_policy="follow_mouse",
    renderers=[heatmap_renderer],
)
heatmap_plot.add_tools(heatmap_hover)
heatmap_plot.legend.nrows=1

init_region_data = region_scores.describe().loc["mean"].sort_values(ascending=True)
init_region_data_dict = dict(
    roi=init_region_data.index.to_list(), score=init_region_data.values
)
region_source = ColumnDataSource(init_region_data_dict)
region_hist = figure(
    title="ROI Scores",
    x_axis_label="Scores",
    y_axis_label="Regions",
    y_range=init_region_data.index[:20].to_list(),
)
region_hist.hbar(
    y="roi",
    right="score",
    source=region_source,
    color="blue",
    height=0.5,
)

# Python callback function
# @pn.cache


def update_barplot(attr, old, new):
    selected_index = new[0] if len(new) > 0 else None
    if selected_index is None:
        bar_source.data = init_bar_data
        region_source.data = init_region_data_dict
        region_hist.y_range.factors = init_region_data.index[:20].to_list()
        return

    data = source_heatmap.data["counts"][selected_index]
    # cohorts= list(data.keys())
    # counts = list(data.values())
    bar_source.data = {"cohort": list(data.keys()), "count": list(data.values())}

    selected_samples = source_heatmap.data["sample_ids"][selected_index]
    selected_samples = region_scores.index.isin(selected_samples)
    roi_data = region_scores[selected_samples].mean(numeric_only=True).sort_values(ascending=True)
    roi_data = roi_data[roi_data > 80]
    roi = roi_data.index.to_list()
    region_source.data = dict(roi=roi, score=roi_data.values)
    region_hist.y_range.factors = roi


# Attach the callback to the heatmap's selection
source_heatmap.selected.on_change("indices", update_barplot)

# def create_barplot()
# bar_data = region_scores.loc[sample_ids].mean().sort_values(ascending=False)[:5]
# bar_source = ColumnDataSource(bar_data.to_dict())
# bar_p = figure(title="ROI Scores", x_axis_label='Anomaly Score', y_axis_label='Region', y_range=bar_data.index.to_list())
# bar_p.hbar(y="cohort", width=0.5, bottom=0, right="value", source=bar_data, color="blue")

# source_heatmap.selected.on_change

# show the plots
p = gridplot([[heatmap_plot, region_hist], [bar_p]])
# p.js_on_event('tap', callback)
bokeh_pane = pn.pane.Bokeh(p, theme="dark_minimal")
bokeh_pane.servable()
