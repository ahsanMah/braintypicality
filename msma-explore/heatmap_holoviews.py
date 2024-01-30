import os
from collections import defaultdict

import holoviews as hv
import hvplot.pandas  # noqa
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
from holoviews import streams

# output_notebook()
hv.extension("bokeh")
pn.extension()

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

ibis_metadata = pd.read_csv(
    f"/ASD/ahsan_projects/braintypicality/workdir/cuda_opt/learnable/eval/heatmaps/ibis_metadata.csv",
)
ibis_metadata.index = ibis_metadata["CandID"].apply(lambda x: "IBIS" + str(x))
ibis_metadata.index.name = "ID"
das_cols = [c for c in ibis_metadata.columns if "DAS" in c]

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

scatter_data = defaultdict(list)
for idx in range(0, data.shape[0]):
    x = data[idx]
    wx, wy = som.winner(x)
    scatter_data["x"].append(wx)
    scatter_data["y"].append(wy)
    scatter_data["cohort"].append(score_label_names[score_target[idx]])
    scatter_data["ID"].append(sample_ids[idx])

scatter_df = pd.DataFrame(scatter_data)
scatter_df.set_index("ID")

df = pd.merge(scatter_df, ibis_metadata, on="ID")
print(df.shape)
hist = df.hvplot(y=das_cols[0], by="cohort", kind="hist")

# show the plots
# ls = hv.link_selections.instance()
# plots = pn.pane.HoloViews(ls(scatter + hist))
# plots.servable()


# Declare points as source of selection stream


BOKEH_TOOLS = {"tools": ["hover", "box_select"]}


# Write function that uses the selection indices to slice points and compute stats
def selected_info(index, col=das_cols[0]):
    if index:
        selected = df.iloc[index]
    else:
        selected = df

    return selected.hvplot(y=col, by="cohort", kind="hist")


scatter = df.hvplot(x="x", y="y", c="cohort", kind="scatter", **BOKEH_TOOLS).opts(
    alpha=0.5
)

selection = streams.Selection1D(source=scatter)

select_cols = pn.widgets.Select(options=das_cols, name="DAS Columns")
stream_col = select_cols.param.value
dmap = hv.DynamicMap(selected_info, index=selection, col=stream_col)

# Layout using Panel
layout = pn.Column(scatter, pn.Row(dmap, select_cols))
layout.servable()
