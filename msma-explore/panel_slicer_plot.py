import os
import re
from collections import defaultdict

import ants
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
from holoviews import opts, streams
from minisom import MiniSom
from sade.datasets.loaders import get_image_files_list

# output_notebook()
hv.extension("bokeh")
hv.renderer('bokeh').theme = 'dark_minimal'

js_files = {'jquery': 'https://code.jquery.com/jquery-1.11.1.min.js',
            'goldenlayout': 'https://golden-layout.com/files/latest/js/goldenlayout.min.js'}
css_files = ['https://golden-layout.com/files/latest/css/goldenlayout-base.css',
             'https://golden-layout.com/files/latest/css/goldenlayout-dark-theme.css']

pn.extension('vtk', js_files=js_files, css_files=css_files, design='material',
            #   theme='dark', #sizing_mode="stretch_width"
             sizing_mode="stretch_width", #TODO: look into this
              )

opts.defaults(hv.opts.Image(responsive=False, tools=['hover']))
opts.defaults(opts.Layout(legend_position="top"), opts.Overlay(legend_position="top"))

BOKEH_TOOLS = {"tools": ["hover", "box_select"]}
D3_COLORS = d3["Category10"][10]
COHORTS = ["ABCD", "LR-Typical", "HR-Inlier", "Atypical", "Down's Syndrome", "ASD +ve"]
COHORT_COLORS = {c: D3_COLORS[i] for i, c in enumerate(COHORTS)}

DATA_DIR = (
    "/ASD/ahsan_projects/braintypicality/workdir/cuda_opt/learnable/eval/heatmaps/"
)

# Some width and height constants
HEATMAP_WIDTH = 550
HEATMAP_HEIGHT = 500
ROI_PLOT_HEIGHT = 500

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
ibis_hr_inlier = score_file["inlier_score_norms"]
ibis_atypical = score_file["ood_score_norms"]

score_file = np.load(
    f"{workdir}/ibis-inlier_ibis-hr-inlier_ibis-ds-sa_results.npz", allow_pickle=True
)
ibis_ds = score_file["ood_score_norms"]

score_file = np.load(
    f"{workdir}/ibis-inlier_ibis-hr-inlier_ibis-asd_results.npz", allow_pickle=True
)
ibis_asd = score_file["ood_score_norms"]

region_scores = pd.read_csv(
    f"/ASD/ahsan_projects/braintypicality/workdir/cuda_opt/learnable/eval/heatmaps/roi/DKT_roi_scores.csv",
    index_col="ID",
)
cohort_renamer = {
    "IBIS-LR-Typical": "LR-Typical",
    "IBIS-HR-Typical": "HR-Inlier",
    "IBIS-Atypical": "Atypical",
    "IBIS-DS": "Down's Syndrome",
    "IBIS-ASD": "ASD +ve",
}
region_scores["Cohort"] = region_scores["Cohort"].apply(lambda x: cohort_renamer[x])

ibis_metadata = pd.read_csv(
    f"/ASD/ahsan_projects/braintypicality/dataset/ibis_metadata.csv",
)
# ibis_metadata = pd.read_csv(
#     f"/ASD/ahsan_projects/braintypicality/workdir/cuda_opt/learnable/eval/heatmaps/ibis_metadata.csv",
# )
ibis_metadata.index = ibis_metadata["CandID"].apply(lambda x: "IBIS" + str(x))
ibis_metadata.index.name = "ID"
das_cols = [c for c in ibis_metadata.columns if "DAS" in c]
cbcl_cols = list(
    filter(
        lambda c: re.match(".*internal.*|.*external.*|.*total.*", c),
        ibis_metadata.columns,
    )
)

dataset_map = {
    "ABCD": abcd_data,
    "LR-Typical": ibis_typical,
    "HR-Inlier": ibis_hr_inlier,
    "Atypical": ibis_atypical,
    "Down's Syndrome": ibis_ds,
    "ASD +ve": ibis_asd,
}

# Ensuring that the data is in the same order as the cohorts
score_data, score_target = [], []
for i, c in enumerate(COHORTS):
    score_data.append(dataset_map[c])
    score_target.append([i] * len(dataset_map[c]))
score_data = np.concatenate(score_data)
score_target = np.concatenate(score_target)

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

# data normalization
data = (score_data - np.mean(abcd_data, axis=0)) / np.std(abcd_data, axis=0)
inlier_data = data[score_target < 2]


num_neurons = 5 * np.sqrt(data.shape[0])
grid_size = int(np.ceil(np.sqrt(num_neurons)))

# Initialization and training
n_neurons = 7
m_neurons = 7
max_iters = 5_000
som = MiniSom(
    n_neurons,
    m_neurons,
    data.shape[1],
    sigma=2,
    learning_rate=0.5,
    neighborhood_function="gaussian",
    # random_seed=42,
    topology="rectangular",
)
som.pca_weights_init(inlier_data)

som.train(inlier_data, max_iters, verbose=True)
print(f"Topographic Error: {som.topographic_error(data[:500])}")


# Create data sources for the plots


distance_map = umatrix = som.distance_map()
weights = som.get_weights()
labels_map = som.labels_map(data, [COHORTS[i] for i in score_target])
win_map = som.win_map(data, return_indices=True)

# For plotting the background heatmap
heatmap_data = defaultdict(list)
for i in range(weights.shape[0]):
    for j in range(weights.shape[1]):
        heatmap_data["x"].append(i)
        heatmap_data["y"].append(j)
        heatmap_data["distance"].append(distance_map[(i, j)])
heatmap_df = pd.DataFrame(heatmap_data)

# For the foreground scatter plot to show
# which samples are mapped to which neurons
scatter_data = defaultdict(list)
for idx in range(0, data.shape[0]):
    x = data[idx]
    wx, wy = som.winner(x)
    scatter_data["x"].append(wx)
    scatter_data["y"].append(wy)
    scatter_data["Cohort"].append(COHORTS[score_target[idx]])
    scatter_data["ID"].append(sample_ids[idx])
scatter_df = pd.DataFrame(scatter_data)
scatter_df.set_index("ID")

df = pd.merge(scatter_df, ibis_metadata, on="ID")


# Write functions that use the selection indices to slice points and compute stats
def plot_behavior_scores(index, col=das_cols[0]):
    if index:
        selected = df.iloc[index]
    else:
        selected = df

    selected = selected[selected[col] > -1]

    return (
        selected.hvplot(y=col, by="Cohort", kind="hist", bins=25)
        .opts(
            opts.Histogram(color=hv.dim("Cohort").categorize(COHORT_COLORS)),
        )
        .opts(legend_position="top", width=450, responsive=True)
    )


def plot_roi_scores(index, quantile_threshold=60, show_bars_max=10):

    if len(index) > 0:
        sids = df.iloc[index].ID
        sample_rois = region_scores.loc[sids]
    else:
        sample_rois = region_scores
    
    roi_median = sample_rois.median(numeric_only=True).sort_values(ascending=True)
    selected_rois = roi_median[roi_median > quantile_threshold].index.to_list()
    selected_rois = selected_rois[:show_bars_max]
    roi_data = (
        sample_rois[selected_rois + ["Cohort"]]
        .reset_index()
        .melt(id_vars=["ID", "Cohort"], var_name="ROI", value_name="Percentile")
    )
    boxplot = roi_data.hvplot(
        by="ROI", kind="box", invert=True, title="ROI Scores", legend=False, color="pink"
    )

    if len(index) > 0:
        scatterplot = roi_data.hvplot(
            x="ROI",
            c="Cohort",
            kind="scatter",
            hover_cols=["ID", "Percentile"],
            cmap=COHORT_COLORS,
            alpha=0.6,
           
        )
        boxplot = boxplot * scatterplot

    return boxplot.opts(height=ROI_PLOT_HEIGHT, width=450, xlim=(50, 100), responsive=True)

@pn.cache
def load_brain_volume(sid):
    return np.load(f"{DATA_DIR}/percentiles/{sid}_pct_score.npy")

def plot_brain_volumes(index=[], min_thresh=80):
    if index:
        vols = []
        for sid in df.iloc[index].ID:  # the np files could be cached
            vols.append(load_brain_volume(sid))
        vols = np.stack(vols)
        brain_vol = np.mean(vols, axis=0)
    else:
        brain_vol = ants.image_read(
            "/ASD/ahsan_projects/braintypicality/workdir/cuda_opt/learnable/eval/heatmaps/ds_mean_var.nii.gz"
        ).numpy()[..., 0]

    return brain_vol

def image_slice(dims, array, lbrt, mapper, thresh=20):
    array = np.asarray(array)
    low = thresh  # mapper['low'] if mapper else array.min()
    high = mapper["high"] if mapper else 100
    cmap = mapper["palette"] if mapper else "fire"
    img = hv.Image(array)
    return img.opts(clim=(low, high), cmap=cmap, colorbar=True, width=400, height=400)


###### Creating Plots ######
scatter = df.hvplot(
    x="x", y="y", c="Cohort", kind="scatter", tools=["box_select"], cmap=COHORT_COLORS
).opts(jitter=0.3)

heatmap_base = heatmap_df.hvplot.heatmap(
    x="x",
    y="y",
    C="distance",
    logz=False,
    reduce_function=np.min,
)
base_plot = heatmap_base * scatter

# Declare points as source of selection stream
stream_selection = streams.Selection1D(source=scatter)

select_das_widget = pn.widgets.Select(options=das_cols, name="DAS Columns")
select_cbcl_widget = pn.widgets.Select(options=cbcl_cols, name="CBCL Columns")
select_vol_thresh_widget = pn.widgets.FloatSlider(
    value=80, start=0, end=99, name="Min Thresh", step=1
)

# Use pn.bind to link the widget and selection stream to the functions
das_plot = pn.bind(
    plot_behavior_scores,
    index=stream_selection.param.index,
    col=select_das_widget.param.value,
)
cbcl_plot = pn.bind(
    plot_behavior_scores,
    index=stream_selection.param.index,
    col=select_cbcl_widget.param.value,
)
roi_plot = pn.bind(plot_roi_scores, index=stream_selection.param.index)

def image_slice_i(si, mapper, vol, thresh):
    arr = vol
    lbrt = [0, 1, 0, 1]
    return image_slice(["y", "z"], arr[si, :, ::-1].T, lbrt, mapper, thresh)

def image_slice_j(si, mapper, vol, thresh):
    arr = vol
    lbrt = [0, 1, 0, 1]
    return image_slice(["x", "z"], arr[si, :, ::-1].T, lbrt, mapper, thresh)

def image_slice_k(si, mapper, vol, thresh):
    arr = vol
    lbrt = [0, 1, 0, 1]
    return image_slice(["x", "y"], arr[si, :, ::-1].T, lbrt, mapper, thresh)


volpane = pn.pane.VTKVolume(
        plot_brain_volumes(),
        height=400,
        width=400,
        # display_slices=True,
        colormap="Black-Body Radiation"
    )
controller = volpane.controls(jslink=True)
vol_plot = pn.Row(controller, volpane)
common = dict(
    mapper=volpane.param.mapper,
    vol=volpane.param.object,
    thresh=select_vol_thresh_widget.param.value,
)

dmap_i = hv.DynamicMap(pn.bind(image_slice_i, si=volpane.param.slice_i, **common))
dmap_j = hv.DynamicMap(pn.bind(image_slice_j, si=volpane.param.slice_j, **common))
dmap_k = hv.DynamicMap(pn.bind(image_slice_k, si=volpane.param.slice_k, **common))


@pn.depends(stream_selection.param.index, select_vol_thresh_widget, watch=True)
def update_volume_object(index, value):
    volpane.object = plot_brain_volumes(index=index, min_thresh=value)

# volume = pn.bind(
#     plot_brain_volumes,
#     index=stream_selection.param.index,
#     min_thresh=select_vol_thresh_widget.param.value,
#     watch=True,
# )

# vol_plot = pn.bind(
#     plot_brain_volumes,
#     index=stream_selection.param.index,
#     min_thresh=select_vol_thresh_widget.param.value,
#     watch=True,
# )

explorer_view = pn.Column(
    pn.Row(
        base_plot.opts(
            width=650,
            height=600,
            xaxis=None,
            yaxis=None,
            legend_position="top",
            **BOKEH_TOOLS,
        ),
        roi_plot,
        sizing_mode="stretch_width",
    ),
    pn.Row(
        pn.Column(
            select_das_widget,
            das_plot,
        ),
        pn.Column(
            select_cbcl_widget,
            cbcl_plot,
        ),
        # scroll=True,
        # width=800,
    ),
)
volume_view = pn.Column(
    select_vol_thresh_widget,
    pn.GridBox(
        vol_plot, dmap_i, dmap_j, dmap_k,
        ncols=2, nrows=2

    ),
)

# volume_view = pn.template.FastListTemplate(
#     sidebar=[controller],
#     main=[
#         pn.Row(
#             pn.Column(volpane, dmap_i),
#             pn.Column(dmap_j, dmap_k),
#         )
#     ],
#     title="VTKSlicer",
# )


# Layout using Panel
layout = pn.Tabs(("Explorer", explorer_view), ("Brain Volumes", volume_view),
                  dynamic=False)
layout.servable()
