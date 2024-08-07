import os
import re
from collections import defaultdict

import ants
import holoviews as hv
import hvplot.pandas  # noqa
import numpy as np
import pandas as pd
import panel as pn
from bokeh.palettes import d3
from holoviews import opts, streams
from holoviews.plotting.links import RangeToolLink
from minisom import MiniSom
from sade.configs.ve import biggan_config
from sade.datasets.loaders import get_image_files_list, get_val_transform

# output_notebook()
hv.extension("bokeh")
hv.renderer("bokeh").theme = "dark_minimal"

js_files = {
    "jquery": "https://code.jquery.com/jquery-1.11.1.min.js",
    "goldenlayout": "https://golden-layout.com/files/latest/js/goldenlayout.min.js",
}
css_files = [
    "https://golden-layout.com/files/latest/css/goldenlayout-base.css",
    "https://golden-layout.com/files/latest/css/goldenlayout-dark-theme.css",
]

pn.extension(
    "vtk",
    #  js_files=js_files, css_files=css_files,
    design="material",
    #   theme='dark', #sizing_mode="stretch_width"
    # sizing_mode="stretch_width",  # TODO: look into this
)

opts.defaults(hv.opts.Image(responsive=False, tools=["pan"]))
opts.defaults(opts.Layout(legend_position="top"), opts.Overlay(legend_position="top"))

BOKEH_TOOLS = {"tools": ["hover", "box_select"]}
D3_COLORS = d3["Category10"][10]
COHORTS = ["ABCD", "LR-Typical", "HR-Inlier", "Atypical", "Down's Syndrome", "ASD +ve"]
COHORT_COLORS = {c: D3_COLORS[i] for i, c in enumerate(COHORTS)}

DATA_DIR = (
    "/ASD/ahsan_projects/braintypicality/workdir/cuda_opt/learnable/eval/heatmaps/"
)

CACHE_DIR = "/ASD/ahsan_projects/braintypicality/dataset/template_cache/"

config = biggan_config.get_config()
img_loader = get_val_transform(config)
procd_ref_img_path = f"{CACHE_DIR}/cropped_niral_mni.nii.gz"
REF_BRAIN_IMG = img_loader({"image": procd_ref_img_path})["image"].numpy()[0]
REF_BRAIN_MASK = (REF_BRAIN_IMG > -1).astype(np.float32)
REF_BRAIN_IMG = (REF_BRAIN_IMG + 1) / 2 * 100

# REF_BRAIN_IMG = ants.image_read(f"{CACHE_DIR}/cropped_niral_mni.nii.gz").numpy()[..., 0]
# REF_BRAIN_IMG = (REF_BRAIN_IMG - REF_BRAIN_IMG.min()) / (REF_BRAIN_IMG.max() - REF_BRAIN_IMG.min())
# REF_BRAIN_IMG = REF_BRAIN_IMG * 100

# Some width and height constants
HEATMAP_WIDTH = 650
HEATMAP_HEIGHT = 600
ROI_PLOT_HEIGHT = 600
ROI_PLOT_WIDTH = 700
IMG_HEIGHT = 400  # for the volume slicer
IMG_WIDTH = 500

CURRENT_VOLUME = None

workdir = os.path.expanduser(
    "/ASD/ahsan_projects/braintypicality/workdir/cuda_opt/learnable/eval/ckpt_1500002/smin=0.01"
)


@pn.cache
def get_region_scores():
    region_scores = pd.read_csv(
        "/ASD/ahsan_projects/braintypicality/workdir/cuda_opt/learnable/eval/heatmaps/roi/DKT_roi_scores.csv",
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
    return region_scores


@pn.cache
def get_ibis_metadata():
    ibis_metadata = pd.read_csv(
        "/ASD/ahsan_projects/braintypicality/dataset/ibis_metadata.csv",
    )
    # ibis_metadata = pd.read_csv(
    #     f"/ASD/ahsan_projects/braintypicality/workdir/cuda_opt/learnable/eval/heatmaps/ibis_metadata.csv",
    # )
    ibis_metadata.index = ibis_metadata["CandID"].apply(lambda x: "IBIS" + str(x))
    ibis_metadata.index.name = "ID"
    ibis_metadata = ibis_metadata.astype(np.float32, errors="ignore")
    return ibis_metadata


@pn.cache
def load_score_data() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    score_file = np.load(
        f"{workdir}/abcd-val_abcd-test_lesion_load_20_results.npz", allow_pickle=True
    )
    abcd_data = np.concatenate(
        [score_file[f] for f in ["eval_score_norms", "inlier_score_norms"]], axis=0
    )
    score_file = np.load(f"{workdir}/abcd-train_abcd-val_lesion_load_20_results.npz", allow_pickle=True)
    abcd_data = np.concatenate([abcd_data, score_file['eval_score_norms']], axis=0)

    score_file = np.load(
        f"{workdir}/ibis-inlier_ibis-hr-inlier_ibis-atypical_results.npz",
        allow_pickle=True,
    )
    ibis_typical = score_file["eval_score_norms"]
    ibis_hr_inlier = score_file["inlier_score_norms"]
    ibis_atypical = score_file["ood_score_norms"]

    score_file = np.load(
        f"{workdir}/ibis-inlier_ibis-hr-inlier_ibis-ds-sa_results.npz",
        allow_pickle=True,
    )
    ibis_ds = score_file["ood_score_norms"]

    score_file = np.load(
        f"{workdir}/ibis-inlier_ibis-hr-inlier_ibis-asd_results.npz", allow_pickle=True
    )
    ibis_asd = score_file["ood_score_norms"]

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
        "abcd-train",
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

    return data, score_target, sample_ids


@pn.cache
def build_som_map(
    data, max_iters=50_000, n_neurons=7, m_neurons=7, sigma=2, learning_rate=0.5
):
    # Ideally ...
    # num_neurons = 5 * np.sqrt(data.shape[0])
    # grid_size = int(np.ceil(np.sqrt(num_neurons)))

    som = MiniSom(
        n_neurons,
        m_neurons,
        data.shape[1],
        sigma=2,
        learning_rate=0.5,
        neighborhood_function="gaussian",
        # random_seed=42,
        topology="hexagonal",
    )
    som.pca_weights_init(data)
    som.train(data, max_iters, verbose=True)

    print(f"Topographic Error: {som.topographic_error(data[:500])}")

    return som


# Create data sources for the plots
ibis_metadata = get_ibis_metadata()
region_scores = get_region_scores()
das_cols = [c for c in ibis_metadata.columns if "DAS" in c]
cbcl_cols = list(
    filter(
        lambda c: re.match(
            ".*internal.*percentile|.*external.*percentile|.*total.*percentile", c
        ),
        ibis_metadata.columns,
    )
)
vineland_cols = list(
    (filter(lambda c: re.match(".*Vine.*PERCENTILE", c), ibis_metadata.columns))
)


########## SOM ##########
M, N = 9, 9
data, score_target, sample_ids = load_score_data()
inlier_data = data[score_target < 2]
som = build_som_map(inlier_data, m_neurons=M, n_neurons=N)
distance_map = umatrix = som.distance_map()
weights = som.get_weights()
###########################

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
df = pd.merge(scatter_df, ibis_metadata, on="ID").set_index("ID")

grid_to_sample = defaultdict(list)
for idx in range(0, data.shape[0]):
    x,y = data[idx], score_target[idx]
    if COHORTS[y] == "ABCD": continue
    wx, wy = som.winner(x)
    grid_to_sample[(wx, wy)].append(sample_ids[idx])

# Write functions that use the selection indices to slice points and compute stats

# Use the slection indices to get the sample IDs
def get_sids_from_selection_event(index):
    sids = []
    pos = heatmap_df.iloc[index].values[:,:2]
    for x,y in pos:
        sids.extend(grid_to_sample[int(x),int(y)])
    return sids

empty_hist = hv.Histogram([]).relabel("No data")
def plot_behavior_scores(index, col=das_cols[0]):

    sids = get_sids_from_selection_event(index) if len(index) > 0 else None
    
    if sids:
        selected = df.loc[sids]
    else:
        selected = df
    
    selected = selected[selected[col] > -1]

    if len(selected) == 0:
        return empty_hist

    return (
        selected.hvplot(y=col, by="Cohort", kind="hist", bins=np.arange(0, 101, 5))
        .opts(
            opts.Histogram(color=hv.dim("Cohort").categorize(COHORT_COLORS), min_width=150, axiswise=False, shared_axes=False, framewise=False),
        )
        .opts(legend_position="top", )
    )


def plot_cohort_count(index):
    cohorts = df["Cohort"]
    sids = get_sids_from_selection_event(index) if len(index) > 0 else None

    if sids:
        cohorts = cohorts.loc[sids]
    
    cohorts = cohorts.value_counts()

    counts = []
    for c in COHORTS[1:]:
        if c in cohorts:
            counts.append(f"**{c}**: {cohorts[c]}")
        else:
            counts.append(f"**{c}**: 0")

    return "    ".join(counts)


def plot_roi_scores(index, quantile_threshold=60, show_bars_max=50):
    sids = get_sids_from_selection_event(index) if len(index) > 0 else None

    if sids:
        sample_rois = region_scores.loc[sids]
    else:
        sample_rois = region_scores

    roi_median = sample_rois.median(numeric_only=True).sort_values(ascending=False)
    roi_median = roi_median[:show_bars_max]
    selected_rois = roi_median.index.to_list()

    significant_rois = roi_median[roi_median > quantile_threshold].index.to_list()
    other_rois = roi_median[roi_median <= quantile_threshold].index.to_list()
    significance_colors = ["lightgrey", "pink"]
    # roi_data["Significant"] = roi_data["ROI"].apply(lambda r: "Y" if roi_median[r] > quantile_threshold else "N")


    boxplots = []

    for rois, color in zip((other_rois, significant_rois), significance_colors):
        roi_data = (
            sample_rois[rois[::-1] + ["Cohort"]]
            .reset_index()
            .melt(id_vars=["ID", "Cohort"], var_name="ROI", value_name="Percentile")
        )
        # roi_data["Significant"] = roi_data["ROI"].apply(
        #     lambda r: "Y" if roi_median[r] > quantile_threshold else "N"
        # )
        box_whisker = roi_data.hvplot(
            kind="box",
            by="ROI",
            y="Percentile",
            invert=True,
            title="ROI Scores",
            legend=False,
            # box_color=hv.dim("Significant").categorize(
            #     {"Y": D3_COLORS[-1], "N": D3_COLORS[-2]})
        )
        boxplots.append(
            box_whisker.opts(
                # width=ROI_PLOT_WIDTH,
                box_fill_color=color,
            )
        )

    minimap_data = sample_rois[selected_rois[::-1]].melt(
        var_name="ROI", value_name="Percentile"
    )
    boxplot = hv.Overlay(boxplots)
    
    if sids:
        roi_data = (
            sample_rois[selected_rois[::-1] + ["Cohort"]]
            .reset_index()
            .melt(id_vars=["ID", "Cohort"], var_name="ROI", value_name="Percentile")
        )
        scatterplot = roi_data.hvplot(
            y="Percentile",
            x="ROI",
            c="Cohort",
            kind="scatter",
            hover_cols=["ID", "Percentile"],
        ).opts(
            cmap=COHORT_COLORS,
            alpha=0.7,
        )
        boxplot = boxplot * scatterplot


    minimap = (
        hv.BoxWhisker(minimap_data, "ROI", "Percentile")
        .relabel("Overview")
        .opts(
            width=ROI_PLOT_WIDTH//4,
            height=ROI_PLOT_HEIGHT,
            invert_axes=True,
            axiswise=True,
            default_tools=[],
            toolbar=None,
            labelled=["x"],
            yaxis=None,
        )
    )

    rtlink = RangeToolLink(minimap, boxplots[0],  axes=["x", "y"], boundsx=(0, 100))

    boxplot = boxplot.relabel("ROI Scores").opts(
        opts.Overlay(min_width=ROI_PLOT_WIDTH, min_height=ROI_PLOT_HEIGHT, show_legend=False)
    )
    layout = pn.Row(boxplot, minimap)
    # layout = (boxplot + minimap).cols(2)
    # layout.opts(opts.Layout(shared_axes=False, merge_tools=False))
    return layout


@pn.cache
def load_brain_volume(sid):
    if not sid:
        return ants.image_read(
            "/ASD/ahsan_projects/braintypicality/workdir/cuda_opt/learnable/eval/heatmaps/ds_mean.nii.gz"
        ).numpy()

    return np.load(f"{DATA_DIR}/percentiles/{sid}_pct_score.npy")


def update_current_volume(index=[]):
    global CURRENT_VOLUME

    sids = get_sids_from_selection_event(index) if len(index) > 0 else None
    if sids:
        vols = []
        for sid in sids:  # the np files could be cached
            vols.append(load_brain_volume(sid))
        vols = np.stack(vols)
        brain_vol = np.mean(vols, axis=0)
    else:
        brain_vol = load_brain_volume(None)
        # brain_vol = REF_BRAIN_IMG

    CURRENT_VOLUME = brain_vol * REF_BRAIN_MASK
    return


def image_slice(ref_slice, heatmap_slice, lbrt, mapper, thresh=20):
    # heatmap_slice[heatmap_slice < thresh] = 0
    low, high = thresh, 100
    cmap = mapper["palette"] if mapper else "fire"
    ratio = ref_slice.shape[0] / ref_slice.shape[1]
    ref_img = hv.Image(ref_slice, bounds=lbrt).opts(cmap="gray", colorbar=False)
    heatmap_img = hv.Image(heatmap_slice, bounds=lbrt).opts(
        cmap=cmap, clim=(low, high), colorbar=True, alpha=0.5
    )
    return (ref_img * heatmap_img).opts(
        # clim=(low, high),
        # colorbar=True,
        width=IMG_WIDTH,
        height=int(IMG_HEIGHT * ratio),
        # xlim=(-1, 1),
        # ylim=(-1, 1),
        axiswise=True,
        # framewise=True,
        bgcolor="black",
    )


###### Creating Plots ######

#### Rectangles at x y coords

cell_groups = df[["x", "y", "Cohort"]].groupby(["x", "y"])
rects = []
# xs = np.linspace(0,1,4, endpoint=False)
xpad = 0.05
ypad = 0.01
nbars = len(COHORTS) - 1
scaler = 0.9
offset = 0.5
xs = np.linspace(0 + xpad, 1 - xpad, nbars + 1) - offset
xs = xs * scaler
ys = np.zeros(nbars) - offset
# ys = ys * scaler
maxcount = cell_groups["Cohort"].value_counts().max()
colors = [COHORT_COLORS[c] for c in COHORTS[1:]]

#!FIXME: use cohorts as dims and categorize them with colors
# this might allow for a legend to be shown
for pos, cell in cell_groups:
    x, y = pos
    counts = cell["Cohort"].value_counts()
    x0 = xs[:-1] + x
    x1 = xs[1:] + x
    y0 = ys + y
    y1 = ys + y
    for i, c in enumerate(COHORTS[1:]):
        if c in counts:
            y1[i] += (counts[c] / maxcount - ypad) * scaler

    rects.extend(list(zip(x0, y0, x1, y1, colors)))
histograms = hv.Rectangles(rects, vdims="cohort")
histograms.opts(color="cohort")

######### Base heatmap view #########
heatmap_base = hv.HeatMap(heatmap_df)
heatmap_selection = streams.Selection1D(source=heatmap_base)
#####################################

# Declare widgets
select_das_widget = pn.widgets.Select(options=das_cols, name="DAS Columns")
select_cbcl_widget = pn.widgets.Select(options=cbcl_cols, name="CBCL Columns")
select_vineland_widget = pn.widgets.Select(
    options=vineland_cols, name="Vineland Columns"
)
select_vol_thresh_widget = pn.widgets.FloatSlider(
    value=80, start=0, end=99, name="Min Thresh", step=1
)

###### Use pn.bind to link the widget and selection stream to the functions
das_plot = pn.bind(
    plot_behavior_scores,
    index=heatmap_selection.param.index,
    col=select_das_widget.param.value,
)
cbcl_plot = pn.bind(
    plot_behavior_scores,
    index=heatmap_selection.param.index,
    col=select_cbcl_widget.param.value,
)
vineland_plot = pn.bind(
    plot_behavior_scores,
    index=heatmap_selection.param.index,
    col=select_vineland_widget.param.value,
)
####

#### Bind the functions to the selection stream
cohort_count_plot = pn.bind(plot_cohort_count, index=heatmap_selection.param.index)
roi_plot = pn.bind(plot_roi_scores, index=heatmap_selection.param.index)

def image_slice_i(si, mapper, vol, thresh):

    arr = CURRENT_VOLUME
    x1,y1,x2,y2 = lbrt = [0.0,0.0, arr.shape[1], arr.shape[2]]
    # lbrt = [-1, -1, 1, 1]
    return image_slice(
        REF_BRAIN_IMG[si, :, ::-1].T, arr[si, :, ::-1].T, lbrt, mapper, thresh
    )


def image_slice_j(sj, mapper, vol, thresh):
    arr = CURRENT_VOLUME
    # lbrt = [-1, -1, 1, 1]
    lbrt = [0.0, 0.0, arr.shape[0], arr.shape[2]]
    return image_slice(
        REF_BRAIN_IMG[:, sj, ::-1].T, arr[:, sj, ::-1].T, lbrt, mapper, thresh
    )


def image_slice_k(sk, mapper, vol, thresh):
    arr = CURRENT_VOLUME
    # lbrt = [-1, -1, 1, 1]
    lbrt = [0.0, 0.0, arr.shape[0], arr.shape[1]]
    return image_slice(
        REF_BRAIN_IMG[:, ::-1, sk].T, arr[:, ::-1, sk].T, lbrt, mapper, thresh
    )

update_current_volume()

volpane = pn.pane.VTKVolume(
    CURRENT_VOLUME,
    max_height=IMG_HEIGHT,
    max_width=IMG_WIDTH,
    display_slices=True,
    colormap="Black-Body Radiation",
    # render_background="grey",
)

# A panel object that holds the current brain volume


# @pn.depends(stream_selection.param.index, watch=True)
def update_volume_object(selection_event):
    update_current_volume(index=selection_event.new)
    volpane.object = CURRENT_VOLUME.copy()
    volpane.param.colormap = "Black-Body Radiation"
    volpane.param.trigger("object")

@pn.depends(select_vol_thresh_widget.param.value, watch=True)
def threshold_volume_object(thresh):
    print("thresholding: ", thresh)
    print(volpane.param.colormap.value)
    volpane.object = CURRENT_VOLUME * (CURRENT_VOLUME > thresh)

heatmap_selection.param.watch(update_volume_object, "index", queued=True)

common = dict(
    mapper=volpane.param.mapper,
    vol=volpane.param.object,
    thresh=select_vol_thresh_widget.param.value,
)

dmap_i = hv.DynamicMap(pn.bind(image_slice_i, si=volpane.param.slice_i, **common))
dmap_j = hv.DynamicMap(pn.bind(image_slice_j, sj=volpane.param.slice_j, **common))
dmap_k = hv.DynamicMap(pn.bind(image_slice_k, sk=volpane.param.slice_k, **common))


# behaviour_view = pn.GridSpec(min_width=1000, ncols=2, nrows=2)

# behaviour_view[0, 0] = das_plot
# behaviour_view[0, 1] = cbcl_plot
# behaviour_view.flat[3] = vineland_plot

base_plot = heatmap_base * histograms


explorer_view = pn.Column(
    pn.Row(
        base_plot.opts(
            opts.HeatMap(tools=["hover", "box_select", "tap"], cmap="Blues_r"),
            opts.Overlay(
                min_width=HEATMAP_WIDTH,
                min_height=HEATMAP_HEIGHT,
                xaxis=None,
                yaxis=None,
                legend_position="top",
                # legend_opts={"click_policy": "hide"},
                axiswise=False,
                shared_axes=False,
                # **BOKEH_TOOLS,
            ),
        ),
        roi_plot,
        sizing_mode="stretch_width",
    ),
    pn.pane.Markdown(cohort_count_plot),
    pn.Row(
        pn.Column(
            select_das_widget,
            das_plot,
        ),
        pn.Column(
            select_cbcl_widget,
            cbcl_plot,
        ),
        pn.Column(
            select_vineland_widget,
            vineland_plot,
        ),
        # scroll=True,
        sizing_mode="stretch_width",
    ),
)

controller = volpane.controls(
    jslink=False,
    parameters=[
        "display_volume",
        "display_slices",
        "slice_i",
        "slice_j",
        "slice_k",
        "colormap",
        "render_background",
    ],
)

gspec = pn.GridSpec(width=1200, height=1000, ncols=2, nrows=2)

gspec[0, 0] = volpane
gspec[0, 1] = dmap_i
gspec[1, 0] = dmap_j
gspec[1, 1] = dmap_k

volume_view = pn.Row(
    pn.WidgetBox("## Controls", select_vol_thresh_widget, controller, max_width=350),
    gspec,
)

# Layout using Panel
layout = pn.Tabs(
    ("Explorer", explorer_view),
    ("Brain Volumes", volume_view),
    dynamic=False,
)
layout.servable()
