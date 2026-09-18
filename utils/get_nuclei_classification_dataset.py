from towbintools.foundation.file_handling import read_filemap
import os
import shutil
import numpy as np
import polars as pl
from tqdm import tqdm
from towbintools.foundation.file_handling import add_dir_to_experiment_filemap
from towbintools.plotting.plotting_structure import build_conditions, _add_conditions_to_filemap

experiment_dirs = [
    "/mnt/towbin.data/shared/aslesarchuk/20260727_ZIVA_60x_405_397_706_684",
    "/mnt/towbin.data/shared/aslesarchuk/20260731_ZIVA_60x_405_397_706_684",
    "/mnt/towbin.data/shared/aslesarchuk/20260821_ZIVA_60x_wBT405_684_451_yap_dynamics",
]

# phase of the larval stage, expressed as a proportion of the stage :
# 0 is the molt the worm just went through, 1 is the next molt
stages = {
    'early': 0.25,
    'mid': 0.5,
    'end': 0.75,
}

# larval stages to sample, each one is (stage_name, start_column, end_column)
# the end of a stage is the entry into the next molt when it has been annotated,
# and the ecdysis itself otherwise
larval_stages = [
    ("L1", "HatchTime", "M1"),
    ("L2", "M1", "M2"),
]

stack_per_experiment = 24

database_dir = "/mnt/towbin.data/shared/spsalmon/towbinlab_cell_type_database"
nuclei_seg_subdir = os.path.join("analysis_stacks", "ch2_seg_cellpose")
seed = 42

experiment_filemaps = [read_filemap(os.path.join(experiment_dir, "analysis", "report", "analysis_filemap_annotated.parquet")) for experiment_dir in tqdm(experiment_dirs, desc="reading filemaps")]
conditions_files = [os.path.join(experiment_dir, "doc", "conditions.yaml") for experiment_dir in experiment_dirs]
stack_dirs = [os.path.join(experiment_dir, "raw_stacks") for experiment_dir in experiment_dirs]
nuclei_seg_dirs = [os.path.join(experiment_dir, nuclei_seg_subdir) for experiment_dir in experiment_dirs]

conditions_dict = [build_conditions(conditions_file) for conditions_file in conditions_files]

experiment_filemaps = [_add_conditions_to_filemap(filemap, conditions) for filemap, conditions in zip(experiment_filemaps, conditions_dict)]

# listing the stack directories is by far the slowest part of the loading
experiment_filemaps = [add_dir_to_experiment_filemap(filemap, stack_dir, subdir_name="raw_stacks") for filemap, stack_dir in tqdm(list(zip(experiment_filemaps, stack_dirs)), desc="adding raw stacks")]
experiment_filemaps = [add_dir_to_experiment_filemap(filemap, nuclei_seg_dir, subdir_name="nuclei_seg") for filemap, nuclei_seg_dir in tqdm(list(zip(experiment_filemaps, nuclei_seg_dirs)), desc="adding nuclei segmentation")]

def keep_annotated_stacks(filemap):
    """Keep only the rows that have both a z-stack and its nuclei segmentation, and
    that belong to a worm worth annotating (not ignored, not arrested, still alive)."""

    filemap = filemap.filter(
        pl.col("raw_stacks").is_not_null()
        & (pl.col("raw_stacks") != "")
        & pl.col("nuclei_seg").is_not_null()
        & (pl.col("nuclei_seg") != "")
    )

    # not every experiment has every annotation column
    for column in ["Ignore", "Arrest"]:
        if column in filemap.columns:
            filemap = filemap.filter(~pl.col(column).fill_null(False))

    if "Death" in filemap.columns:
        filemap = filemap.filter(
            pl.col("Death").is_null() | (pl.col("Time") < pl.col("Death"))
        )

    return filemap


def compute_stage_proportions(filemap, stage, start_column, end_column):
    """Restrict the filemap to the rows falling inside a larval stage and express the
    time of every remaining row as a proportion of that stage."""

    # the stage ends when the worm enters the next molt, if that has been annotated
    entry_column = f"{end_column}Entry"
    if entry_column in filemap.columns:
        end = pl.coalesce([pl.col(entry_column), pl.col(end_column)])
    else:
        end = pl.col(end_column)

    filemap = filemap.with_columns(
        pl.lit(stage).alias("stage"),
        pl.col(start_column).alias("stage_start"),
        end.alias("stage_end"),
    )

    filemap = filemap.filter(
        pl.col("stage_start").is_not_null()
        & pl.col("stage_end").is_not_null()
        & (pl.col("stage_end") > pl.col("stage_start"))
        & (pl.col("Time") >= pl.col("stage_start"))
        & (pl.col("Time") <= pl.col("stage_end"))
    )

    return filemap.with_columns(
        (
            (pl.col("Time") - pl.col("stage_start"))
            / (pl.col("stage_end") - pl.col("stage_start"))
        ).alias("proportion")
    )


def build_candidates(filemap):
    """Gather every row that could be picked, for every larval stage."""

    filemap = keep_annotated_stacks(filemap)
    candidates = [
        compute_stage_proportions(filemap, stage, start_column, end_column)
        for stage, start_column, end_column in larval_stages
        if start_column in filemap.columns and end_column in filemap.columns
    ]
    if not candidates:
        return None
    return pl.concat(candidates, how="vertical")


def build_quotas(strains, n_stacks, rng):
    """Split the budget over the strain x stage x phase grid. Every cell gets the same
    quota, the stacks left over are spread one by one over randomly picked cells."""

    cells = [
        (strain, stage, phase)
        for strain in strains
        for stage, _, _ in larval_stages
        for phase in stages
    ]

    quotas = {cell: n_stacks // len(cells) for cell in cells}
    remainder = n_stacks % len(cells)
    for index in rng.permutation(len(cells))[:remainder]:
        quotas[cells[index]] += 1

    return quotas


def pick_cell(candidates, strain, stage, phase, quota, already_picked, rng):
    """Pick the stacks of a single cell of the grid : the ones closest to the target
    proportion, at most one per worm."""

    if quota == 0:
        return None

    target = stages[phase]
    cell = candidates.filter((pl.col("strain") == strain) & (pl.col("stage") == stage))
    # a stack that has already been picked by another cell cannot be picked again
    if already_picked is not None:
        cell = cell.join(already_picked, on=["Point", "Time"], how="anti")
    if cell.is_empty():
        return None

    cell = cell.with_columns(
        (pl.col("proportion") - target).abs().alias("proportion_error"),
        pl.Series("tie_break", rng.random(cell.height)),
    )

    # keep the closest time point of every worm, then the closest worms
    cell = (
        cell.sort(["proportion_error", "tie_break"])
        .unique(subset=["Point"], keep="first", maintain_order=True)
        .head(quota)
    )

    target_time = pl.col("stage_start") + target * (pl.col("stage_end") - pl.col("stage_start"))
    return cell.with_columns(
        pl.lit(f"{stage}_{phase}").alias("window"),
        pl.lit(phase).alias("phase"),
        target_time.alias("target_time"),
        (pl.col("Time") - target_time).abs().alias("snap_error"),
    )


def select_stacks(filemap, n_stacks, rng):
    """Select the stacks of a single experiment, balanced over the strains."""

    candidates = build_candidates(filemap)
    if candidates is None or candidates.is_empty():
        tqdm.write("  no candidate stack found, skipping experiment")
        return None

    strains = sorted(candidates["strain"].drop_nulls().unique().to_list())
    quotas = build_quotas(strains, n_stacks, rng)

    selection = []
    already_picked = None
    for (strain, stage, phase), quota in quotas.items():
        picked = pick_cell(candidates, strain, stage, phase, quota, already_picked, rng)
        picked_count = 0 if picked is None else picked.height
        if picked_count < quota:
            tqdm.write(f"  {strain} {stage}_{phase} : only {picked_count} stack(s) available out of {quota}")
        if picked is not None:
            selection.append(picked)
            picked_keys = picked.select(["Point", "Time"])
            already_picked = picked_keys if already_picked is None else pl.concat([already_picked, picked_keys])

    if not selection:
        return None
    return pl.concat(selection, how="vertical").sort(["window", "Point", "Time"])


def export_selection(selection, experiment_dir):
    """Copy the selected stacks and their segmentation into the database, and write the
    manifest describing where every one of them comes from."""

    experiment_name = os.path.basename(os.path.normpath(experiment_dir))
    output_dir = os.path.join(database_dir, experiment_name)
    raw_dir = os.path.join(output_dir, "raw")
    masks_dir = os.path.join(output_dir, "masks")
    os.makedirs(raw_dir, exist_ok=True)
    os.makedirs(masks_dir, exist_ok=True)

    name = (
        pl.col("window")
        + "_P" + pl.col("Point").cast(pl.Int64).cast(pl.Utf8).str.zfill(4)
        + "_T" + pl.col("Time").cast(pl.Int64).cast(pl.Utf8).str.zfill(5)
        + ".ome.tiff"
    )

    manifest = selection.with_columns(
        pl.col("raw_stacks").alias("raw_src"),
        pl.col("nuclei_seg").alias("seg_src"),
        (pl.lit(raw_dir + os.sep) + name).alias("raw_dst"),
        (pl.lit(masks_dir + os.sep) + name).alias("seg_dst"),
    ).select([
        "window", "stage", "phase", "strain", "Point", "Time", "target_time",
        "proportion", "snap_error", "raw_src", "seg_src", "raw_dst", "seg_dst",
    ])

    copied = 0
    for row in tqdm(manifest.iter_rows(named=True), total=manifest.height, desc="  copying stacks", leave=False):
        for source, destination in [(row["raw_src"], row["raw_dst"]), (row["seg_src"], row["seg_dst"])]:
            if not os.path.exists(destination):
                shutil.copy2(source, destination)
                copied += 1

    manifest_path = os.path.join(output_dir, "manifest.csv")
    manifest.write_csv(manifest_path)
    # every stack is two files : the raw one and its segmentation
    tqdm.write(f"  {manifest.height} stacks in {output_dir}, {copied} file(s) newly copied")

    return manifest
for experiment_dir, filemap in tqdm(list(zip(experiment_dirs, experiment_filemaps)), desc="experiments"):
    tqdm.write(f"Selecting stacks of {os.path.basename(os.path.normpath(experiment_dir))}")
    rng = np.random.default_rng(seed)
    selection = select_stacks(filemap, stack_per_experiment, rng)
    if selection is None:
        continue
    export_selection(selection, experiment_dir)
