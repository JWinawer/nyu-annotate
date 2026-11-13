# Cortical sulcus labels (fsaverage)

These files come from the Natural Scenes Dataset (NSD) bucket,
`nsddata/freesurfer/fsaverage/label/` (Kendrick Kay):

- `lh.corticalsulc.mgz`, `rh.corticalsulc.mgz`: one label per fsaverage
  vertex (0 = none, 1-28 = a sulcus or gyrus).
- `corticalsulc.mgz.txt`: the name of each label number.

The Dockerfile copies `datasets/` into the image, so these end up in
`/home/jovyan/datasets/sulci`. `config/nod.yaml` and `config/nsd.yaml` use
them for the "sulci" flatmap panel, and the 3D viewer uses them for the
"Sulci" overlay. Both are skipped if the folder is missing.

All NOD and NSD subjects here are stored on fsaverage vertices, so the same
labels apply to every subject. (The NSD per-subject `corticalsulc` files are
these fsaverage labels moved onto each subject.) They are atlas labels: small,
variable sulci (for example ptCoS, mFus, OTS) can miss a subject's own folds,
so check them against the curvature.
