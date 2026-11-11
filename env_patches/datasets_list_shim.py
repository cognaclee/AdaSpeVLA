"""The LeRobot parquet metadata uses the datasets>=4 "List" feature; the OpenPI-locked
datasets 3.6 reads it as Sequence. Loaded at interpreter start via datasets_list_shim.pth so
spawned DataLoader workers get it too."""
try:
    import datasets.features.features as _f
    _f._FEATURE_TYPES.setdefault("List", _f.Sequence)
except Exception:  # datasets not importable: nothing to patch
    pass
