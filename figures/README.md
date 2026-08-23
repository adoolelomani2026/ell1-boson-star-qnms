# Figures

`manuscript/` contains the vector PDF figures used by the current paper. They
are regenerated from the machine-readable records by:

```powershell
python paper/build_preprint.py
```

The temporary LaTeX build receives copies of the same files; `figures/` is the
stable repository-facing location.
