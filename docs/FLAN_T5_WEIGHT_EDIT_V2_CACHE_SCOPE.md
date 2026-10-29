# Clarification of parent and child cache checks

The prospective v2 resource note's reference to checking the parent and child
temporary/cache paths needs a more precise scope. The parent checks and records
its `TEMP` and `TMP` paths inside the E: study root; it does not import the
machine-learning libraries or check the parent's library-cache variables.
The child is given, checks and records its own `TEMP`, `TMP`, and eleven
HF, Transformers, Torch, XDG and plotting-library cache paths inside its E:
attempt before importing the study driver. The parent checks the actual E:
bare Git/common/object store separately. The source and its manifest remain
unchanged; this is a pre-forward clarification of the existing checks, not a
new scientific method or a guarantee against all C: writes.
