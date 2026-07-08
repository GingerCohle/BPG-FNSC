# Module Name Mapping

This release uses paper-facing method names while preserving legacy code names
needed for checkpoint compatibility.

| Paper name | Short name | Code-level name | Main implementation |
|---|---|---|---|
| Balanced Prototype-Margin Sampling | BPMS | BPM-PGAFS / `proto_margin_balanced` | `Loss/AutoSSVH_loss.py::build_pgafs_mask_and_labels` |
| Prototype-Aware Hard-View Learning | PA-HVL | SHVD + PCMR | `compute_shvd_loss`, `get_positive_prototype`, `model/AutoSSVH.py::pc_decoder_proto_proj` |
| False-Negative Suppressed Contrast | FNSC | FNS-DCL / `dcl_fns` | `Loss/AutoSSVH_loss.py::_fns_contrast_one_direction`, `dcl_fns` |

## BPMS

BPMS rebuilds the second view mask after warm-up by selecting both:

- positive-representative frames: high positive-prototype similarity;
- margin-hard frames: high positive similarity and low competing-prototype similarity.

Key config fields:

```python
pgafs_enable = True
pgafs_score_type = "proto_margin_balanced"
pgafs_margin_ratio = rho
pgafs_prob = p
```

## PA-HVL

PA-HVL groups two training-time modules:

- SHVD: prototype-distribution alignment from reference view to semantic-hard view.
- PCMR: positive-prototype-conditioned decoder mask token for the hard view.

Key config fields:

```python
shvd_enable = True
shvd_weight = 0.02
shvd_tau = 0.2
pc_decoder_enable = True
pc_decoder_gamma = 0.05
pc_decoder_view = "view2"
```

## FNSC

FNSC replaces the original view contrastive loss when enabled.

Key config fields:

```python
fns_dcl_enable = True
fns_dcl_temperature = 0.2
fns_dcl_threshold = 0.7
fns_dcl_min_weight = 0.2
fns_dcl_suppress_scale = 0.05
fns_dcl_symmetric = True
```

## Inference

BPMS, PA-HVL, and FNSC are training-only mechanisms. During inference, the model
uses only the encoder-hash branch and Hamming-distance retrieval.

