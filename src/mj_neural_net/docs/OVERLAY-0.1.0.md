# MJ Five-Template Diffusion 0.1.0

Status: provisional engineering definition. Request profile:
`MJ-Neural-Net/0.1.0`. Output profile: `MJ-Neural-Overlay/0.1.0`.
Projection profile: `MJ-Five-Template-Diffusion/0.1.0`.
This definition does not formalize lowercase biological or runtime `r0`.

## Inputs and source boundary

The closed request contains `profile`, `neural` and `memory_recall`.
`neural` is the unchanged `MJ-Bangel-Neural/0.1.0` object. `memory_recall` is
null or a validated `MJ-Memory-Recall/0.1.0` object. Composition requires exact
equality of `as_of`, every original query field, original memory IDs and all
common memory fields. This binds raw cues, entity, relationship, context,
previous command, timestamps and provenance before native execution.

The source catalogs retain five distinct namespaces, 100 picture occurrences
and all 500 Teal source mappings. The source generator assigns picture mode as
`int(sha256(UTF8(yellow_id)).hexdigest()[:8], 16) % 5`. The 100 instances are
provenance records, not 100 independently justified semantic pathways. All
five distinct template layers receive equal computational weight. No original
SVG file was present in the recovered archive. Original PDF vector geometry
and the generator establish the recovered pictured graphs.

Source pictures contain undirected connectors. Their symmetric use below is
new engineering, and no approved direction is inferred. Textual paths that
conflict with their hash-assigned picture remain separate catalog fields.
The N1-to-N9 source polyline retains both diagonal row-reset connectors.
Containment arcs and disconnected decorative elements do not become edges.

## State and transition rule

There is no persistent mutable model. A request computes the following native
values in order:

1. Prior command transition weights; optional proposed weights; held-out
   comparison; selected history and weights; conditional nine-command forecast.
2. Optional admitted recall forecast; initial nine-node placement scores.
3. Five native source-coordinate bounds, projected node lists and adjacency
   lists; five diffused score vectors; combined score vector and placement.
4. Optional original playbook curves, eleven nine-sample paths, a sixth
   diffusion layer and its admission decision.
5. For composition, a candidate conflict and independent-review decision.

The nine command positions are `let`, `measure`, `derive`, `require`, `if`,
`for`, `match`, `emit`, `return`, in that order. These are command features,
not source lifecycle state labels. The selected row of the baseline's
Laplace-smoothed transition model is the initial linguistic forecast `f`.
Its baseline training and numerical source are unchanged.

The optional proposed transition model is selected only when the existing
native held-out comparison reports improvement and the complete-envelope
review is PASS. This comparison requires at least three valid held-out
transitions; it remains a request-local experiment, not the memory kernel's
30-signal retention qualification or a model installation.

For each source picture separately, let `xmin`, `xmax`, `ymin`, `ymax` be the
minima and maxima of its source node centers, measured in PDF points from the
page's upper left. Normalize:

```
x = (source_x - xmin) / (xmax - xmin)
y = (ymax - source_y) / (ymax - ymin)
bin(v) = 0 if v < 0.25; 2 if v >= 0.75; otherwise 1
feature_node = 1 + bin(x) + 3*bin(y)
```

A degenerate axis maps to 0.5. Thus feature cells 1/2/3 are the bottom row,
4/5/6 the middle, and 7/8/9 the top. Multiple source roles can project to one
feature cell; the source identities remain separately namespaced. Projected
parallel and self-connectors are retained, rather than silently deduplicated.
A self-connector contributes twice to the degree under symmetric adjacency.

Let `cap(v) = max(-0.1, min(0.1, v))` and
`bound(v) = max(-0.9, min(0.9, v))`. Both use existing `std.topology`
operations; `dampen(v)` is `v/3`.

If optional memory recall has a matching candidate and an available forecast,
and the complete-envelope review is PASS, form target `t` with the recalled
forecast probability at its command node and zero at other nodes. Then:

```
initial[i] = bound(f[i] + cap((t[i] - f[i])/3))
```

Otherwise `initial = f`. Recall support is included only when this forecast
is admitted. Memory forecast data is produced by native recall; the Python
adapter validates and serializes it without computing forecast probabilities.

For each template `k`, take every projected connector in both orientations.
Let `neighbor_mean[k,i]` be the mean of `initial` at those neighboring cells,
retaining edge multiplicity. Then:

```
layer[k,i] = bound(initial[i] + cap((neighbor_mean[k,i] - initial[i])/3))
combined[i] = bound(initial[i] + cap((mean(layer[0..4,i]) - initial[i])/3))
```

An isolated feature cell retains `initial[i]` in its layer. One synchronous
diffusion step is executed on every template; there is no repeated convergence
claim. Scores are bounded placement features, not calibrated probabilities.
All coordinate arithmetic, selection, diffusion, reduction and decisions run
through Elsa, independent JP admission and Joanna. Python supplies closed
transport validation, typed serialization and verified result presentation.

## Optional spatial playbook layer

If native memory recall identifies a matching original memory with a non-null
`selection_id`, load that exact packaged selection and its original frame.
The memory catalog verifies its source shard hash. Bind a separate canonical
SHA-256 of the complete selection plus frame into the generated native source.
All eleven original actor curves, including their source kind and labels, are
retained in the response. Selection and frame values are never client-supplied
source code or replacement coordinates.

For each original quadratic curve, `mj.recall.projected_path` computes the
points `B(t) = (1-t)^2*start + 2*(1-t)*t*control + t^2*end` for `t = step/8`,
`step=0..8`. It projects the original schematic frame into exact thirds with
bottom-row node numbering. This fixed-frame projection is distinct from the
per-template bounding-box projection above. Canvas coordinates are not yards.
All eleven `Result` values must be valid. Repeated sampled nodes are retained.

The new `mj.neural_playbook` module treats each adjacent sampled pair in both
orientations, preserving 11 times 8 connectors and self-connector multiplicity.
It applies the same native neighbor-mean diffusion equation to `initial`.
With a matching recall candidate, valid projections and a whole-envelope PASS,
`combined` is recomputed using the mean of the five Teal layers plus this sixth
layer; damping and caps are unchanged. Otherwise `combined` equals
`teal_combined` exactly. There is no zero-filled sixth layer when unavailable.

The playbook layer alone supplies geometry, not new observed command support.
Absent command evidence can therefore still hold placement. `playbook_layer`
reports admission, original memory/selection IDs, frame, all paths and sampled
nodes, selection/source hashes and native scores. Missing or ambiguous recall,
or a memory without a selected play, produces null. HOLD/DENY review may leave
valid geometry visible for inspection while excluding it from placement.
The source curves' order supports sampling; symmetric feature diffusion is a
new engineering rule and is not approval of an executable traversal.

## Outputs and uncertainty

Output includes the linguistic forecast, admitted initial scores, all five
layers, combined scores, projected nodes and edges, source template IDs,
catalog hashes, update/recall admission flags, native review result, placement,
JP/source hashes, receipt root and resource use. `include_receipt=True` adds
the verified receipt to the in-process result. HTTP omits that optional body.
Unavailable held-out losses and placement nodes are presented as null.

A placement is available only with command evidence and a best-versus-second
score margin greater than `0.000000001`. Absent evidence, equal scores or a
smaller margin hold. Missing review also holds; DENY dominates. A placement
does not become a candidate identity or an executable command authorization.

The composition decision accepts a candidate from either successful child
retrieval. If both match different original memory IDs, the candidate is
unavailable and held. Any child or whole-envelope DENY dominates. A candidate
without conflict and a bound PASS is an advisory for review. If recall holds
or is unavailable, a matching baseline candidate may still be retained; that
fallback is part of composition and differs from standalone memory recall.
Placement
ambiguity remains visible separately and does not rewrite a retrieved memory.
Original observations and candidates are never replaced by forecast output.

Reviews have profile `MJ-Neural-Net-Review/0.1.0`, canonical whole-request
SHA-256 and verdict PASS/HOLD/DENY. Validation and review binding occur before
child execution. Rebinding to the original neural and memory review profiles
happens only after whole-envelope verification. Native source binds whole
request/catalog hashes and any child receipt roots into the final receipt.

## Falsification and limits

Geometry preservation is falsified by a missing occurrence, changed node
namespace, unaccounted connector, fabricated direction or mismatch with the
recorded recovered source hashes. Numerical correctness is falsified if an
independent calculation violates projection, symmetric degree counts, damping,
per-step caps or the equal five-layer mean. A no-evidence placement, promoted
update without bound PASS, accepted mismatched child, conflicting-candidate
advisory or overridden DENY falsifies the uncertainty/review contract. For the
optional playbook layer, independently compare all 99 sampled nodes, all 88
adjacent pairs, original frame/source values, sixth-layer mean and exclusion
under HOLD/DENY. Missing or ambiguous recall must not create spatial evidence.

Any claimed predictive benefit requires paired forecast-only and five-template
evaluation on a held-out sequence. Report correct/false placements, coverage,
abstentions and calibration only if measured; keep training, validation and
test observations disjoint and retain ambiguous/unknown cases. A recall
benefit requires a separate paired unchanged-associative versus composed
retrieval comparison, including false matches. A favorable synthetic example
or source-aligned fixture cannot establish general performance. No such
general improvement is asserted by this version. The memory plugin's 32-case
standalone comparison does not establish composition or placement performance;
the optional spatial layer requires its own paired ablation.
