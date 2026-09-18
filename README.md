# MJ Neural Net

**MJ Neural Net 0.1.0** is an independently installable API plugin cloned from
MJ-Bangel-Neural-Network-Plugin. It adds five source-grounded diagram layers,
bounded command-driven node placement, and optional composition with the
standalone **MJ Memory Recall 0.1.0** physics-playbook recall plugin. All neural,
projection, diffusion and decision mathematics execute in `.bangel`, compiled
by Elsa, independently admitted as JP, and executed by Joanna.

The original `evaluate()` contract and `source/neural.bangel` bytes remain
unchanged. It retains Hebbian associative recall, relational discrimination,
three nine-node layers and command-transition prediction. New behavior uses
the separately versioned `MJ-Neural-Net/0.1.0` envelope.

Lowercase `r0` remains **PROVISIONAL** under `MJ-Neural-r0/0.1.0`. It is neither
a biological state nor uppercase `R0`. See the [provisional contract](src/mj_neural_net/docs/r0-0.1.0.md)
and [neuroscience references](src/mj_neural_net/docs/NEUROSCIENCE.md). This
engineering simulation does not establish biological fidelity or improved recall.

## Install and run

Python 3.12+ and exact dependencies `bangel-language==0.2.0rc1` and
`mj-memory-recall==0.1.0` are required. From the Bangel repository:

```sh
python -m pip install ./language/python ./plugins/mj_memory_recall ./plugins/mj_neural_net
```

For a standalone checkout, install the supplied matching language and memory
recall wheels, then `python -m pip install .`. No sibling repository imports
or package-index publication are assumed.

```python
from mj_neural_net import compose, example_envelope, overlay

request = example_envelope()
placement = overlay(request)
assert len(placement["layers"]) == 5
result = compose(request)
assert result["review_status"] == "DOWNSTREAM_HOLD"
```

The new closed envelope has exactly these fields:

```json
{
  "profile": "MJ-Neural-Net/0.1.0",
  "neural": "<original MJ-Bangel-Neural/0.1.0 request object>",
  "memory_recall": null
}
```

The quoted placeholder represents an object; use the complete
[overlay example](src/mj_neural_net/examples/overlay-request.json).
To compose recall, replace `null` with an `MJ-Memory-Recall/0.1.0` request.
Its query, first-observation time, original memory IDs and common memory
values must exactly match the neural request. Additional namespace, frame,
trace and play-selection values belong to the memory request. The
[composition example](src/mj_neural_net/examples/recall-composition.json) is a
synthetic fixture, with observed-role fields used only to exercise the contract.
The [spatial playbook example](src/mj_neural_net/examples/playbook-overlay.json)
adds an original offense selection and isolates its geometric contribution.

```sh
mj-neural-net overlay envelope.json --output placement.json
mj-neural-net compose envelope.json --output result.json
mj-neural-net lower-overlay envelope.json --output lowered-overlay
mj-neural-net run original-request.json --output original-result.json
mj-neural-net lower original-request.json --output lowered-original
```

Use fresh output paths. Both lowering commands produce native source and
independently admitted JP. `--review review.json` supplies a separate local
review record. Execution exit code 0 means an advisory is ready after PASS
review; 3 means it remains held or denied; 2 means input or execution failed.
Lowering returns 0 on successful admission. `python -m mj_neural_net` provides
the same commands. The unchanged baseline contract is in
[INPUT-CONTRACT.md](src/mj_neural_net/docs/INPUT-CONTRACT.md).

## Teal, Purple and Blue references

The [original paper map](src/mj_neural_net/docs/PAPER-REFERENCES.json) retains
the cloned Teal, Purple and Blue references. The new
[diagram catalog](src/mj_neural_net/data/teal-diagrams.json) and
[500-paper source map](src/mj_neural_net/data/teal-mappings.json) record the
recovered Teal edition and its referenced Yellow pictures, with source hashes,
namespaces, labels, vector coordinates, connector geometry and containment.

The recovered 500 Teal papers contain no node pictures themselves. TP-301
through TP-400 reference the 100 Yellow diagram pages; TP-499 also references
their bundles. The recovered generator assigns one of five picture templates
by the SHA-256 of each Yellow paper ID, rather than by the paper's semantics.
All 100 occurrences are catalogued, including conflicts between picture and
nearby textual sequence. This establishes coverage of that recovered edition;
it does not establish that every historical edition has been recovered.

| Template namespace suffix | Source node labels | Undirected connectors | Occurrences |
| --- | --- | ---: | ---: |
| `Template0` | INTAKE, CONE, UMBRELLA, LAKE | 3 | 19 |
| `Template1` | N1 through N9, bottom row first | 8 | 24 |
| `Template2` | R, R0, R1, N1, R2, OUT | 5 | 17 |
| `Template3` | STABILITY, INVESTMENT, SHIVA, MJ | 3 | 27 |
| `Template4` | SOURCE, LECANTO, EVIDENCE | 2 | 13 |

Every source node retains its `MJ.Yellow.Reference.TemplateX:` namespace.
The grid picture is a straight polyline through N1 to N9, including diagonal
row resets. No connector has a source arrowhead. The pictures are reference
geometry, not approved executable command traversals. Source state labels,
the larger predictive dependency chain and runtime states remain distinct.
Historical product names in display metadata were explicitly migrated by
`tools/import_diagrams.py`; original source hashes were retained.

The original five `SIMULATION_REFERENCE` permutations in
[PATHWAYS.json](src/mj_neural_net/PATHWAYS.json) still belong to the compatibility
baseline. The new overlay uses all five recovered template adjacency layers.
It normalizes their source node centers into a shared nine-cell feature space,
applies symmetric bounded diffusion to a linguistic forecast, and combines the
five layers with equal weight. Duplicate picture occurrences do not increase
template weight. Projection and symmetric traversal are new engineering rules;
they do not confer source approval. See the exact
[overlay definition](src/mj_neural_net/docs/OVERLAY-0.1.0.md).

## Review and API

`create_app()` returns a WSGI application without starting a server:

```python
from mj_neural_net import create_app
application = create_app(review_provider=independent_review)
```

| Route | Request | Result |
| --- | --- | --- |
| `GET /v1/profile` | No query parameters | Profiles, limits and provisional marker |
| `GET /v1/catalog` | No query parameters | All five templates, 100 instances and 500 Teal mappings |
| `POST /v1/evaluate` | Original neural request | Unchanged baseline behavior |
| `POST /v1/overlay` | New complete envelope | Executed layers and node placement |
| `POST /v1/compose` | New complete envelope | Baseline, optional recall, layers and candidate decision |

The trusted host callback receives a deep copy of the complete validated
request. For new routes it returns `None` or exactly:

```json
{
  "profile": "MJ-Neural-Net-Review/0.1.0",
  "request_sha256": "<SHA-256 of canonical complete envelope>",
  "verdict": "PASS"
}
```

Canonical JSON uses sorted keys, compact separators and UTF-8 without ASCII
escaping. Only after the whole envelope is verified does composition bind
the same verdict to each child request. The original evaluate route retains
`MJ-Neural-Review/0.1.0`. `HOLD` and `DENY` are valid verdicts. The host owns
reviewer authentication; a content hash does not authenticate its issuer.

Absent review holds. Any bound DENY dominates. Conflicting matching candidate
IDs hold, even with PASS. If recall is unavailable or held while the baseline
has a verified match, composition may retain that baseline candidate; recall
HOLD is not itself DENY. A unique candidate may be reported for inspection
while downstream use remains held. Template placement is a separate advisory;
it does not rewrite candidate identity, evidence, or the recalled observation.

POST accepts UTF-8 JSON with `Content-Type: application/json` and a valid
`Content-Length` of at most 65,536 bytes. Duplicate keys, nonfinite numbers,
unknown fields, source text, paths and request-supplied review are rejected.
Invalid or unavailable review returns 503 without an advisory. A completed
held result uses HTTP 200; inspect `review_status` and `advisory_ready`.
No action permission, model installation or background service is granted.

## Reusable memory and self-healing integration

Optional recall uses `mj_memory_recall.recall` from the standalone dependency.
Its observed interaction traces and physics-playbook projections can recover
a candidate when the associative baseline is ambiguous. Only a verified
whole-envelope PASS and an available matching recall forecast can add recall
evidence to node-placement scores. Unknown commands and unsupported forecasts
remain unavailable.

When the matching recalled memory names a packaged play selection, the new
spatial layer binds its original coordinate frame and all eleven actor curves.
Native Bangel samples each quadratic curve at nine phases, retaining repeated
nodes, then uses adjacent samples for symmetric graph diffusion. All 99 samples
and the original curves are inspectable in `playbook_layer`. With an independent
whole-envelope PASS, this layer contributes to an equal mean of six layers.
Without admission, the five-Teal-layer result remains exactly unchanged. An
ambiguous memory or absent selection creates no playbook layer. The source
curve-to-grid sampling and diffusion rules are engineering adaptations.

The neural baseline's small held-out transition comparison is an **ephemeral
selection within one request**. The new overlay requires independent PASS in
addition to native measured improvement before selecting that proposed
transition model. This operation installs no model and does not implement the
playbook's persistent retention or training qualification.

For reusable episodic storage and self-healing evaluation, hosts can call
`mj_memory_recall.store.EpisodicStore` and
`mj_memory_recall.kernel.assess_update` directly. That separate native kernel
seam requires at least 30 domain-qualified evidence signals, at least three
later disjoint validation receipts, strictly improved held-out loss,
unchanged protected descriptors and independent review. It also installs no
model. Its resource-memory lifecycle remains separate from neural `r0`.

## Verification and claims

After installing all three packages, run:

```sh
python -B -m unittest discover -s tests -v
```

The suites preserve the original 28 behavioral checks and exact baseline
example result, exercise HTTP boundaries, verify all recovered source
occurrences, independently check the bounded diffusion arithmetic, test
ambiguous placement, whole-envelope binding, recall composition and denial,
and execute the maximum bounded neural history/update/validation input.

The synthetic composition fixture demonstrates one possible tie recovery.
It does not measure general recall improvement. The memory plugin's controlled
physics-playbook comparison and the placement experiment are separate
comparisons. Its 32-case report measures standalone memory recall, and does
not automatically apply to composition's baseline fallback or node placement.
To evaluate this overlay, hold observations and splits
fixed, compare forecast-only with forecast-plus-templates, and report accuracy,
false placements, abstentions and coverage together. Compare the optional
spatial playbook layer as its own ablation. To evaluate composed
retrieval, compare against the unchanged associative baseline on paired
cases. No significant or general benefit is claimed without that evidence.

Copyright (c) 2026 Michael Patrick Bangel. MIT license; see [LICENSE](LICENSE).
