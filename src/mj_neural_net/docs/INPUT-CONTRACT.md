# Input and integration contract, version 0.1.0

The runnable fixture is [partial-cue.json](../examples/partial-cue.json). The
adapter rejects missing or additional fields, duplicate JSON keys, malformed
identifiers, binary floats in feature values, nonfinite values and source injection.

| Object | Exact fields |
| --- | --- |
| Request | `profile`, `as_of`, `memories`, `history`, `updates`, `validation`, `query` |
| Query | `id`, `observed_at`, `cue`, `entity`, `relation`, `context`, `previous_command` |
| Memory | `id`, `pattern`, `entity`, `relation`, `context`, `observed_at`, `available_at`, `provenance` |
| History/update/validation record | `id`, `previous`, `following`, `observed_at`, `available_at`, `provenance` |
| Independent review | `profile`, `request_sha256`, `verdict` |

Request profile is `MJ-Bangel-Neural/0.1.0`. `as_of` equals the query's first
`observed_at`, and every supporting record satisfies
`observed_at <= available_at < as_of`. Timestamps are exact nonnegative integers
in one host-declared comparable time base. This profile does not calibrate clocks.
The earliest observation is supplied by the host; the plugin cannot recover an
earlier event that was never recorded or independently authenticate that timestamp.

Memory patterns contain exactly nine bipolar decimal strings (`0.9` or `-0.9`).
Cues contain exactly nine finite decimal strings in [-0.9,0.9] or JSON null.
Decimals have at most 12 fractional digits. Known numeric zero is valid and
counted separately from null. Entity/relation/context are bounded identifiers;
query absence is null. There are 1-4 memories, 0-32 prior transitions, 0-32 update
transitions and 0-16 validation records. No online label is accepted in a query.

All supporting records claim provenance `OBSERVED`, with unique IDs across the
query, memories and all splits. Synthetic-near-miss provenance rejects as live
corroboration. The host must authenticate its source; this string and a SHA-256
digest do not authenticate an observation. The examples and comparison are wholly
synthetic fixtures representing a virtual observation history, not real robots
or biological observations. The plugin exposes no live action capability.

Validation observations must follow all prior/update availability times. Updates
must follow prior history. Unavailable/future evidence rejects the request rather
than being silently reclassified. The model never learns from the query's outcome,
the simulation's labels, a candidate prediction or the downstream verdict.

Review profile is `MJ-Neural-Review/0.1.0`. `request_sha256` is SHA-256 over the
complete canonical request (`adapter.digest(request)`). Verdict is PASS, HOLD or
DENY. The API receives review as a separate keyword-only argument; the CLI uses
`--review`. Never let the producer create supervisor reviews. Hash binding is
not a signature, an authentication system or an independently validated CBF.

No source path, instruction budget, threshold, arbitrary command expression,
model outcome, new pathway or permit can be supplied in the request. The package
reads its own ordinary `.bangel` module and versioned path catalog. Both are bound
to result hashes; source code is trusted at the same boundary as the interpreter.

Logical limits are 50,000 instructions/events, 1,000,000 evaluations, call depth
64, 16,000,000 allocation units, 1 MiB output and 16 MiB receipt bytes. A
45-second host cancellation deadline additionally bounds runtime evaluation.
Logical units are not bytes of process RAM. CLI and HTTP input are at most 65,536 bytes.
The deadline is a trusted in-process watchdog, not an operating-system sandbox.

The existing Bangel distributions and grammar remain unchanged. MJ Neural Net
is an independently installable clone imported as `mj_neural_net`, requiring
`bangel-language==0.2.0rc1`. No package-index publication is implied. `lower` makes the computation reviewable as source
and admitted JP. The returned matching data remain advisory; no model result
grants an MJ permit, sends an action or bypasses another plugin's control gate.
