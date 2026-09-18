# MJ-Bangel neuroscience foundation

CA3 has extensive recurrent pyramidal-neuron connectivity. Autoassociative
models use that organization to explain associative retrieval and pattern
completion from incomplete cues. In a mouse study, disrupting CA3 NMDA-receptor
function impaired associative recall under partial cues. This supports a role
in retrieval; it does not make CA3 the sole location of fundamental memory
storage. [Nakazawa et al., Science (2002)](https://doi.org/10.1126/science.1071795).

CA2 is a distinct hippocampal subfield. Selective suppression of CA2 pyramidal
output impaired social-recognition memory in mice while sparing several other
tested behaviors. This does not establish a universal identity processor or a
sole social-identity store. [Hitti and Siegelbaum, Nature (2014)](https://www.nature.com/articles/nature13028).

Counts depend on species, strain, sex and estimation method. An updated rat
synthesis reports approximate male averages of 210,000 CA3 and 30,000 CA2 neurons.
These are biological estimates, not universal constants or artificial-network
dimensions. [Updated Neuronal Numbers of the Rat Hippocampal Formation](https://pubmed.ncbi.nlm.nih.gov/41549060/).
An older rat study reported about 330,000 versus 210,000 CA3 neurons in two strains,
illustrating why an earlier shorthand of roughly 300,000 needs qualification.
[1987 stereological study](https://pubmed.ncbi.nlm.nih.gov/3567627/).

The proposed MJ-Bangel architecture is an engineering interpretation: a CA3-inspired
recurrent associative stage reconstructs candidate patterns, then a CA2-inspired
relational-discrimination stage checks known entity, relation and context. The
implemented second stage is a categorical filter. The placement model uses prior
Bangel command transitions, not biological neuron counts. Three nine-unit layers
and their shared weights are a deliberately bounded software profile.

This implementation does **not reproduce hippocampal biology**. It contains no
spiking dynamics, anatomical connectivity fit, electrophysiology, animal social
behavior or biological validation. Reducing false matches in the attached toy
comparison is a software result with an explicit recall tradeoff. Lowercase `r0`
remains a provisional project marker with no asserted CA2 or biological meaning.
