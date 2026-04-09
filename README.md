# Sloviter-87
A spiking neural network simulation of temporal lobe epilepsy based on the dormant basket cell hypothesis, implemented in Python using Brian2.
Models the dentate gyrus microcircuit across two conditions — healthy and epileptic — to demonstrate how mossy cell loss silences basket cells and produces granule cell hyperexcitability without any change to intrinsic GC excitability.

Model Architecture
Three leaky integrate-and-fire (LIF) neuron populations:

Granule Cells (GC) — principal excitatory neurons; hyperexcitable in epileptic condition
Mossy Cells (MC) — hilar excitatory interneurons; destroyed in epileptic condition
Basket Cells (BC) — parvalbumin+ GABAergic interneurons; go dormant without MC drive

Synaptic currents use exponential decay kinetics (τ_syn = 5 ms). Connectivity is probabilistic.
