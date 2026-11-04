package nn

// MaxHostPrefillChunk is the most rows a step on the CPU device carries.
const MaxHostPrefillChunk = 512

// MaxPrefillChunk is the narrower sub-chunk a device that cannot hold its
// wide scratch takes (gpu.Layers).
const MaxPrefillChunk = 128

// MaxDevicePrefillChunk is the widest chunk a device's batched prefill takes in
// one submission: every chunk re-reads the weights and ends in a host round
// trip, so fewer, wider chunks pay. A device that cannot hold the scratch for
// it runs narrower sub-chunks (gpu.Layers).
const MaxDevicePrefillChunk = 512
