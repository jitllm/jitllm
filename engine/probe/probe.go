// Package probe initialises the machine's devices and measures what the
// composer's estimator weighs: each memory level's bandwidth for one thread
// and for all, the rate of each class of instruction, and what a call or a
// launch costs. It runs measuring kernels only; the estimator never runs a
// model's code.
package probe

// rounds is how many timed runs a measurement takes the median of, after
// one run that warms it.
const rounds = 7
