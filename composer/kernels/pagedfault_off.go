//go:build !jitllmfault

package kernels

// pagedFault names a deliberate error in the paged kernels' generation, for a
// gate to show it fails. Only a jitllmfault build can set it; here it is a
// constant and every branch on it compiles away.
const pagedFault = ""
