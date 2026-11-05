//go:build linux && !amd64 && !arm64

package sched

// sysGetcpu is unknown here, so currentCPU reports -1.
const sysGetcpu = 0
