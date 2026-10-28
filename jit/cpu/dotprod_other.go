//go:build !arm64 && !amd64

package cpu

func hostDotProd() bool { return false }
