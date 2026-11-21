//go:build !arm64 && !amd64

package cpu

func dotProd() bool { return false }
