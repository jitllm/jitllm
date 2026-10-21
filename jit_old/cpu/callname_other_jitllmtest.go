//go:build jitllmtest && !linux

package cpu

func codeName(*Code) string { return "" }
