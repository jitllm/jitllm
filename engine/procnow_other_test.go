//go:build !linux

package engine

import "testing"

// procCounts is empty where there is no /proc: the leak gates there read the
// live heap, the goroutines, the device ledger and the driver objects owned.
type procCounts struct {
	ok       bool
	rss, fds int
	th       struct{ goMs, foreign int }
}

func procNow(*testing.T) procCounts { return procCounts{} }
