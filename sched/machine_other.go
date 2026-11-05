//go:build !linux

package sched

// host is the machine this process runs on, read through the platform's own
// calls.
func host() machine { return machine{} }
