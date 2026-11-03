//go:build !windows

package main

// attachParentConsole is Windows' case alone: elsewhere a standard error that
// does not stat has nowhere else to go.
func attachParentConsole(*stdio) bool { return false }
