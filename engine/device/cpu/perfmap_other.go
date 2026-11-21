//go:build !linux

package cpu

// perfMap writes nothing off Linux: its samplers do not read a perf-style JIT
// map, so a name is recorded on the Code and otherwise unused.
type perfMap struct{}

func (*perfMap) open()                    {}
func (*perfMap) add(uintptr, int, string) {}
func (*perfMap) close() error             { return nil }
