//go:build linux

package sched

// host is the machine this process runs on.
func host() machine {
	return machine{proc: "/proc", sys: "/sys", bind: kernelBindNodes, affinity: readAffinity}
}
