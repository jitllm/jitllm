package sched

// machine is where this package reads the running machine from: on Linux its
// /proc and /sys trees, the process's memory policy (a bind's nodes, nil for
// none) and the CPUs it may run on; elsewhere the platform's own calls. A
// test points one at a synthetic tree (its config's machine).
type machine struct {
	proc, sys string
	bind      func() []int
	affinity  func() map[int]bool
}

// PCores, ECores, SMTSiblings, CoreSource and NUMANodes read the running
// machine (host); see machine's methods of the same names.
func PCores() []int      { return host().pCores() }
func ECores() []int      { return host().eCores() }
func SMTSiblings() []int { return host().smtSiblings() }
func CoreSource() string { return host().coreSource() }
func NUMANodes() []int   { return host().numaNodes() }

// CoreSet is m's cores by name: "" or "p" for P-cores, "pe" to add the
// E-cores, "e" for the E-cores alone, "smt" for hyperthread siblings, "all"
// for everything.
func CoreSet(name string) []int { return host().coreSet(name) }

func (m machine) coreSet(name string) []int {
	switch name {
	case "pe":
		return append(m.pCores(), m.eCores()...)
	case "e":
		return m.eCores()
	case "smt":
		return m.smtSiblings()
	case "all":
		return append(m.smtSiblings(), m.eCores()...)
	}
	return m.pCores()
}
