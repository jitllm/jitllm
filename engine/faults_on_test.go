//go:build jitllmfault

package engine

// faultsInjected says the build injects device faults (tier/fault.go), so a
// test instance may be demoted to the CPU device on purpose.
const faultsInjected = true

// FaultsInjected is faultsInjected for the external test package.
const FaultsInjected = faultsInjected
