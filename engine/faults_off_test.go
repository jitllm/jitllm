//go:build !jitllmfault

package engine

// faultsInjected says the build injects device faults (tier/fault.go); in
// this build nothing does, so a test instance demoted to the CPU device is a
// device failure the test did not ask for.
const faultsInjected = false

// FaultsInjected is faultsInjected for the external test package.
const FaultsInjected = faultsInjected
