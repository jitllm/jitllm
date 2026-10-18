//go:build jitllmtest

package tier

// PersistDepLaunchesForTest is how many dependency-counter persistent kernels
// this process has launched.
func PersistDepLaunchesForTest() int64 { return persistDepTotal.Load() }

// PersistLaunchesForTest is how many persistent kernels this process has
// launched, for a gate that must show the path ran.
func PersistLaunchesForTest() int64 { return persistTotal.Load() }
