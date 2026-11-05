//go:build !race

package engine

// raceEnabled mirrors bench/guard_{race,norace}.go. TestPagedModelOpenCloseDoesNotLeak
// measures process RSS, which under -race includes the detector's shadow
// memory growth.
const raceEnabled = false
