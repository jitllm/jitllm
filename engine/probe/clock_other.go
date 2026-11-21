//go:build !windows

package probe

import "time"

// origin is where the probe's clock counts from.
var origin = time.Now()

// mark reads the probe's clock: time's monotonic reading, which resolves
// well under a microsecond here.
func mark() int64 { return int64(time.Since(origin)) }

// since is the seconds from mark m to now.
func since(m int64) float64 { return float64(mark()-m) * 1e-9 }
