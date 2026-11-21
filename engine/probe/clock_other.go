//go:build !windows

package probe

import "time"

// clock is the probe's clock: time's monotonic reading, which resolves well
// under a microsecond here.
type clock struct{ origin time.Time }

func newClock() (clock, error) { return clock{origin: time.Now()}, nil }

// mark reads the clock.
func (c clock) mark() int64 { return int64(time.Since(c.origin)) }

// since is the seconds from mark m to now.
func (c clock) since(m int64) float64 { return float64(c.mark()-m) * 1e-9 }
