//go:build !darwin

// Package metal is Apple-only. This stub keeps the package importable
// everywhere so callers select a backend at runtime rather than with build
// tags: `if c, err := metal.Open(); err == nil` reads the same on every host.
package metal

import (
	"errors"
	"unsafe"
)

type Ctx struct{}
type Buf struct{}
type Kernel struct{}

// Batch mirrors the darwin type so jit/gpu/backend compiles everywhere. Nothing
// constructs one here: metal.Open already refuses.
type Batch struct{}

// Queue is a command queue of its own; there is none off darwin.
type Queue struct{}

var errNoMetal = errors.New("metal: not available on this platform")

func Open() (*Ctx, error)                           { return nil, errNoMetal }
func OpenWith(Opts) (*Ctx, error)                   { return nil, errNoMetal }
func (c *Ctx) Name() string                         { return "" }
func (c *Ctx) Mem() (free, total uint64, err error) { return 0, 0, errNoMetal }
func (c *Ctx) UnifiedMemory() bool                  { return false }
func (c *Ctx) MaxBuffer() uint64                    { return 0 }
func (c *Ctx) Close()                               {}

// Unretained is a no-op off darwin; see the darwin file for what it trades.
func (c *Ctx) Unretained(bool) {}

// Residency cannot exist off darwin, and says so rather than succeeding: a
// caller that asked for one must not be told it has one.
func (c *Ctx) Residency() error { return errNoMetal }

// GPUBusy and GPUBusyLast have nothing to report off darwin; they exist so
// callers (engine/model/devrate_test.go) compile everywhere.
func GPUBusy() (ns, buffers uint64)                     { return 0, 0 }
func GPUBusyLast() uint64                               { return 0 }
func GPUIdle() (ns, gaps uint64)                        { return 0, 0 }
func HostGap() (ns, n uint64)                           { return 0, 0 }
func RoundTrip() (ns, n uint64)                         { return 0, 0 }
func (c *Ctx) Alloc(int) (*Buf, error)                  { return nil, errNoMetal }
func (c *Ctx) Copy(*Buf, int, *Buf, int, int) error     { return errNoMetal }
func (c *Ctx) CopyList([]CopyOp) error                  { return errNoMetal }
func (c *Ctx) ImportAlign() int                         { return 0 }
func (c *Ctx) Import(unsafe.Pointer, int) (*Buf, error) { return nil, errNoMetal }
func (b *Buf) Imported() bool                           { return false }
func (c *Ctx) Compile(string, string) (*Kernel, error)  { return nil, errNoMetal }
func (b *Buf) Bytes() []byte                            { return nil }
func (b *Buf) Free()                                    {}
func (k *Kernel) Launch(int, int, ...*Buf) error        { return errNoMetal }
func (k *Kernel) Close()                                {}
func (k *Kernel) Limits() (int, int)                    { return 0, 0 }
func (k *Kernel) SetWrites(uint64)                      {}

// Barriers has nothing to count off darwin.
func Barriers() uint64                                   { return 0 }
func (c *Ctx) NewBatch() *Batch                          { return nil }
func (c *Ctx) NewQueue() (*Queue, error)                 { return nil, errNoMetal }
func (q *Queue) NewBatch() *Batch                        { return nil }
func (q *Queue) Wait() error                             { return errNoMetal }
func (q *Queue) Close()                                  {}
func (c *Ctx) HasQueues() bool                           { return false }
func (c *Ctx) WaitOwn() error                            { return errNoMetal }
func (b *Batch) Encode(*Kernel, int, int, ...*Buf) error { return errNoMetal }
func (b *Batch) Commit() error                           { return errNoMetal }

// Wait is the no-op twin of the darwin synchroniser: there is no device here,
// so there is never anything committed to wait for.
func (c *Ctx) Wait() error { return nil }

func SubmitStats() (commits, waits, blocked uint64) { return 0, 0, 0 }

func SubmitTimes() (encodeNs, waitNs, launches uint64) { return 0, 0, 0 }

func SoloStats() (n, ns uint64) { return 0, 0 }

func AddSessionNs(ns uint64)       {}
func SessionStats() (n, ns uint64) { return 0, 0 }

func PhaseStats() (newNs, commitNs uint64) { return 0, 0 }

// CounterSampling and CounterSets are darwin-only questions; the stubs let
// callers compile everywhere.
func (c *Ctx) CounterSampling() (stage, dispatch bool) { return false, false }
func (c *Ctx) CounterSets() []string                   { return nil }

// GPUCores is 0 off darwin.
func GPUCores() int { return 0 }

func SetKernelTiming(bool) {}

func KernelTimes() (map[string]float64, map[string]int) { return nil, nil }
