//go:build darwin

package sched

import (
	"runtime"
	"sync"
	"testing"
	"time"
)

// TestLowerToEKeepsTheClassForTheCallOnly: LowerToE puts the calling thread at
// the E-core class, Raise puts back what was there, and the goroutine is
// unlocked again, so no lowered thread is left in the runtime's pool.
func TestLowerToEKeepsTheClassForTheCallOnly(t *testing.T) {
	if !CanSplitE() {
		t.Skip("no E-cores a thread can be kept on here")
	}
	done := make(chan struct{})
	go func() {
		defer close(done)
		runtime.LockOSThread()
		defer runtime.UnlockOSThread()
		was := QoSSelf()
		l := LowerToE()
		if got := QoSSelf(); got != EClass() {
			t.Errorf("lowered thread is at class %#x, want %#x", got, EClass())
		}
		l.Raise()
		if got := QoSSelf(); got != was {
			t.Errorf("after Raise the thread is at class %#x, want %#x it started at", got, was)
		}
	}()
	<-done
}

// TestAnEPoolsWorkersAskForTheECoreClass: a pool wholly on the E-cores runs its
// workers at the E-core class and a pool on the P-cores does not.
func TestAnEPoolsWorkersAskForTheECoreClass(t *testing.T) {
	if !CanSplitE() {
		t.Skip("no E-cores a thread can be kept on here")
	}
	classes := func(cpus []int) map[uint32]int {
		p := New(cpus)
		defer p.Close()
		var mu sync.Mutex
		got := map[uint32]int{}
		for range 20 {
			p.Do(len(cpus)*8, 1, func(_, _, _ int) {
				for t0 := time.Now(); time.Since(t0) < 200*time.Microsecond; {
				}
				c := QoSSelf()
				mu.Lock()
				got[c]++
				mu.Unlock()
			})
		}
		return got
	}
	if got := classes(CoreSet("e")); got[EClass()] == 0 {
		t.Fatalf("an E-core pool's regions ran at classes %v, none at %#x", got, EClass())
	}
	if got := classes(CoreSet("p")); got[EClass()] != 0 {
		t.Fatalf("a P-core pool's regions ran at the E-core class: %v", got)
	}
}
