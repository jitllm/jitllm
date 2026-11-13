//go:build linux

package engine

import (
	"slices"
	"testing"
)

// TestPhaseMeterAccountsForEveryByte checks the phase rows account for the
// pager's total exactly: every byte in one row. The meter takes deltas of the
// pager's own counter, so the check is equality. A budget forcing eviction
// makes the decode row nonzero, which is also asserted.
func TestPhaseMeterAccountsForEveryByte(t *testing.T) {
	m, openOpts := hybridModelOpt(t, hyOpt{moe: true, layers: 8, manyExp: true})
	defer m.Close()
	if m.Container == nil {
		t.Skip("no container: there is no pager to split")
	}
	// One frame for eight blocks, so decode re-faults and the row is not zero.
	st := testInstance(t, m, 64, append(slices.Clip(openOpts), WithPageBudget(m.Container.PageBytes(0)))...)
	defer st.Close()

	before := m.PagedBytes()
	ph := NewPhaseMeter(m)
	ph.Mark("load")

	ids := []int32{1, 2, 3, 4}
	_, err := st.Prefill(ids)
	if err != nil {
		t.Fatal(err)
	}
	ph.Mark("prefill")

	const nTok = 4
	for i := 0; i < nTok; i++ {
		if _, err = st.Forward(int32(5 + i)); err != nil {
			t.Fatal(err)
		}
	}
	dec := ph.Mark("decode")

	var sum uint64
	var reads int64
	for _, r := range ph.Rows() {
		t.Logf("%-8s %12d B  %6d read(s)  %4d fault(s)  %v",
			r.Name, r.Bytes, r.Reads, r.Faults, r.Dur)
		sum += r.Bytes
		reads += r.Reads
	}

	// The meter starts AFTER `before`, so the rows account for everything the
	// pager did from that point on and nothing earlier.
	if got := m.PagedBytes() - before; sum != got {
		t.Errorf("the rows sum to %d B and the pager read %d B since the meter "+
			"started: %+d unaccounted -- a phase is being lost or double-counted",
			sum, got, int64(sum)-int64(got))
	}
	if dec.Bytes == 0 {
		t.Fatal("the decode row is zero: this fixture did not re-fault, so the " +
			"one row this meter exists for proved nothing")
	}
	if per := dec.PerToken(nTok); per == 0 || per != dec.Bytes/nTok {
		t.Errorf("PerToken(%d) = %d against %d B over the phase", nTok, per, dec.Bytes)
	}
	t.Logf("decode reads %s B/token over %d token(s)",
		commasU(dec.PerToken(nTok)), nTok)

	// PerToken(0) must be 0 rather than a division by zero (EOS on the first
	// sample is ordinary).
	if got := dec.PerToken(0); got != 0 {
		t.Errorf("PerToken(0) = %d, want 0", got)
	}
}

func commasU(v uint64) string {
	s, out := "", ""
	for v > 0 {
		s = string(rune('0'+v%10)) + s
		v /= 10
	}
	if s == "" {
		return "0"
	}
	for i, c := range s {
		if i > 0 && (len(s)-i)%3 == 0 {
			out += ","
		}
		out += string(c)
	}
	return out
}
