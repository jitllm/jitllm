package graph

import (
	"strings"
	"testing"
)

// The MLP partition is found on the dense FFN by its edges and is a run-time
// alternative: legal, listed with its cross-card cost, never taken by the
// speed policy (Auto), refused on a Target with no split executor even when
// a caller asks (an unmet Take), and offered where a Target states a peer
// card. The gated delta sublayer, whose act_mul reads a head norm, is no
// match.
func TestMLPPartitionIsARunTimeAlternative(t *testing.T) {
	f := stated(false)
	f.Config.NEmbd, f.Config.NFFN, f.Config.NHead, f.Config.NKVHead, f.Config.HeadDim = 2048, 8192, 32, 8, 64
	g := f.Config.Build()
	os := Match(g, MLPPartition)
	if len(os) != 1 {
		t.Fatalf("%d matches", len(os))
	}
	b := MLPPartition.Bounds(g, os[0].Region)
	if want := int64(3 * 2048 * 8192 / 2); b.WeightElems != want || b.Crossings != 2 {
		t.Fatalf("bounds %+v, want %d elements off each card and 2 crossings", b, want)
	}
	lone := Target{API: "ptx", Rows: RowsDecode}
	if err := os[0].Offer(lone); err == nil || !strings.Contains(err.Error(), "split executor") {
		t.Fatalf("one card: %v", err)
	}
	if taken, met := os[0].Chosen(lone, ChoiceTake); taken || met {
		t.Fatalf("a Take with no executor: taken %v met %v, want an unmet choice", taken, met)
	}
	pair := lone
	pair.PeerCards = 1
	if err := os[0].Offer(pair); err == nil || !strings.Contains(err.Error(), "contract") {
		t.Fatalf("two cards, no executor's gate yet: %v", err)
	}
	// With a contract (the executor's gate, once built), two cards offer it,
	// and Auto still does not take it.
	withGate := *MLPPartition
	withGate.Contract = map[string]Contract{"ptx": {Band: 1e-2, Gate: "the split executor's gate"}}
	o := Match(g, &withGate)[0]
	if err := o.Offer(pair); err != nil {
		t.Fatalf("two cards: %v", err)
	}
	if taken, _ := o.Chosen(pair, ChoiceAuto); taken {
		t.Fatal("a run-time alternative was taken unasked")
	}
	if taken, met := o.Chosen(pair, ChoiceTake); !taken || !met {
		t.Fatalf("a caller's Take on two cards: taken %v met %v", taken, met)
	}
	if taken, met := o.Chosen(pair, ChoiceRefuse); taken || !met {
		t.Fatalf("Refuse: taken %v met %v", taken, met)
	}
	ds := Apply(g, MLPPartition)
	if len(ds) != 1 || ds[0].Err != nil || ds[0].When != WhenRun || ds[0].Cost == nil ||
		!strings.Contains(ds[0].String(), "open alternative") || !strings.Contains(ds[0].String(), "capacity") {
		t.Fatalf("decision: %v", ds)
	}
	df := deltaFacts(ActSiLU, 0)
	if os := Match(df.Delta.Build(), MLPPartition); len(os) != 0 {
		t.Fatalf("delta sublayer: %d matches", len(os))
	}
}

// Every rewrite a block graph is matched against states when it is decided
// and who decides it, and a Choices map naming no rewrite is refused.
func TestEveryRewriteStatesItsDecisionTime(t *testing.T) {
	for _, rw := range append(Rewrites, DeepTile) {
		if rw.When == 0 || rw.Chooser == "" {
			t.Errorf("%s: When %v, chooser %q", rw.Name, rw.When, rw.Chooser)
		}
		if rw.When == WhenRun && (rw.Objective == "" || rw.Evidence == "") {
			t.Errorf("%s: a run-time alternative without an objective or evidence", rw.Name)
		}
	}
	if err := (Choices{"normgate": ChoiceTake}).Check(); err != nil {
		t.Fatal(err)
	}
	if err := (Choices{"norm-gate": ChoiceTake}).Check(); err == nil {
		t.Fatal("a misspelt choice was accepted")
	}
}
