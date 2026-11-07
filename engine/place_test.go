package engine

import (
	"github.com/jitllm/jitllm/internal/testmodels"
	"os"
	"testing"

	"github.com/jitllm/jitllm/model"
)

// TestBlockReadBytesIsNotBlockSize is the accounting the head placement rests
// on, and the whole point is that the two numbers differ.
//
// A placement removes a read and costs a size. On a mixture they differ by an
// order of magnitude (a token reads NExpertUsed of NExpert); on a dense model
// they are equal, which is the control.
func TestBlockReadBytesIsNotBlockSize(t *testing.T) {
	for name, path := range models() {
		t.Run(name, func(t *testing.T) {
			if _, err := os.Stat(path); err != nil {
				testmodels.Missing(t, "model not present: %s", path)
			}
			m, err := model.Open(jlmOf(t, path))
			if err != nil {
				t.Fatal(err)
			}
			defer m.Close()
			if len(m.Layers) == 0 {
				t.Skip("no blocks")
			}
			l := &m.Layers[0]
			size := model.TensorBytes(l.Wq) + model.TensorBytes(l.Wk) + model.TensorBytes(l.Wv) +
				model.TensorBytes(l.Wo) + model.TensorBytes(l.Router) +
				model.TensorBytes(l.Gate) + model.TensorBytes(l.Up) + model.TensorBytes(l.Down)
			read := m.BlockReadBytes(0)
			if read == 0 || size == 0 {
				t.Fatalf("read %d size %d: neither may be zero", read, size)
			}
			if read > size {
				t.Fatalf("read %d exceeds size %d: a token cannot read more than the block holds", read, size)
			}
			c := m.Cfg
			if c.MoE() {
				// The expert banks are the whole difference, so the ratio is
				// pinned by the architecture rather than approximated.
				exp := model.TensorBytes(l.Gate) + model.TensorBytes(l.Up) + model.TensorBytes(l.Down)
				want := size - exp + exp/uint64(c.NExpert)*uint64(c.NExpertUsed)
				if read != want {
					t.Errorf("MoE read %d, want %d (%d of %d experts)", read, want, c.NExpertUsed, c.NExpert)
				}
				if read >= size {
					t.Errorf("MoE read %d is not less than size %d; the banks are not being discounted", read, size)
				}
			} else if read != size {
				t.Errorf("dense read %d != size %d: every byte of a dense block is read every token", read, size)
			}
			t.Logf("%s: block size %.1f MiB, read per token %.1f MiB, head read %.1f MiB",
				name, float64(size)/(1<<20), float64(read)/(1<<20),
				float64(m.HeadReadBytes())/(1<<20))
		})
	}
}

// TestHeadTradeNeverGivesUpMoreThanItGains is the invariant placeHead's release
// loop exists to hold.
//
// The assertion is bytes, not blocks: how many blocks the projection buys
// depends on the ratio of two sizes. The trade is sound when the reads given up
// never exceed the read gained less one crossing.
func TestHeadTradeNeverGivesUpMoreThanItGains(t *testing.T) {
	for name, path := range models() {
		t.Run(name, func(t *testing.T) {
			if _, err := os.Stat(path); err != nil {
				testmodels.Missing(t, "model not present: %s", path)
			}
			m, err := model.Open(jlmOf(t, path))
			if err != nil {
				t.Fatal(err)
			}
			defer m.Close()
			head := m.HeadReadBytes()
			if head == 0 {
				t.Skip("tied embedding: there is no separate projection to place")
			}
			// Replay placeHead's rule and total what it would release.
			budget, released, blocks := head, uint64(0), 0
			if budget < seamCrossBytes {
				budget = 0
			} else {
				budget -= seamCrossBytes
			}
			for li := len(m.Layers) - 1; li > 0; li-- {
				c := m.BlockReadBytes(li)
				if c > budget {
					break
				}
				budget -= c
				released += c
				blocks++
			}
			t.Logf("%s: projection %.2f MiB buys %d of %d blocks (%.2f MiB of reads given up)",
				name, float64(head)/(1<<20), blocks, len(m.Layers), float64(released)/(1<<20))

			// The invariant. Against the violation (the `c > budget` check
			// dropped) it fails on every model whose projection is smaller than
			// the blocks it would take. Releasing nothing is always sound.
			if released > 0 && released+seamCrossBytes > head {
				t.Errorf("gave up %d bytes of per-token reads plus a %d-byte crossing "+
					"to gain %d: the trade loses", released, uint64(seamCrossBytes), head)
			}
			// And the projection must never cost the device its whole prefix:
			// Layers(0, gpuLayers) needs at least one block to submit.
			if blocks >= len(m.Layers) {
				t.Errorf("released all %d blocks", blocks)
			}
		})
	}
}

// TestMoEDiscountBuysMoreBlocks is the reason the read accounting exists at
// all: on a mixture of experts the same projection is worth strictly more
// blocks than it would be if a token read every expert.
func TestMoEDiscountBuysMoreBlocks(t *testing.T) {
	path := models()["qwen3moe"]
	if _, err := os.Stat(path); err != nil {
		testmodels.Missing(t, "model not present: %s", path)
	}
	m, err := model.Open(jlmOf(t, path))
	if err != nil {
		t.Fatal(err)
	}
	defer m.Close()
	if !m.Cfg.MoE() {
		t.Skip("not a mixture of experts")
	}
	l := &m.Layers[0]
	full := model.TensorBytes(l.Wq) + model.TensorBytes(l.Wk) + model.TensorBytes(l.Wv) + model.TensorBytes(l.Wo) +
		model.TensorBytes(l.Router) + model.TensorBytes(l.Gate) + model.TensorBytes(l.Up) + model.TensorBytes(l.Down)
	head := m.HeadReadBytes()
	// headReadBytes is zero on a tied embedding, and `head - seamCrossBytes` on
	// uint64 would underflow to an astronomical budget, so refuse explicitly.
	if head <= seamCrossBytes {
		t.Skipf("the projection reads %d bytes, at or under the %d-byte seam crossing "+
			"(a tied embedding has no output tensor): there is no budget to price, "+
			"so this comparison would prove nothing", head, seamCrossBytes)
	}
	budget := head - seamCrossBytes
	count := func(per uint64) int {
		if per == 0 {
			t.Fatalf("a block priced at zero bytes buys an unbounded number of blocks")
		}
		return int(budget / per)
	}
	got, naive := count(m.BlockReadBytes(0)), count(full)
	t.Logf("a %d-byte budget buys %d blocks priced by READ, %d priced by SIZE (%d layers)",
		budget, got, naive, len(m.Layers))
	if got <= naive {
		t.Errorf("read-priced %d blocks, size-priced %d: discounting the banks must buy more", got, naive)
	}
}
