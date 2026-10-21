package model

import (
	"fmt"
	"testing"

	"github.com/jitllm/jitllm/internal/testdevices"
	"github.com/jitllm/jitllm/jit_old/gpu/tier"
)

// cardClash names the first two of a tier's devices that are one physical card
// (equal Hardware keys, the UUID both CUDA and Vulkan report), or "" when the
// tier's devices are distinct. A key the backend could not supply is the Ref
// and matches nothing but itself.
func cardClash(hw []tier.Hardware) string {
	for i := range hw {
		for j := i + 1; j < len(hw); j++ {
			if hw[i].Key == hw[j].Key {
				return fmt.Sprintf("%s and %s are one physical card (%s)", hw[i].Ref, hw[j].Ref, hw[i].Key)
			}
		}
	}
	return ""
}

// distinctCards fails a gate that moves or splits blocks between two devices
// when those devices are one card reached through two backends: they share one
// memory pool, so the gate would measure a card too small for both arms rather
// than a move between cards. It fails and does not skip, so a mis-selected
// JITLLM_TEST_DEVICES is seen.
func distinctCards(t testing.TB, g *tier.GPU) {
	t.Helper()
	if c := cardClash(g.Hardware()); c != "" {
		t.Fatalf("a two-card gate was given one card: %s; name two cards in JITLLM_TEST_DEVICES (a CUDA ordinal and a Vulkan ordinal of another card)", c)
	}
}

// TestCardClashNamesBothDevices is the check's violation on stand-in hardware:
// two entries with one key are refused, naming both; distinct keys pass.
func TestCardClashNamesBothDevices(t *testing.T) {
	same := []tier.Hardware{{Key: "u1", Ref: "cuda:0"}, {Key: "u1", Ref: "vulkan:4"}}
	if c := cardClash(same); c == "" {
		t.Fatal("cuda:0 and vulkan:4 with one key were not flagged")
	} else if want := "cuda:0 and vulkan:4"; len(c) < len(want) || c[:len(want)] != want {
		t.Fatalf("the message does not name both devices: %q", c)
	}
	if c := cardClash([]tier.Hardware{{Key: "u1", Ref: "cuda:0"}, {Key: "u2", Ref: "vulkan:4"}}); c != "" {
		t.Fatalf("two cards were flagged: %s", c)
	}
}

// TestOneCardUnderTwoBackendsIsOneBudget opens a CUDA and a Vulkan device that
// are one card as a single tier and holds two things: the tier spends one pool
// for both (the card's bytes are counted once), and the two-card gates' check
// refuses the pair.
func TestOneCardUnderTwoBackendsIsOneBudget(t *testing.T) {
	c := testdevices.One("cuda:0")
	vk := []string{"vulkan:0", "vulkan:1", "vulkan:2", "vulkan:3", "vulkan:4", "vulkan:5", "vulkan:6", "vulkan:7"}
	for _, v := range testdevices.Within(vk) {
		g, err := openTier(t, tier.WithDevices(c+","+v), tier.WithDeviceTune(tier.TuneOff))
		if err != nil || g == nil {
			continue
		}
		clash := cardClash(g.Hardware())
		bs := g.Budgets()
		g.Close()
		if clash == "" {
			continue
		}
		t.Logf("%s + %s: %s", c, v, clash)
		if len(bs) != 2 || bs[0].Pool != bs[1].Pool || bs[0].PoolLimit != bs[1].PoolLimit {
			t.Fatalf("one card under two backends has budgets %+v: they must spend one pool", bs)
		}
		if bs[0].PoolLimit >= bs[0].Limit+bs[1].Limit {
			t.Fatalf("pool limit %d is the sum of the devices' limits (%d + %d): the card is counted twice",
				bs[0].PoolLimit, bs[0].Limit, bs[1].Limit)
		}
		return
	}
	noDevice(t, c+" and a Vulkan device of the same card", nil)
}
