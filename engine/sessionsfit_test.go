package engine

import (
	"context"
	"strings"
	"testing"

	"github.com/jitllm/jitllm/engine/device/gpu"
	"github.com/jitllm/jitllm/internal/testdevices"
	"github.com/jitllm/jitllm/internal/testmodels"
	"github.com/jitllm/jitllm/model"
)

// TestACardHoldsTheSessionsItsRoomStates measures the sessions-per-device
// budget on a card. An instance is placed wholly on it, and sessions leasing
// the instance are prefilled to the same context one by one. After the first,
// the card's budget is what it holds plus room for fitMore sessions' history
// and half of one more, the first session's history the measure of one. Each
// session's prompt is its own, so no page is shared between them. Every
// session up to the room holds its history on the card -- the card's history
// grows by one session's (Stats.KVBytes), no page is sent home
// (Stats.KVEvictions) and no block's weights page out -- and the one past it
// holds only part of its history there and still runs. The violation is a
// budget with no room for a second session: the second holds part of its.
func TestACardHoldsTheSessionsItsRoomStates(t *testing.T) {
	const ctxLen, fitMore = 512, 2
	m, err := model.Open(jlmOf(t, testmodels.Path("Llama-3.2-1B-Instruct-Q4_K_M.gguf")))
	if err != nil {
		t.Fatal(err)
	}
	defer m.Close()
	ids := m.Vocab.Encode(strings.Repeat("The quick brown fox jumps over the lazy dog by the river bank. ", 64), true)
	if len(ids) < ctxLen {
		t.Fatalf("the prompt is %d tokens, under the %d-token context", len(ids), ctxLen)
	}
	ids = ids[:ctxLen]
	run := func(t *testing.T, room int) (held int, perSession uint64) {
		t.Helper()
		g, err := openGPU(t, gpu.WithDevices(testdevices.One("cuda:0")), gpu.WithDeviceTune(gpu.TuneOff))
		if err != nil || g == nil {
			noDevice(t, "device", err)
		}
		defer g.Close()
		in := testInstance(t, m, ctxLen+8)
		defer in.Close()
		if err := in.SetDevice(g); err != nil {
			t.Fatal(err)
		}
		if n := in.devCount(); n != m.Cfg.NLayer {
			t.Fatalf("%d of %d blocks on the card (%s): the instance is not wholly placed", n, m.Cfg.NLayer, g.Err())
		}
		pool := NewPool()
		if err := pool.Serve(in, 0); err != nil {
			t.Fatal(err)
		}
		defer func() {
			if err := pool.Unserve(in); err != nil {
				t.Error(err)
			}
		}()
		ctx := context.Background()
		var sess []*Session
		defer func() {
			for _, s := range sess {
				s.Close()
			}
		}()
		prefill := func(i int) {
			t.Helper()
			s, err := pool.NewSession(ctxLen + 8)
			if err != nil {
				t.Fatal(err)
			}
			sess = append(sess, s)
			// A prompt of its own: sessions agreeing on a prefix share its
			// pages (KV held once a device), and room would count them once.
			p := append([]int32{ids[0]}, ids[1+7*i:]...)
			p = append(p, ids[1:1+7*i]...)
			if err := s.Run(ctx, PhasePrefill, func(in *Instance) error {
				_, err := in.Prefill(p)
				return err
			}); err != nil {
				t.Fatalf("session %d: %v (%s)", i, err, g.Err())
			}
		}
		prefill(0)
		// What the card holds, the pool's free pages given back, then room
		// for room sessions and half of one more.
		st := g.Stats()
		if _, err := g.SetBudget(st.BudgetUsed); err != nil {
			t.Fatal(err)
		}
		st = g.Stats()
		perSession = st.KVBytes
		if _, err := g.SetBudget(st.BudgetUsed + uint64(room)*perSession + perSession/2); err != nil {
			t.Fatal(err)
		}
		pageOuts := st.PageOuts
		kv := st.KVBytes
		for i := 1; ; i++ {
			prefill(i)
			after := g.Stats()
			if after.PageOuts != pageOuts {
				t.Fatalf("session %d paged %d block(s) out: the model fits, and history is what the room is for",
					i, after.PageOuts-pageOuts)
			}
			// Held is the whole of the session's history on the card and none
			// of another's sent home.
			if after.KVEvictions > 0 || after.KVBytes != kv+perSession {
				t.Logf("session %d moved the card's history by %d bytes (a whole session is %d), %d page(s) sent home: "+
					"%d session(s) of %d positions held, a room of %d and a half", i, int64(after.KVBytes)-int64(kv),
					perSession, after.KVEvictions, i, ctxLen, room)
				return i, perSession
			}
			kv = after.KVBytes
			if i > room+2 {
				t.Fatalf("%d sessions of %d positions on a card with room for %d and a half all held their history "+
					"(KV %d bytes, budget used %d)", i+1, ctxLen, room, after.KVBytes, after.BudgetUsed)
			}
		}
	}
	held, per := run(t, fitMore)
	if held != fitMore+1 {
		t.Fatalf("the card held %d session(s) of %d positions, %d bytes of history each, where its room states %d",
			held, ctxLen, per, fitMore+1)
	}
	t.Run("violation/noRoom", func(t *testing.T) {
		if held, _ := run(t, 0); held != 1 {
			t.Fatalf("THE GATE IS NOT A GATE: a card with room for half a session held %d", held)
		}
		t.Log("violation seen to fail as required: with no room the second session's history goes home")
	})
}
