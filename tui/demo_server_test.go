package main

import (
	"context"
	"io"
	"net/http/httptest"
	"os"
	"strings"
	"sync"
	"testing"
	"time"

	tea "github.com/charmbracelet/bubbletea"
	"github.com/jitllm/jitllm/internal/testmodels"
	"github.com/jitllm/jitllm/internal/testmodels/fresh"
	"github.com/jitllm/jitllm/server"
	"github.com/jitllm/jitllm/server/client"
	v1 "github.com/jitllm/jitllm/server/gen/jitllm/v1"
)

// tap runs the demo as a program with no terminal and keeps what it saw: the
// moves it was told of, and the last screen it drew.
type tap struct {
	d     *demo
	mu    sync.Mutex
	moves []moveInfo
	view  string
	done  bool
}

func (t *tap) Init() tea.Cmd { return t.d.Init() }

func (t *tap) Update(msg tea.Msg) (tea.Model, tea.Cmd) {
	_, cmd := t.d.Update(msg)
	t.mu.Lock()
	defer t.mu.Unlock()
	if m, ok := msg.(moveMsg); ok {
		t.moves = append(t.moves, m.info)
	}
	if _, ok := msg.(doneMsg); ok {
		t.done = true
	}
	t.view = stripANSI(t.d.View())
	return t, cmd
}

func (t *tap) View() string { return "" }

func (t *tap) snapshot() ([]moveInfo, string, bool) {
	t.mu.Lock()
	defer t.mu.Unlock()
	return append([]moveInfo(nil), t.moves...), t.view, t.done
}

// TestDemoRelocatesAStreamingReplyOnARealService: a jitllmd in this process
// holds a model on two devices (the card under CUDA and under Vulkan), the demo
// streams a reply from it and moves every block to the other device at token
// 20, brings them back with the m key after the reply, and everything it shows
// is what the service reported: the reply streamed all its tokens, each move
// moved all of the blocks with history and a path, no prompt row was computed
// again, and the blocks are where the placement says.
func TestDemoRelocatesAStreamingReplyOnARealService(t *testing.T) {
	const name = "stories15M-q8_0.jlm"
	path := testmodels.Path(name)
	fresh.Must(t, path)
	if _, err := os.Stat(path); err != nil {
		testmodels.Missing(t, "MODEL MISSING: %v (set JITLLM_MODELS to the model directory) -- this gate proved nothing", err)
	}
	devs := []string{"cuda:0", "vulkan:1"}
	if v := os.Getenv("JITLLM_MOVE_DEVICES"); v != "" {
		devs = strings.Split(v, ",")
	}
	e := server.New(server.Config{Version: "test", NoMemCache: true})
	t.Cleanup(e.Close)
	if _, err := e.LoadModel(server.LoadOptions{Path: path, ModelID: "dev", DeviceIDs: devs}); err != nil {
		t.Skipf("NO DEVICE: loading onto %v failed (%v) -- this gate proved nothing", devs, err)
	}
	srv := httptest.NewServer(e.Handler())
	t.Cleanup(srv.Close)

	ctx := context.Background()
	d, err := newDemo(ctx, client.Dial(srv.URL), demoOptions{Model: "dev", MaxSeq: 128, MaxTokens: 100,
		IgnoreEOS: true, Auto: true, Prompt: "Once upon a time"})
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(d.close)
	// The model's context is 128 positions and it decodes them in a blink: the
	// script moves once, early, and the test moves back with the key when the
	// reply is over.
	d.moveAt = []int{20}
	_, _, from, onDevice := deviceBlocks(d.blocks)
	if !onDevice {
		t.Skip("NO DEVICE ROOM: the model's blocks are on the host -- this gate proved nothing")
	}

	tp := &tap{d: d}
	p := tea.NewProgram(tp, tea.WithInput(nil), tea.WithOutput(io.Discard), tea.WithoutRenderer())
	ran := make(chan error, 1)
	go func() { _, err := p.Run(); ran <- err }()
	p.Send(tea.WindowSizeMsg{Width: 150, Height: 40})

	deadline := time.After(2 * time.Minute)
	pressed := false
	for {
		moves, _, done := tp.snapshot()
		if done && len(moves) >= len(d.moveAt) && !pressed {
			p.Send(tea.KeyMsg{Type: tea.KeyRunes, Runes: []rune("m")})
			pressed = true
		}
		if pressed && len(moves) >= len(d.moveAt)+1 {
			break
		}
		select {
		case <-deadline:
			p.Quit()
			_, view, _ := tp.snapshot()
			t.Fatalf("the scripted run did not finish: %d move(s) so far\n%s", len(moves), view)
		case <-time.After(50 * time.Millisecond):
		}
	}
	// Stopped, so that what is read below is not being written.
	p.Quit()
	if err := <-ran; err != nil {
		t.Fatal(err)
	}
	moves, view, _ := tp.snapshot()
	nBlocks := int32(len(d.blocks))
	for i, m := range moves {
		if m.Err != nil {
			t.Fatalf("move %d refused: %v", i, m.Err)
		}
		r := m.Resp
		if m.From != from || m.To == from {
			t.Errorf("move %d went %q -> %q, want it to leave %q", i, m.From, m.To, from)
		}
		if r.GetBlocksMoved() != nBlocks || len(r.GetDeclined()) != 0 {
			t.Errorf("move %d moved %d of %d blocks, declined %v", i, r.GetBlocksMoved(), nBlocks, r.GetDeclined())
		}
		if r.GetHistory().GetBytes() == 0 || r.GetPath() == v1.RelocationPath_RELOCATION_PATH_UNSPECIFIED {
			t.Errorf("move %d reports no history or no path: %+v", i, r)
		}
		if m.Recomputed != 0 {
			t.Errorf("move %d recomputed %d prompt rows", i, m.Recomputed)
		}
		if i == 0 && (m.AtToken < d.moveAt[0] || m.AtToken >= d.opt.MaxTokens) {
			t.Errorf("the scripted move was asked for at token %d, scripted for %d", m.AtToken, d.moveAt[0])
		}
		if i == 1 && m.AtToken != 0 {
			t.Errorf("the move after the reply says it was made at token %d", m.AtToken)
		}
		for _, b := range r.GetPlacement().GetBlocks() {
			if b.GetDeviceId() != m.To {
				t.Errorf("move %d: block %d is on %q, not %q", i, b.GetIndex(), b.GetDeviceId(), m.To)
			}
		}
		from = m.To
	}
	if d.tokens != d.opt.MaxTokens || d.genErr != nil {
		t.Errorf("the reply streamed %d of %d tokens, error %v", d.tokens, d.opt.MaxTokens, d.genErr)
	}
	// What is on screen is what happened.
	for _, want := range []string{"prompt rows recomputed 0", "history moved", "projection moved", "CPU host", devs[0], devs[1]} {
		if !strings.Contains(view, want) {
			t.Errorf("the screen lacks %q:\n%s", want, view)
		}
	}
	if d.moves != 2 || d.recomputed != 0 {
		t.Errorf("%d moves counted, %d prompt rows recomputed", d.moves, d.recomputed)
	}
}
