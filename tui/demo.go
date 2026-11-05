package main

import (
	"context"
	"errors"
	"fmt"
	"slices"
	"strings"
	"time"

	"connectrpc.com/connect"
	tea "github.com/charmbracelet/bubbletea"
	"github.com/charmbracelet/lipgloss"
	"github.com/jitllm/jitllm/common/session"
	"github.com/jitllm/jitllm/nn"
	"github.com/jitllm/jitllm/server/client"
	v1 "github.com/jitllm/jitllm/server/gen/jitllm/v1"
)

// The relocation demo: `tui demo relocate`. One screen whose only job is to
// show a model's blocks, and the history they hold, moving between the host
// and the devices of a running jitllmd while a reply streams. It reads
// everything it draws from the service (the placement, the session, the move's
// response) and moves blocks through the same RelocateBlocks call
// `jitllmd place -move` makes.

// demoOptions are the flags of `tui demo relocate`.
type demoOptions struct {
	Addr      string
	Model     string   // loaded model id, name or file; loaded when absent
	Devices   []string // for a model that has to be loaded
	MaxSeq    int
	MaxTokens int
	Prompt    string
	IgnoreEOS bool
	Auto      bool // start the scripted run at once
}

const defaultDemoPrompt = "Write a long, detailed story about a lighthouse keeper who finds a message in a bottle. Take your time and keep going."

// The scripted run moves the blocks when the reply reaches these tokens.
func autoMoveTokens() []int { return []int{80, 160} }

const (
	rateWindow = 250 * time.Millisecond // one sparkline sample
	rateKeep   = 120
	pollEvery  = 400 * time.Millisecond
	frameEvery = time.Second / 12
)

// moveReq is one RelocateBlocks call the keys ask for.
type moveReq struct {
	First, Last int
	To          string // a device id; hostID when Home
	Home        bool
}

// device is the device the request names: the CPU when Home.
func (r moveReq) device() string {
	if r.Home {
		return nn.CPU
	}
	return r.To
}

// moveInfo is a finished move as the banner tells it.
type moveInfo struct {
	From, To    string // column ids
	First, Last int
	AtToken     int
	Resp        *v1.RelocateBlocksResponse
	Err         error
	// Recomputed is how many prompt rows the service prefilled during the
	// call: the growth of the session's tokens_prefilled.
	Recomputed int64
	Took       time.Duration
}

type demo struct {
	c         *client.Conn
	opt       demoOptions
	sid       string
	modelName string
	chat      bool // the model has a chat template; a bare prompt otherwise
	devices   []string
	maxSeq    int

	w, h  int
	frame int

	cols   []string
	blocks []mapBlock
	slides map[int]slide
	pos    int // live position of the history

	// The reply.
	reply        strings.Builder
	tokens       int
	promptTokens int
	generating   bool
	finished     *v1.GenerateFinished
	cancel       context.CancelFunc
	events       chan tea.Msg
	genErr       error

	// The rate: tokens per sample window, with the samples a move landed at.
	rates      []float64
	marks      []int
	lastSample time.Time
	lastCount  int

	auto bool
	// moveAt are the tokens an `a` run moves the blocks at, and which of them
	// have been.
	moveAt   []int
	autoDone []bool

	// upBlocks is how many blocks were on a device when the demo started.
	upBlocks int

	pending    *moveReq
	last       *moveInfo
	moves      int
	recomputed int64
	note       string
}

// newDemo finds the model on the service (loading it when it is not there),
// opens the session the demo drives and reads its placement.
func newDemo(ctx context.Context, c *client.Conn, opt demoOptions) (*demo, error) {
	ml, err := c.Model.ListModels(ctx, connect.NewRequest(&v1.ListModelsRequest{LoadedOnly: true}))
	if err != nil {
		return nil, fmt.Errorf("asking the service for its models: %w", err)
	}
	var info *v1.ModelInfo
	for _, m := range ml.Msg.GetLoaded() {
		if opt.Model == "" || slices.Contains([]string{m.GetModelId(), m.GetName(), m.GetPath()}, opt.Model) ||
			strings.HasSuffix(m.GetPath(), "/"+opt.Model) {
			info = m
			break
		}
	}
	if info == nil {
		if opt.Model == "" {
			return nil, errors.New("the service has no model loaded: name one with -model")
		}
		if len(opt.Devices) == 0 {
			return nil, fmt.Errorf("%s is not loaded: pass -devices (for example cuda:0,vulkan:1) to load it on the cards, or load it with jitllmd serve -load", opt.Model)
		}
		lr, err := c.Model.LoadModel(ctx, connect.NewRequest(&v1.LoadModelRequest{Path: opt.Model, DeviceIds: opt.Devices}))
		if err != nil {
			return nil, fmt.Errorf("loading %s: %w", opt.Model, err)
		}
		info = lr.Msg.GetModel()
	}
	maxSeq := opt.MaxSeq
	if maxSeq <= 0 {
		maxSeq = 512
	}
	cs, err := c.Session.CreateSession(ctx, connect.NewRequest(&v1.CreateSessionRequest{ModelId: info.GetModelId(), MaxSeq: int32(maxSeq)}))
	if err != nil {
		return nil, fmt.Errorf("opening a session on %s: %w", info.GetName(), err)
	}
	d := &demo{c: c, opt: opt, sid: cs.Msg.GetSession().GetSessionId(), modelName: info.GetName(), chat: info.GetChatCapable(),
		devices: info.GetDefaultDeviceIds(), maxSeq: int(cs.Msg.GetSession().GetMaxSeq()),
		slides: map[int]slide{}, events: make(chan tea.Msg, 64), auto: opt.Auto, moveAt: autoMoveTokens()}
	if opt.Prompt == "" {
		d.opt.Prompt = defaultDemoPrompt
	}
	if d.opt.MaxTokens <= 0 {
		d.opt.MaxTokens = 260
	}
	pl, err := c.Placement.GetPlacement(ctx, connect.NewRequest(&v1.GetPlacementRequest{SessionId: d.sid}))
	if err != nil {
		d.close()
		return nil, fmt.Errorf("reading the placement: %w", err)
	}
	d.blocks = blocksOf(pl.Msg.GetPlacement())
	d.cols = columnsOf(d.devices, d.blocks)
	if _, last, _, ok := deviceBlocks(d.blocks); ok {
		d.upBlocks = last + 1
	}
	if len(d.cols) < 2 {
		d.close()
		return nil, errors.New("this model has no device to move blocks to: load it with -devices")
	}
	return d, nil
}

// close stops the reply and closes the session the demo opened.
func (d *demo) close() {
	if d.cancel != nil {
		d.cancel()
	}
	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()
	d.c.Session.CloseSession(ctx, connect.NewRequest(&v1.CloseSessionRequest{SessionId: d.sid}))
}

// ----------------------------------------------------------------- messages

type (
	demoFrameMsg struct{}
	pollMsg      struct{}
	placementMsg struct {
		p   *v1.Placement
		pos int // the session's position, -1 when it was not read
	}
	startedMsg struct{ s *v1.GenerateStarted }
	tokenMsg   struct{ text string }
	doneMsg    struct {
		fin *v1.GenerateFinished
		err error
	}
	moveMsg struct{ info moveInfo }
	noteMsg struct{ text string }
)

func (d *demo) Init() tea.Cmd {
	cmds := []tea.Cmd{d.poll(0), frameCmd()}
	if d.auto {
		cmds = append(cmds, d.generate())
	}
	return tea.Batch(cmds...)
}

func frameCmd() tea.Cmd {
	return tea.Tick(frameEvery, func(time.Time) tea.Msg { return demoFrameMsg{} })
}

// poll reads the placement and the position after a wait: what another client
// moved shows here too.
func (d *demo) poll(wait time.Duration) tea.Cmd {
	return tea.Tick(wait, func(time.Time) tea.Msg {
		ctx, cancel := context.WithTimeout(context.Background(), 3*time.Second)
		defer cancel()
		pl, err := d.c.Placement.GetPlacement(ctx, connect.NewRequest(&v1.GetPlacementRequest{SessionId: d.sid}))
		if err != nil {
			return noteMsg{"the service did not answer: " + err.Error()}
		}
		pos := -1
		if s, err := d.c.Session.GetSession(ctx, connect.NewRequest(&v1.GetSessionRequest{SessionId: d.sid})); err == nil {
			pos = int(s.Msg.GetSession().GetPosition())
		}
		return placementMsg{pl.Msg.GetPlacement(), pos}
	})
}

// waitEvent is the next message the reply's goroutine sent.
func (d *demo) waitEvent() tea.Cmd {
	ch := d.events
	return func() tea.Msg { return <-ch }
}

// generate starts a reply from a fresh history, and streams it into events.
func (d *demo) generate() tea.Cmd {
	if d.generating {
		return nil
	}
	d.generating = true
	d.reply.Reset()
	d.tokens, d.promptTokens, d.finished, d.genErr = 0, 0, nil, nil
	d.rates, d.marks, d.lastCount, d.lastSample = nil, nil, 0, time.Time{}
	d.autoDone = make([]bool, len(d.moveAt))
	ctx, cancel := context.WithCancel(context.Background())
	d.cancel = cancel
	prompt := &v1.PromptInput{Input: &v1.PromptInput_Text{Text: d.opt.Prompt}}
	if d.chat {
		prompt = &v1.PromptInput{Input: &v1.PromptInput_Chat{Chat: &v1.ChatPrompt{
			Messages:            []*v1.ChatMessage{{Role: "user", Content: d.opt.Prompt}},
			AddGenerationPrompt: true,
		}}}
	}
	req := &v1.GenerateRequest{
		SessionId: d.sid,
		Prompt:    prompt,
		MaxTokens: int32(d.opt.MaxTokens),
		IgnoreEos: d.opt.IgnoreEOS,
	}
	ch, c, sid := d.events, d.c, d.sid
	go func() {
		// A new reply starts from an empty history, so the bars grow from
		// nothing as they did the first time.
		if _, err := c.Session.ResetSession(ctx, connect.NewRequest(&v1.ResetSessionRequest{SessionId: sid})); err != nil {
			ch <- doneMsg{err: err}
			return
		}
		stream, err := c.Inference.Generate(ctx, connect.NewRequest(req))
		if err != nil {
			ch <- doneMsg{err: err}
			return
		}
		defer stream.Close()
		var fin *v1.GenerateFinished
		for stream.Receive() {
			switch ev := stream.Msg().GetEvent().(type) {
			case *v1.GenerateResponse_Started:
				ch <- startedMsg{ev.Started}
			case *v1.GenerateResponse_Token:
				ch <- tokenMsg{ev.Token.GetText()}
			case *v1.GenerateResponse_Finished:
				fin = ev.Finished
			}
		}
		ch <- doneMsg{fin: fin, err: stream.Err()}
	}()
	return d.waitEvent()
}

// ----------------------------------------------------------------- moves

// deviceBlocks is the first and last block on a device, and the first's.
func deviceBlocks(blocks []mapBlock) (first, last int, dev string, ok bool) {
	for _, b := range blocks {
		if b.Dev == hostID {
			continue
		}
		if !ok {
			first, dev, ok = b.Index, b.Dev, true
		}
		last = b.Index
	}
	return
}

// nextOf is the device after cur among the columns' devices, wrapping.
func nextOf(cols []string, cur string) string {
	devs := cols[1:]
	return devs[(slices.Index(devs, cur)+1)%len(devs)]
}

// planAll is `m`: every device block to the next device. With one device it
// is the only move there is: the blocks go up, or come home when they are up.
// From home, up is the first `up` blocks: the ones that were on a device when
// the demo started (all of them when none were).
func planAll(cols []string, blocks []mapBlock, up int) (moveReq, error) {
	first, last, cur, ok := deviceBlocks(blocks)
	switch {
	case !ok:
		if up <= 0 || up > len(blocks) {
			up = len(blocks)
		}
		return moveReq{0, up - 1, cols[1], false}, nil
	case len(cols) == 2:
		return moveReq{0, last, hostID, true}, nil
	}
	return moveReq{first, last, nextOf(cols, cur), false}, nil
}

// planHome is `h`: the device set comes home.
func planHome(blocks []mapBlock) (moveReq, error) {
	_, last, _, ok := deviceBlocks(blocks)
	if !ok {
		return moveReq{}, errors.New("every block is already on the host")
	}
	return moveReq{0, last, hostID, true}, nil
}

// planGroup is a digit key: the blocks are cut into nine groups, and group n
// goes to the next device after the one its first block is on.
func planGroup(n int, cols []string, blocks []mapBlock) (moveReq, error) {
	size := (len(blocks) + 8) / 9
	first, last := (n-1)*size, min(n*size, len(blocks))-1
	if first > last {
		return moveReq{}, fmt.Errorf("this model has no group %d: %d blocks in groups of %d", n, len(blocks), size)
	}
	cur := blocks[first].Dev
	to := cols[1]
	if cur != hostID {
		to = nextOf(cols, cur)
	}
	return moveReq{first, last, to, false}, nil
}

// relocate asks the service for the move and reads what it reports.
func (d *demo) relocate(r moveReq) tea.Cmd {
	d.pending = &r
	from := d.blocks[r.First].Dev
	to := r.To
	atToken := 0
	if d.generating {
		atToken = d.tokens
	}
	c, sid := d.c, d.sid
	return func() tea.Msg {
		ctx, cancel := context.WithTimeout(context.Background(), 2*time.Minute)
		defer cancel()
		prefilled := func() int64 {
			s, err := c.Session.GetSession(ctx, connect.NewRequest(&v1.GetSessionRequest{SessionId: sid}))
			if err != nil {
				return 0
			}
			return s.Msg.GetSession().GetTokensPrefilled()
		}
		before := prefilled()
		t0 := time.Now()
		resp, err := c.Placement.RelocateBlocks(ctx, connect.NewRequest(&v1.RelocateBlocksRequest{
			SessionId: sid, FirstBlock: int32(r.First), LastBlock: int32(r.Last), ToDeviceId: r.device()}))
		took := time.Since(t0)
		info := moveInfo{From: from, To: to, First: r.First, Last: r.Last, AtToken: atToken, Took: took,
			Recomputed: prefilled() - before, Err: err}
		if err == nil {
			info.Resp = resp.Msg
		}
		return moveMsg{info}
	}
}

// ----------------------------------------------------------------- update

func (d *demo) Update(msg tea.Msg) (tea.Model, tea.Cmd) {
	switch msg := msg.(type) {
	case tea.WindowSizeMsg:
		d.w, d.h = msg.Width, msg.Height
	case tea.KeyMsg:
		return d, d.key(msg)
	case demoFrameMsg:
		d.frame++
		d.sample()
		return d, frameCmd()
	case pollMsg:
	case placementMsg:
		d.applyPlacement(msg.p)
		if msg.pos >= 0 && !d.generating {
			d.pos = msg.pos
		}
		return d, d.poll(pollEvery)
	case noteMsg:
		d.note = msg.text
		return d, d.poll(pollEvery)
	case startedMsg:
		d.promptTokens = int(msg.s.GetPromptTokens())
		d.pos = d.promptTokens
		d.lastSample = time.Now()
		return d, d.waitEvent()
	case tokenMsg:
		d.reply.WriteString(msg.text)
		d.tokens++
		d.pos = d.promptTokens + d.tokens
		return d, tea.Batch(d.waitEvent(), d.script())
	case doneMsg:
		d.generating, d.finished, d.genErr = false, msg.fin, msg.err
		if d.cancel != nil {
			d.cancel()
			d.cancel = nil
		}
		if msg.fin != nil {
			d.pos = int(msg.fin.GetPosition())
		}
	case moveMsg:
		d.pending = nil
		info := msg.info
		d.last = &info
		if info.Err != nil {
			return d, nil
		}
		d.moves++
		d.recomputed += info.Recomputed
		d.applyPlacement(info.Resp.GetPlacement())
		// A scripted move that fell due while this one ran goes now.
		return d, d.script()
	}
	return d, nil
}

// script moves the blocks at the scripted tokens of an `a` run.
func (d *demo) script() tea.Cmd {
	if !d.auto || !d.generating || d.pending != nil {
		return nil
	}
	for i, at := range d.moveAt {
		if d.tokens >= at && !d.autoDone[i] {
			d.autoDone[i] = true
			r, err := planAll(d.cols, d.blocks, d.upBlocks)
			if err != nil {
				d.note = err.Error()
				return nil
			}
			return d.relocate(r)
		}
	}
	return nil
}

func (d *demo) key(k tea.KeyMsg) tea.Cmd {
	s := k.String()
	switch s {
	case "q", "ctrl+c", "esc":
		d.close()
		return tea.Quit
	case "g":
		d.auto = false
		return d.generate()
	case "a":
		if d.generating {
			d.note = "a reply is running: s stops it first"
			return nil
		}
		d.auto = true
		return d.generate()
	case "s":
		if d.cancel != nil {
			d.cancel()
		}
		return nil
	}
	var r moveReq
	var err error
	switch {
	case s == "m":
		r, err = planAll(d.cols, d.blocks, d.upBlocks)
	case s == "h":
		r, err = planHome(d.blocks)
	case len(s) == 1 && s[0] >= '1' && s[0] <= '9':
		r, err = planGroup(int(s[0]-'0'), d.cols, d.blocks)
	default:
		return nil
	}
	if err != nil {
		d.note = err.Error()
		return nil
	}
	if d.pending != nil {
		d.note = "a move is already in flight"
		return nil
	}
	d.note = ""
	return d.relocate(r)
}

// applyPlacement takes the service's placement as the map, and starts a slide
// for every block that is in another column than it was.
func (d *demo) applyPlacement(p *v1.Placement) {
	nb := blocksOf(p)
	if len(nb) == 0 {
		return
	}
	moved := 0
	for _, b := range nb {
		if b.Index < len(d.blocks) && d.blocks[b.Index].Dev != b.Dev {
			d.slides[b.Index] = slide{From: d.blocks[b.Index].Dev, Start: d.frame, Rank: moved}
			moved++
		}
	}
	if moved > 0 && d.generating {
		d.marks = append(d.marks, len(d.rates))
	}
	d.blocks = nb
	d.cols = columnsOf(d.devices, nb)
}

// sample appends a decode rate to the sparkline once a window has passed.
func (d *demo) sample() {
	if !d.generating || d.lastSample.IsZero() {
		return
	}
	dt := time.Since(d.lastSample)
	if dt < rateWindow {
		return
	}
	d.rates = append(d.rates, float64(d.tokens-d.lastCount)/dt.Seconds())
	if len(d.rates) > rateKeep {
		d.rates = d.rates[1:]
		for i := range d.marks {
			d.marks[i]--
		}
	}
	d.lastCount, d.lastSample = d.tokens, time.Now()
}

// ----------------------------------------------------------------- view

func (d *demo) mapView() mapView {
	return mapView{Cols: d.cols, Blocks: d.blocks, Pos: d.pos, MaxSeq: d.maxSeq, Frame: d.frame, Slides: d.slides}
}

const (
	bannerRows = 6 // the move banner, border included
	sparkRows  = 3
)

func (d *demo) View() string {
	if d.w < 60 || d.h < 24 {
		return sDim().Render("the relocation demo wants a terminal of at least 60 x 24")
	}
	bodyH := d.h - 1 - bannerRows - 2
	leftW := max(34, d.w*2/5)
	sparkH := sparkRows + 4
	left := lipgloss.JoinVertical(lipgloss.Left,
		d.replyPanel(leftW, bodyH-sparkH),
		d.ratePanel(leftW, sparkH))
	v := lipgloss.JoinVertical(lipgloss.Left,
		d.header(),
		lipgloss.JoinHorizontal(lipgloss.Top, left, d.mapView().render(d.w-leftW, bodyH)),
		banner(d.last, d.pending, d.blocks, d.w, bannerRows),
		d.footer())
	return fitHeight(fitWidth(v, d.w), d.h)
}

func (d *demo) header() string {
	l := sBrand().Render("jitllm") + sDim().Render(" · live relocation · ") + sFg().Render(strings.TrimSuffix(d.modelName, ".jlm")) +
		sDim().Render(" · "+d.c.Base)
	state := sDim().Render("idle")
	if d.generating {
		state = lipgloss.NewStyle().Foreground(cBrand).Render("● streaming")
	}
	return l + strings.Repeat(" ", max(1, d.w-lipgloss.Width(l)-lipgloss.Width(state))) + state
}

func (d *demo) replyPanel(w, h int) string {
	iw, ih := inner(w, h)
	text := d.reply.String()
	if text == "" {
		text = sDim().Render("g starts a reply · a runs the scripted demo")
	}
	if d.genErr != nil && !errors.Is(d.genErr, context.Canceled) {
		text += "\n" + sErr().Render(d.genErr.Error())
	}
	wrapped := strings.Split(lipgloss.NewStyle().Width(iw).Foreground(cFg).Render(text), "\n")
	if len(wrapped) > ih {
		wrapped = wrapped[len(wrapped)-ih:]
	}
	title := fmt.Sprintf("reply · %d tokens", d.tokens)
	return panel(title, strings.Join(wrapped, "\n"), w, h, cBrand)
}

func (d *demo) ratePanel(w, h int) string {
	iw, _ := inner(w, h)
	last, peak, mean := rateStats(d.rates)
	head := sBrand().Render(fmt.Sprintf("%.1f", last)) + sDim().Render(" tok/s") +
		sDim().Render(fmt.Sprintf("   peak %.1f   mean %.1f", peak, mean))
	if len(d.rates) == 0 {
		head = sDim().Render("decode rate appears with the first tokens")
	}
	return panel("decode tok/s", head+"\n"+sparkline(d.rates, d.marks, iw, sparkRows), w, h, cBlue)
}

func (d *demo) footer() string {
	keys := sDim().Render("m") + sMuted().Render(" next device  ") + sDim().Render("h") + sMuted().Render(" home  ") +
		sDim().Render("1-9") + sMuted().Render(" block group  ") + sDim().Render("a") + sMuted().Render(" demo  ") +
		sDim().Render("g") + sMuted().Render(" reply  ") + sDim().Render("s") + sMuted().Render(" stop  ") + sDim().Render("q") + sMuted().Render(" quit")
	proof := sDim().Render(fmt.Sprintf("prompt rows recomputed across %d move(s): ", d.moves)) +
		sBrand().Render(fmt.Sprint(d.recomputed)) +
		sDim().Render(fmt.Sprintf(" · position %d of %d", d.pos, d.maxSeq))
	if d.note != "" {
		proof = lipgloss.NewStyle().Foreground(cOrange).Render(d.note)
	}
	return keys + "\n" + proof
}

// rateStats is the newest sample, the peak and the mean.
func rateStats(r []float64) (last, peak, mean float64) {
	if len(r) == 0 {
		return
	}
	last = r[len(r)-1]
	var sum float64
	for _, v := range r {
		peak = max(peak, v)
		sum += v
	}
	return last, peak, sum / float64(len(r))
}

// sparkline draws the newest w samples rows cells tall, in eighths. A sample a
// move landed in is drawn orange.
func sparkline(v []float64, marks []int, w, rows int) string {
	if len(v) > w {
		off := len(v) - w
		v = v[off:]
		marks = slices.Clone(marks)
		for i := range marks {
			marks[i] -= off
		}
	}
	top := 1.0
	for _, x := range v {
		top = max(top, x)
	}
	glyphs := []rune(" ▁▂▃▄▅▆▇█")
	out := make([]string, rows)
	for r := range rows {
		var b strings.Builder
		for i, x := range v {
			level := int(x/top*float64(rows*8) + 0.5)
			cell := min(max(level-(rows-1-r)*8, 0), 8)
			st := lipgloss.NewStyle().Foreground(cBrand)
			if slices.Contains(marks, i) {
				st = lipgloss.NewStyle().Foreground(cOrange)
				if cell == 0 {
					cell = 1
					if r != rows-1 {
						cell = 0
					}
				}
			}
			b.WriteString(st.Render(string(glyphs[cell])))
		}
		out[r] = b.String()
	}
	return strings.Join(out, "\n")
}

// banner is what the last move reported, or the move in flight.
func banner(m *moveInfo, pending *moveReq, blocks []mapBlock, w, h int) string {
	if pending != nil {
		to := columnLabel(pending.To)
		body := lipgloss.NewStyle().Foreground(cOrange).Bold(true).Render("moving") +
			sFg().Render(fmt.Sprintf(" blocks %s to %s", span(pending.First, pending.Last), to)) +
			sDim().Render("  waiting for the service")
		return panel("move", body, w, h, cOrange)
	}
	if m == nil {
		return panel("move", sDim().Render("nothing moved yet. m moves the device blocks to the next device, h brings them home."), w, h, cLine)
	}
	if m.Err != nil {
		iw, _ := inner(w, h)
		return panel("move", lipgloss.NewStyle().Foreground(cRed).Width(iw).Render("the service refused: "+connect.CodeOf(m.Err).String()+": "+errText(m.Err)), w, h, cRed)
	}
	r := m.Resp
	arrow := lipgloss.NewStyle().Foreground(cBrand).Bold(true).Render(" → ")
	l1 := sBold().Render(columnLabel(m.From)) + arrow + sBold().Render(columnLabel(m.To)) +
		sDim().Render("   blocks ") + sFg().Render(fmt.Sprintf("%s (%d moved)", span(m.First, m.Last), r.GetBlocksMoved())) +
		sDim().Render("   history moved ") + sFg().Render(session.Bytes(r.GetHistory().GetBytes())) +
		sDim().Render("   ") + pauseText(r, m.Took)
	if m.AtToken > 0 {
		l1 += sDim().Render(fmt.Sprintf("   at token %d", m.AtToken))
	}
	path := "host cache"
	switch r.GetPath() {
	case v1.RelocationPath_RELOCATION_PATH_PEER:
		path = "card to card"
	case v1.RelocationPath_RELOCATION_PATH_STAGED:
		path = "through a host buffer"
	case v1.RelocationPath_RELOCATION_PATH_UNSPECIFIED:
		path = "host and device"
	}
	head := "no"
	if r.GetHeadMoved() {
		head = "yes"
	}
	l2 := sDim().Render("path ") + sFg().Render(path)
	if c := r.GetCopied().GetBytes(); c > 0 {
		l2 += sDim().Render("   copied ") + sFg().Render(session.Bytes(c))
	}
	l2 += sDim().Render("   prompt rows recomputed ") + sBrand().Render(fmt.Sprint(m.Recomputed)) +
		sDim().Render("   projection moved ") + sFg().Render(head)
	l3 := ""
	if n := r.GetStagedBlocks(); n > 0 {
		l3 = sDim().Render(fmt.Sprintf("%d block(s) prepared in %d ms while the reply kept decoding", n, r.GetPrepareMillis()))
	}
	for _, dec := range r.GetDeclined() {
		l3 += "  " + lipgloss.NewStyle().Foreground(cOrange).Render(fmt.Sprintf("block %d declined: %s", dec.GetBlockIndex(), dec.GetReason()))
	}
	return panel("move", strings.Join([]string{l1, l2, l3}, "\n"), w, h, cBrand)
}

// pauseText is what the sequence waited, or the call's own time when the
// service reports no pause (a move between the host and a device).
func pauseText(r *v1.RelocateBlocksResponse, took time.Duration) string {
	if us := r.GetPauseMicros(); us > 0 {
		return sDim().Render("paused ") + sBrand().Render(fmt.Sprintf("%.1f ms", float64(us)/1000))
	}
	return sDim().Render("took ") + sFg().Render(fmt.Sprintf("%d ms", max(r.GetMillis(), took.Milliseconds())))
}

func span(first, last int) string {
	if first == last {
		return fmt.Sprint(first)
	}
	return fmt.Sprintf("%d-%d", first, last)
}

func errText(err error) string {
	var ce *connect.Error
	if errors.As(err, &ce) {
		return ce.Message()
	}
	return err.Error()
}
