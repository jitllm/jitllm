package main

import (
	"context"
	"errors"
	"fmt"
	"os"
	"strings"
	"sync"

	"github.com/charmbracelet/bubbles/progress"
	tea "github.com/charmbracelet/bubbletea"
	"github.com/charmbracelet/lipgloss"
	"github.com/jitllm/jitllm/common/convertjob"
	"github.com/jitllm/jitllm/common/discover"
	"github.com/jitllm/jitllm/convert/hf"
	"github.com/jitllm/jitllm/convert/library"
)

// jobs is the one conversion or download running, as Models and Discover
// show it. One at a time, as in the window: two would fight over the disk and
// the engine's queue.
type jobs struct {
	env *env
	bar progress.Model

	title, stage, label, err string
	running, done            bool
	// dst is the container the job writes, opened when it is done.
	dst string

	mu   sync.Mutex
	stop context.CancelFunc // a running download's
}

// jobMsg is one step of the running job, from its goroutine.
type jobMsg struct {
	stage string
	frac  float64
	label string
	err   string
	done  bool
}

func newJobs(env *env) *jobs {
	return &jobs{env: env, bar: progress.New(progress.WithGradient(string(cBrand), string(cCyan)),
		progress.WithFillCharacters('━', '━'), progress.WithoutPercentage())}
}

// send hands a step to the loop without blocking the job.
func (j *jobs) send(m jobMsg) { j.env.f.queue <- m }

func (j *jobs) busy() bool { return j.running }

// start begins a job, or refuses while one runs.
func (j *jobs) start(title, stage, dst string) bool {
	if j.running {
		return false
	}
	j.title, j.stage, j.label, j.err, j.dst = title, stage, "", "", dst
	j.running, j.done = true, false
	return true
}

// convert runs req on the engine's queue, where it cannot overlap a decode.
func (j *jobs) convert(req convertjob.Request, title string) tea.Cmd {
	plan, err := convertjob.Plan(req)
	if err != nil {
		return notify(err.Error())
	}
	if !j.start(title, "converting", plan.Dst) {
		return notify("a job is running: wait for " + j.title)
	}
	if !j.env.e.Run(func() { convertjob.Run(plan, j.convertSteps(nil)) }) {
		j.running, j.err = false, "the engine is busy: try again in a moment"
	}
	return j.bar.SetPercent(0)
}

// convertSteps turns a conversion's progress into job steps; cleanup runs
// when it is done.
func (j *jobs) convertSteps(cleanup func()) convertjob.Progress {
	return func(u convertjob.Update) {
		switch u.Stage {
		case convertjob.Started, convertjob.Writing:
			j.send(jobMsg{stage: "converting", frac: u.Fraction, label: u.Label})
		case convertjob.Done:
			if cleanup != nil {
				cleanup()
			}
			j.send(jobMsg{frac: 1, label: u.Label, done: true})
		case convertjob.Failed:
			msg := "conversion failed"
			if u.Err != nil {
				msg = u.Err.Error()
			}
			j.send(jobMsg{err: msg, done: true})
		}
	}
}

// download turns a library model into its container, converting the GGUF as
// it streams from the Hub, as the window's Discover tab does
// ([discover.Get]). cancel stops it, keeping what it wrote.
func (j *jobs) download(lm library.Model) tea.Cmd {
	dir := discover.Dir(j.env.dirs, j.env.cfg.DefaultModelDir())
	dst := discover.ContainerFor(dir, lm)
	if note := discover.SpaceNote(lm, dir, discover.FreeBytes); note != "" {
		return notify(note)
	}
	if !j.start(lm.Title, "downloading", dst) {
		return notify("a job is running: wait for " + j.title)
	}
	ctx, cancel := context.WithCancel(context.Background())
	j.mu.Lock()
	j.stop = cancel
	j.mu.Unlock()
	hub := discover.Hub{
		Endpoint: j.env.hub,
		Token:    hf.FindToken(os.Getenv, os.UserHomeDir),
		// The fallback's conversion goes on the engine's queue, where it
		// cannot overlap a decode; Get waits for it.
		Convert: func(job func()) bool {
			done := make(chan struct{})
			if !j.env.e.Run(func() { defer close(done); job() }) {
				return false
			}
			<-done
			return true
		},
	}
	go func() {
		defer cancel()
		line, err := discover.Get(ctx, lm, dir, hub, func(s discover.Step) {
			j.send(jobMsg{stage: s.Stage, frac: s.Fraction, label: s.Label})
		})
		switch {
		case errors.Is(err, context.Canceled):
			j.send(jobMsg{err: "stopped: enter on " + lm.Title + " carries on where it stopped", done: true})
		case err != nil:
			j.send(jobMsg{err: discover.FailMessage(lm, err), done: true})
		default:
			j.send(jobMsg{frac: 1, label: line, done: true})
		}
	}()
	return j.bar.SetPercent(0)
}

// cancel stops a running download.
func (j *jobs) cancel() {
	j.mu.Lock()
	defer j.mu.Unlock()
	if j.stop != nil {
		j.stop()
	}
}

func (j *jobs) Update(msg tea.Msg) tea.Cmd {
	switch msg := msg.(type) {
	case jobMsg:
		if msg.stage != "" {
			j.stage = msg.stage
		}
		if msg.label != "" {
			j.label = msg.label
		}
		if msg.done {
			j.running, j.done, j.err = false, msg.err == "", msg.err
			if msg.err == "" {
				return tea.Batch(j.bar.SetPercent(1), rescan(), notify(j.title+" is ready: enter loads it"))
			}
			return notify("failed: " + j.title)
		}
		return j.bar.SetPercent(msg.frac)
	case progress.FrameMsg:
		m, cmd := j.bar.Update(msg)
		j.bar = m.(progress.Model)
		return cmd
	}
	return nil
}

// View is the job's card, w cells wide outside its frame; empty when there
// has been no job.
func (j *jobs) View(w int) string {
	if j.title == "" {
		return ""
	}
	iw := w - sCard().GetHorizontalFrameSize()
	j.bar.Width = iw
	var b strings.Builder
	head := sBold().Render(truncate(j.title, iw-14))
	switch {
	case j.err != "":
		b.WriteString(sErr().Bold(true).Render("✗ ") + head + "\n")
		b.WriteString(lipgloss.NewStyle().Foreground(cRed).Width(iw).Render(j.err))
	case j.done:
		b.WriteString(sBrand().Render("✓ ") + head + "\n")
		b.WriteString(lipgloss.NewStyle().Foreground(cMuted).Width(iw).Render(j.label))
	default:
		b.WriteString(j.env.spin.View() + " " + sMuted().Render(j.stage) + " " + head + "\n")
		b.WriteString(j.bar.View() + "\n")
		b.WriteString(sDim().Render(truncate(j.label, iw)))
	}
	return sCard().Width(w - sCard().GetHorizontalBorderSize()).Render(b.String())
}

// header is the job in a few words, for the top bar.
func (j *jobs) header() string {
	if !j.running {
		return ""
	}
	return j.env.spin.View() + " " + sMuted().Render(fmt.Sprintf("%s %s · %.0f%%", j.stage, j.title, 100*j.bar.Percent()))
}
