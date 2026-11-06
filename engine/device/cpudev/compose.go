package cpudev

import (
	"errors"
	"runtime"

	"github.com/jitllm/jitllm/composer"
	hw "github.com/jitllm/jitllm/engine/cost"
	"github.com/jitllm/jitllm/engine/device/cpu"
	"github.com/jitllm/jitllm/engine/probe"
	"github.com/jitllm/jitllm/graph"
	"github.com/jitllm/jitllm/nn"
)

// Composer is the CPU device's composition for one instance: the model's
// forms on this CPU (Forms), the target they are composed for -- its ISA,
// its workers and the launch words its profile states, probed once -- the
// float reference where it is asked for, and the first error a form met.
type Composer struct {
	forms   *Forms
	isa     cpu.ISA
	workers int
	float   bool
	launch  int
	probed  bool
	err     error
}

// NewComposer is a composition through forms for a CPU of ISA isa and
// workers workers; float forms the float reference.
func NewComposer(forms *Forms, isa cpu.ISA, workers int, float bool) *Composer {
	return &Composer{forms: forms, isa: isa, workers: workers, float: float}
}

// Target is the composer's description of a CPU of ISA isa and workers
// workers.
func Target(isa cpu.ISA, workers int) composer.Device {
	return composer.Device{Kind: composer.DeviceCPU, Slots: max(workers, 1), Lanes: 8, ISA: uint64(isa), ExactDiv: true}
}

// Lowers says the composer lowers kernels for this CPU.
func (c *Composer) Lowers() bool { return composer.LowersCPU(uint64(c.isa)) }

// ISA is the CPU's instruction set the forms are composed for.
func (c *Composer) ISA() cpu.ISA { return c.isa }

// Device is the target the forms are composed for.
func (c *Composer) Device() composer.Device {
	d := Target(c.isa, c.workers)
	if !c.probed {
		if name, err := probe.HostName(); err == nil {
			if p, ok := hw.Known(name, runtime.GOARCH); ok {
				c.launch = p.Device.LaunchWords()
			}
		}
		c.probed = true
	}
	d.Launch = c.launch
	return d
}

// Form is request req's form of graph gr at rows rows, through the model's
// cache.
func (c *Composer) Form(req composer.Request, gr *graph.Graph, rows int, apart, whole, f32 bool) (*Form, error) {
	return c.forms.Of(req, gr, rows, apart, whole, f32, c.float)
}

// Decline is the reason err declines its form by, the first error kept.
func (c *Composer) Decline(err error) nn.Decline {
	c.Note(err)
	var d DeclineError
	if errors.As(err, &d) {
		return d.Why
	}
	return nn.DeclineNotFormed
}

// Note keeps err where no error is kept yet.
func (c *Composer) Note(err error) {
	if c.err == nil {
		c.err = err
	}
}

// Err is the first error a form met, nil for none.
func (c *Composer) Err() error { return c.err }

// Block is a block's composed forms on the CPU device: its graph, the form
// of one row and the rows' form, or why each is not served.
type Block struct {
	Graph *graph.Graph
	F     *Form
	Rows  *Form
	// SideOff is where the block's slice of the rows' per-layer inputs
	// starts in each row: the call binds the side input there, so blocks
	// that differ by their slice alone share a form.
	SideOff int
	Why     nn.Decline
	// RowsWhy is why the rows' form is not served, where it is not.
	RowsWhy nn.Decline
	// Bidir says the block attends bidirectionally inside a prompt's runs,
	// Whole that its rows read the whole picture whatever their windows.
	Bidir, Whole bool
	// Rec is a linear block (nn.BlockSpec.Rec).
	Rec bool
}
