package engine

import (
	"fmt"
	"strings"

	"github.com/jitllm/jitllm/engine/device/cpu"
	"github.com/jitllm/jitllm/engine/device/gpu"
	"github.com/jitllm/jitllm/nn"
)

// offCards is the blocks of [lo, hi) the engine was to place and no card
// holds: what is left on the CPU, less the blocks a placement keeps there.
func (s *Instance) offCards(lo, hi int) []int {
	var off []int
	for li := lo; li < hi; li++ {
		if s.devAt(li) {
			continue
		}
		if pl, named := s.placeOf(li); named && onCPU(pl.On) {
			continue
		}
		off = append(off, li)
	}
	return off
}

// pageOnCards is the placement of an instance whose caller did not list the CPU
// (WithDevices without cpu) when the cards do not hold every block resident. Every
// block it decides is marked streamed (Place.Stream) where it is -- a block
// already on a card stays there, formed once, and becomes a page the pager
// may take (the device pager's victim()) -- and the blocks no card took are
// offered again, so they page through the cards' slots instead of running on
// the CPU, a mixture's routed experts streamed to the card rather than run on
// the host. A block that not even that places -- a plan no card runs, a
// budget below the slots a swap needs -- is the placement's error, by name.
func (s *Instance) pageOnCards(lo, hi int) error {
	if s.listed(routeLanes) || s.ld == nil || s.ldCand == nil {
		return nil
	}
	off := s.offCards(lo, hi)
	if len(off) == 0 {
		return nil
	}
	nd, ok := s.ld.(nn.NamedDevice)
	if !ok {
		return errCPUUnlisted(fmt.Sprintf("blocks %v fit no card resident and this device cannot page a block", off))
	}
	ep, _ := s.ld.(nn.ExpertPlacer)
	for li := lo; li < hi; li++ {
		pl, named := s.placeOf(li)
		if named && (onCPU(pl.On) || pl.On != "" && !pl.Stream) {
			continue
		}
		nd.Stream(li, true)
		if ep != nil && !s.devAt(li) && s.c.MoEAt(li) && s.opt.experts == "" && !(named && pl.Experts != "") {
			ep.PlaceExperts(li, "card")
		}
	}
	s.offerRange(lo, hi)
	if s.placeErr != nil {
		return s.placeErr
	}
	if off = s.offCards(lo, hi); len(off) == 0 {
		return nil
	}
	why := "the cards refused them"
	if er, ok := s.ld.(nn.ErrReporter); ok && er.Err() != "" {
		why = er.Err()
	} else if d := s.DeviceDeclines(); len(d) > 0 {
		why = d[0].Why
	}
	return errCPUUnlisted(fmt.Sprintf("blocks %v run on no listed card even paged (%s)", off, why))
}

// cpuOfDevices is what the device specs a model was opened with state about
// the CPU device: whether it is listed (an entry naming cpu, or auto -- the
// default with no specs) and the options its entries' variants give its
// Target (cpu.SpecOptions).
func cpuOfDevices(specs []string) (opts []cpu.Option, listed bool, err error) {
	spec := "auto"
	if len(specs) > 0 {
		spec = strings.Join(specs, ",")
	}
	es, err := gpu.ParseDevices(spec)
	if err != nil {
		return nil, false, fmt.Errorf("model: %w", err)
	}
	for _, e := range es {
		switch e.API {
		case "auto":
			listed = true
		case "cpu":
			listed = true
			o, err := cpu.SpecOptions(e.Sel)
			if err != nil {
				return nil, false, fmt.Errorf("model: %w", err)
			}
			opts = append(opts, o...)
		}
	}
	return opts, listed, nil
}
