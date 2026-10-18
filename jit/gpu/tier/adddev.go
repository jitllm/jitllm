package tier

import (
	"fmt"

	"github.com/jitllm/jitllm/jit/gpu/backend"
)

// AddDevices opens the devices spec names (ParseDevices' grammar) and adds
// them to the tier after the ones it holds, budgeted and pooled as OpenWith
// does, so a model's blocks can be placed or moved onto them without a second
// tier: the plasticity a running model needs to take a card it was not loaded
// with. A device the tier already holds (the same physical device, by
// identity) is refused, as OpenWith refuses one named twice; "cpu" adds
// nothing. It returns the names of the devices added.
//
// The caller quiesces the tier first: no placement, step or move may run
// across the call, since the device list is read without a lock everywhere
// else (a server holds every session of the model off the tier for it).
func (g *GPU) AddDevices(spec string, options ...Option) ([]string, error) {
	var o openOpts
	for _, f := range options {
		f(&o)
	}
	o.apply()
	es, err := ParseDevices(spec)
	if err != nil {
		return nil, err
	}
	devs, asks, names, err := openEntries(es, o)
	if err != nil {
		return nil, err
	}
	if len(devs) == 0 {
		return nil, nil
	}
	closeNew := func() {
		for _, d := range devs {
			d.Close()
		}
	}
	all := make([]backend.Device, 0, len(g.devs)+len(devs))
	for _, d := range g.devs {
		all = append(all, d.dev)
	}
	if g.xdev != nil {
		all = append(all, g.xdev.dev)
	}
	if err := noDuplicates(append(all, devs...), spec); err != nil {
		closeNew()
		return nil, fmt.Errorf("tier: %w (a device the tier already holds)", err)
	}
	slots := planSlots(devs, asks, o)
	g.mu.Lock()
	defer g.mu.Unlock()
	base := len(g.devs)
	if g.xdev != nil {
		base++
	}
	added := make([]string, len(slots))
	for i := range slots {
		slots[i].Name = names[i]
		g.addSlot(base+i, slots[i])
		added[i] = names[i]
	}
	return added, nil
}
