package screen

import (
	"fmt"

	"github.com/gogpu/ui/state"
	"github.com/gogpu/ui/widget"

	"github.com/jitllm/jitllm/ui/app"
	"github.com/jitllm/jitllm/ui/widgets"
)

// memoryMap is the Placement section's picture: where each block runs, and
// what every tier of memory holds of the model.
func memoryMap(sh *app.Shell) widget.Widget {
	st := sh.Store
	view := state.NewComputed(func() widgets.MemView {
		return MemView(st.BlockMap.Get(), st.Alloc.Get())
	}, st.BlockMap.AsReadonly(), st.Alloc.AsReadonly())
	c := sh.P.Colors
	return widgets.NewMemoryMap(view, widgets.MemColors{
		Host:    c.Outline,
		Devices: []widget.Color{c.Primary, c.Tertiary, c.Secondary},
		Dense:   c.OnSurfaceVariant,
		KV:      sh.P.Warn,
		Other:   c.OutlineVariant,
		Track:   c.SurfaceVariant,
		Line:    c.Outline,
		Seam:    c.Error,
		Text:    sh.P.Text(),
		Muted:   sh.P.Muted(),
		OnCell:  c.OnPrimary,
	}, sh.P.Type.BodySmall.FontSize)
}

// MemView lays the allocation out as lanes: the file, the host, then each
// device in the tier's order. Nothing is summed across lanes, for the reason
// DeviceUseLine names the pool.
func MemView(blocks []byte, a app.Allocation) widgets.MemView {
	v := widgets.MemView{Blocks: blocks, Pos: a.Pos, MaxSeq: a.MaxSeq}
	if len(blocks) == 0 {
		return v
	}
	n := len(blocks)

	all := make([]int, n)
	var host []int
	for i, b := range blocks {
		all[i] = i
		if widgets.Placement(b).Resident() {
			host = append(host, i)
		}
	}
	file := uint64(n)*a.PageBytes + a.Dense
	v.Lanes = append(v.Lanes, widgets.MemLane{
		Name: "file (.jlm)", Cells: all, CellBytes: a.PageBytes, Disk: true,
		Segs: []widgets.MemSeg{{Kind: widgets.SegDense, Bytes: a.Dense}},
		Caption: fmt.Sprintf("every block, always: %d page(s) of %s + %s dense = %s",
			n, app.Bytes(a.PageBytes), app.Bytes(a.Dense), app.Bytes(file)),
	})

	hostLane := widgets.MemLane{Name: "host memory", Cells: host, CellBytes: a.PageBytes, Caption: HostLine(a)}
	if a.HostBudget > a.HostUsed {
		hostLane.Segs = append(hostLane.Segs, widgets.MemSeg{Kind: widgets.SegFree, Bytes: a.HostBudget - a.HostUsed})
	}
	hostLane.Segs = append(hostLane.Segs,
		widgets.MemSeg{Kind: widgets.SegDense, Bytes: a.Dense},
		widgets.MemSeg{Kind: widgets.SegKV, Bytes: a.CPUKV})
	if a.CPUKV > 0 {
		hostLane.Caption += "  + " + app.Bytes(a.CPUKV) + " kv"
	}
	v.Lanes = append(v.Lanes, hostLane)

	pools := map[string]int{}
	for _, d := range a.Devices {
		pools[d.Pool]++
	}
	for k, d := range a.Devices {
		name := d.Short
		if name == "" {
			name = d.Name
		}
		v.Devices = append(v.Devices, name)
		var cells []int
		for i, b := range blocks {
			if widgets.Placement(b).Device() == k {
				cells = append(cells, i)
			}
		}
		l := widgets.MemLane{Name: name, Cells: cells, Caption: d.Name + ": " + DeviceUseLine(d)}
		if len(cells) > 0 {
			l.CellBytes = d.Weights / uint64(len(cells))
		}
		l.Segs = append(l.Segs, widgets.MemSeg{Kind: widgets.SegKV, Bytes: d.KV})
		if d.KVPool > d.KV {
			l.Segs = append(l.Segs, widgets.MemSeg{Kind: widgets.SegKVFree, Bytes: d.KVPool - d.KV})
		}
		// A pool two devices share holds both of their bytes, so its remainder
		// is nobody's scratch and its free room is not this device's alone.
		if pools[d.Pool] == 1 {
			if held := d.Weights + max(d.KVPool, d.KV); d.Used > held {
				l.Segs = append(l.Segs, widgets.MemSeg{Kind: widgets.SegOther, Bytes: d.Used - held})
			}
			if d.Limit > d.Used {
				l.Segs = append(l.Segs, widgets.MemSeg{Kind: widgets.SegFree, Bytes: d.Limit - d.Used})
			}
		}
		v.Lanes = append(v.Lanes, l)
	}
	return v
}
