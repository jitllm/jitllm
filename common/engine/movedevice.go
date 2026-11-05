package engine

import (
	"context"
	"fmt"
	"slices"
	"time"

	"connectrpc.com/connect"

	"github.com/jitllm/jitllm/common/session"
	"github.com/jitllm/jitllm/engine"
	"github.com/jitllm/jitllm/server/client"
	v1 "github.com/jitllm/jitllm/server/gen/jitllm/v1"
)

// nextDevice is the device after cur in names, wrapping; "" when there is no
// other device to go to.
func nextDevice(names []string, cur string) string {
	if len(names) < 2 {
		return ""
	}
	i := slices.Index(names, cur)
	return names[(i+1)%len(names)]
}

// moveNote is the status line for a finished move between devices.
func moveNote(n int, to string, took time.Duration, history uint64, via string) string {
	return fmt.Sprintf("moved %d block(s) to %s in %s, %s of history via %s",
		n, to, took.Round(time.Millisecond), session.Bytes(history), via)
}

// MoveToNextDevice moves every device block of the chat to the next device
// of its tier, history and all, between turns: the device-to-device
// relocation (server.Engine.MoveBlocks), one contiguous run at a time.
func (e *Engine) MoveToNextDevice() {
	e.post("move", func() {
		if e.m == nil || e.sess == nil || e.gpu == nil {
			e.status("load a model on a device to move its blocks")
			return
		}
		var names []string
		for _, b := range e.gpu.Budgets() {
			names = append(names, b.Name)
		}
		var runs [][2]int
		cur := ""
		e.inspect(func(st *engine.Instance) {
			runs = st.DeviceBlocks()
			if len(runs) > 0 {
				cur, _ = st.DeviceOf(runs[0][0])
			}
		})
		to := nextDevice(names, cur)
		if len(runs) == 0 || to == "" {
			e.status("there is no other device to move the blocks to")
			return
		}
		start := time.Now()
		var moved int
		var hist uint64
		via := ""
		for _, r := range runs {
			mv, err := e.srv.MoveBlocks(context.Background(), e.sess, r[0], r[1]-1, to)
			if err != nil {
				e.status("moving blocks to %s: %v", to, err)
				break
			}
			moved += len(mv.Moved)
			hist += mv.KVBytes + mv.RecBytes
			via = mv.Path
		}
		nb := e.m.Cfg.NLayer
		var blocks []byte
		var alloc session.Allocation
		e.inspect(func(st *engine.Instance) {
			blocks = placementOf(e.m, st, e.gpu)
			alloc = allocStat(e.m, st, e.gpu, nb)
		})
		msg := moveNote(moved, to, time.Since(start), hist, via)
		e.sh.Post(func() {
			e.st.BlockMap.Set(blocks)
			e.st.Alloc.Set(alloc)
			e.st.Status.Set(msg)
		})
	})
}

// MoveToNextDevice asks the service to move every device block of the chat
// to the next device of its session, history and all.
func (r *Remote) MoveToNextDevice() {
	r.post("move", func() {
		if r.session == "" {
			r.status("no model is loaded")
			return
		}
		ctx, cancel := bg()
		defer cancel()
		ss, err := r.c.Session.GetSession(ctx, connect.NewRequest(&v1.GetSessionRequest{SessionId: r.session}))
		if err != nil {
			r.fail("reading the session of", r.active, err)
			return
		}
		pl, err := r.c.Placement.GetPlacement(ctx, connect.NewRequest(&v1.GetPlacementRequest{SessionId: r.session}))
		if err != nil {
			r.fail("reading the placement of", r.active, err)
			return
		}
		runs, cur := deviceRuns(pl.Msg.GetPlacement())
		to := nextDevice(ss.Msg.GetSession().GetDeviceIds(), cur)
		if len(runs) == 0 || to == "" {
			r.status("there is no other device to move the blocks to")
			return
		}
		moved, hist, via, took, err := relocateRuns(r.c, r.session, runs, to)
		if err != nil {
			r.fail("moving the blocks of", r.active, err)
		}
		r.publishSession()
		msg := moveNote(moved, to, took, hist, via)
		r.sh.Post(func() { r.st.Status.Set(msg) })
	})
}

// deviceRuns is the placement's device blocks as contiguous runs, and the
// device of the first.
func deviceRuns(p *v1.Placement) ([][2]int32, string) {
	var runs [][2]int32
	cur := ""
	for _, b := range p.GetBlocks() {
		if b.GetLocation() != v1.BlockLocation_BLOCK_LOCATION_DEVICE {
			continue
		}
		if cur == "" {
			cur = b.GetDeviceId()
		}
		if n := len(runs); n > 0 && runs[n-1][1] == b.GetIndex()-1 {
			runs[n-1][1] = b.GetIndex()
			continue
		}
		runs = append(runs, [2]int32{b.GetIndex(), b.GetIndex()})
	}
	return runs, cur
}

// relocateRuns moves each run to the device named to through RelocateBlocks.
func relocateRuns(c *client.Conn, sid string, runs [][2]int32, to string) (int, uint64, string, time.Duration, error) {
	ctx, cancel := bg()
	defer cancel()
	var moved int
	var hist uint64
	var took time.Duration
	via := ""
	for _, r := range runs {
		resp, err := c.Placement.RelocateBlocks(ctx, connect.NewRequest(&v1.RelocateBlocksRequest{
			SessionId: sid, FirstBlock: r[0], LastBlock: r[1], ToDeviceId: to,
		}))
		if err != nil {
			return moved, hist, via, took, err
		}
		m := resp.Msg
		moved += int(m.GetBlocksMoved())
		hist += m.GetHistory().GetBytes()
		took += time.Duration(m.GetMillis()) * time.Millisecond
		via = pathName(m.GetPath())
	}
	return moved, hist, via, took, nil
}

// pathName is a RelocationPath as the status line says it.
func pathName(p v1.RelocationPath) string {
	switch p {
	case v1.RelocationPath_RELOCATION_PATH_HOST:
		return "host"
	case v1.RelocationPath_RELOCATION_PATH_STAGED:
		return "staged"
	case v1.RelocationPath_RELOCATION_PATH_PEER:
		return "peer"
	}
	return "device"
}
