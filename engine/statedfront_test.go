package engine

import (
	"bytes"
	"errors"
	"fmt"
	"testing"

	"github.com/jitllm/jitllm/convert"
	"github.com/jitllm/jitllm/format/jlm"
	"github.com/jitllm/jitllm/graph"
	"github.com/jitllm/jitllm/model"
)

// TestStatedFrontIsTheBuiltOne holds each part outside the blocks a
// container states (jlm.Config.Front) to the graph the front end builds from
// its config, byte for byte, over every model the audit converts at 2 GiB or
// less: the builders are the spec while they exist. A gate that compared
// nothing proves nothing, so it fails below a floor of parts.
func TestStatedFrontIsTheBuiltOne(t *testing.T) {
	compared := map[string]int{}
	for _, src := range auditModels(t) {
		if src.size > 2<<30 {
			continue
		}
		t.Run(src.name, func(t *testing.T) {
			path := src.path
			if src.hf {
				path = jlmOfSafetensors(t, src.path)
			} else {
				p, err := convertPair(src.path, src.mmproj)
				switch {
				case errors.Is(err, convert.ErrNotImplemented), errors.Is(err, convert.ErrPart), errors.Is(err, convert.ErrBadInput):
					t.Skipf("not converted: %v", err)
				case err != nil:
					t.Fatalf("convert: %v", err)
				}
				path = p
			}
			m, err := model.Open(path)
			if err != nil {
				t.Fatal(err)
			}
			defer m.Close()
			for part, d := range statedFrontDiffs(m) {
				compared[part]++
				if d != "" {
					t.Errorf("the stated %s is not the built one: %s", part, d)
				}
			}
		})
	}
	n := 0
	for _, k := range compared {
		n += k
	}
	t.Logf("%d stated parts held to the built ones: %v", n, compared)
	if compared["pickhead"] == 0 {
		t.Errorf("no pick head derived from a stated head was held to the built one")
	}
	if n < 100 {
		t.Fatalf("%d stated parts compared: the gate saw too few to prove anything", n)
	}
}

// statedFrontDiffs is each part m's container states and how its stated
// graph differs from the one the front end builds with the statement taken
// away, "" where they are the same.
func statedFrontDiffs(m *model.Model) map[string]string {
	c := m.Container.Config()
	of := func() map[string]*graph.Graph {
		f := m.FrontGraphs()
		parts := map[string]*graph.Graph{jlm.FrontHead: f.Head, jlm.FrontHeadHidden: f.CpuHidden, jlm.FrontOutNorm: f.OutNorm,
			jlm.FrontEmbed: f.Embed, jlm.FrontGiven: f.Given, jlm.FrontPool: f.Pool,
			jlm.FrontCollapsed: f.CpuHead, jlm.FrontCollapsedHidden: f.CpuHidden}
		for i, g := range f.MtpEntry {
			parts[jlm.FrontPredict(jlm.FrontPredictEntry, i)] = g
			parts[jlm.FrontPredict(jlm.FrontPredictHead, i)] = f.MtpHead[i]
			parts[jlm.FrontPredict(jlm.FrontPredictHidden, i)] = f.MtpHidden[i]
		}
		return parts
	}
	stated := of()
	kept := c.Front
	c.Front = jlm.FrontGraphs{}
	m.Front.Restated++
	built := of()
	c.Front = kept
	m.Front.Restated++
	out := map[string]string{}
	// The pick head is derived from the stated head, not stated: held to the
	// built one up to its values' names and order.
	if kept.Graph(jlm.FrontHead) != "" && m.Cfg.Streams() == 1 {
		s := m.FrontGraphs().PickHead
		c.Front = jlm.FrontGraphs{}
		m.Front.Restated++
		b := m.FrontGraphs().PickHead
		c.Front = kept
		m.Front.Restated++
		if s != nil || b != nil {
			out["pickhead"] = encodedDiff(canonical(s), canonical(b))
		}
	}
	for _, part := range kept.Names {
		s, ok := stated[part]
		if !ok {
			out[part] = "a part the gate does not compare yet"
			continue
		}
		out[part] = encodedDiff(s, built[part])
	}
	return out
}

// canonical is g with its values renumbered in the order its inputs and its
// ops' outputs name them, and unnamed: two graphs of the same ops over the
// same values in another order or under other names encode alike.
func canonical(g *graph.Graph) *graph.Graph {
	if g == nil {
		return nil
	}
	at := map[graph.ValueID]graph.ValueID{}
	var vals []graph.Value
	see := func(v graph.ValueID) graph.ValueID {
		if n, ok := at[v]; ok {
			return n
		}
		n := graph.ValueID(len(vals))
		at[v] = n
		w := g.Values[v]
		w.Name = ""
		vals = append(vals, w)
		return n
	}
	c := &graph.Graph{Version: g.Version, Block: g.Block, Routed: g.Routed}
	for _, v := range g.In {
		c.In = append(c.In, see(v))
	}
	for _, o := range g.Ops {
		n := o
		n.In, n.Out = nil, nil
		for _, v := range o.In {
			n.In = append(n.In, see(v))
		}
		for _, v := range o.Out {
			n.Out = append(n.Out, see(v))
		}
		c.Ops = append(c.Ops, n)
	}
	for _, v := range g.Out {
		c.Out = append(c.Out, see(v))
	}
	c.Values = vals
	return c
}

// encodedDiff is where two graphs' encodings first differ, "" for none.
func encodedDiff(a, b *graph.Graph) string {
	if a == nil || b == nil {
		return fmt.Sprintf("stated %v, built %v", a != nil, b != nil)
	}
	ea, err := graph.Encode(a)
	if err != nil {
		return err.Error()
	}
	eb, err := graph.Encode(b)
	if err != nil {
		return err.Error()
	}
	if bytes.Equal(ea, eb) {
		return ""
	}
	return fmt.Sprintf("encodings of %d and %d bytes differ\nstated\n%s\nbuilt\n%s", len(ea), len(eb), a, b)
}
