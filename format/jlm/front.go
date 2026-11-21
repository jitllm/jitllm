package jlm

import (
	"fmt"
	"slices"
	"strconv"
	"strings"
)

// FrontGraphs are the graphs of the model's parts outside its blocks where
// the converter states them, each as the graph package stores one
// (graph.Encode), by name (FrontGraphs.valid): the input rows, the head and
// its forms, a readout, and each prediction block's entry and head. The
// engine decodes a stated part's graph rather than building it. The
// thirty-first optional tail; see encodeConfig.
type FrontGraphs struct {
	Names  []string
	Graphs []string
}

// The parts a front statement names. A prediction block's parts are named
// with its index among the prediction blocks after a dot (predict.entry.0).
const (
	FrontEmbed           = "embed"
	FrontGiven           = "given"
	FrontHead            = "head"
	FrontHeadHidden      = "head.hidden"
	FrontCollapsed       = "head.collapsed"
	FrontCollapsedHidden = "head.collapsed.hidden"
	FrontOutNorm         = "outnorm"
	FrontPool            = "pool"
	FrontPredictEntry    = "predict.entry"
	FrontPredictHead     = "predict.head"
	FrontPredictHidden   = "predict.hidden"
)

// FrontPredict is prediction block i's part named part (FrontPredictEntry,
// FrontPredictHead, FrontPredictHidden).
func FrontPredict(part string, i int) string { return part + "." + strconv.Itoa(i) }

func encodeFrontGraphs(w *wbuf, g *FrontGraphs) {
	w.u32(uint32(len(g.Names)))
	for i, n := range g.Names {
		w.str(n)
		w.str(g.Graphs[i])
	}
}

func decodeFrontGraphs(r *rbuf) FrontGraphs {
	var g FrontGraphs
	// A part is at least its name's length and its graph's.
	for range r.count(8) {
		g.Names = append(g.Names, r.str())
		g.Graphs = append(g.Graphs, r.str())
	}
	return g
}

// valid refuses a statement naming a part twice, a part no model has, a
// prediction block past the predict blocks there are, or an empty graph.
func (g *FrontGraphs) valid(predict int) error {
	if len(g.Names) != len(g.Graphs) {
		return fmt.Errorf("jlm: %d front parts named for %d graphs", len(g.Names), len(g.Graphs))
	}
	parts := []string{FrontEmbed, FrontGiven, FrontHead, FrontHeadHidden, FrontCollapsed, FrontCollapsedHidden, FrontOutNorm,
		FrontPool}
	for i, n := range g.Names {
		if slices.Contains(g.Names[:i], n) {
			return fmt.Errorf("jlm: the front part %q is stated twice", n)
		}
		if len(g.Graphs[i]) == 0 || len(g.Graphs[i]) > MaxBlockGraph {
			return fmt.Errorf("jlm: the front part %q's graph is %d bytes", n, len(g.Graphs[i]))
		}
		if slices.Contains(parts, n) {
			continue
		}
		part, at, ok := strings.Cut(n[min(len(n), len("predict.")):], ".")
		k, err := strconv.Atoi(at)
		if !ok || !strings.HasPrefix(n, "predict.") || err != nil || strconv.Itoa(k) != at ||
			!slices.Contains([]string{"entry", "head", "hidden"}, part) {
			return fmt.Errorf("jlm: %q names no front part", n)
		}
		if k >= predict {
			return fmt.Errorf("jlm: the front part %q names prediction block %d of %d", n, k, predict)
		}
	}
	return nil
}

// Graph is the stated graph of the part named name, empty where the
// converter states none.
func (g *FrontGraphs) Graph(name string) string {
	if i := slices.Index(g.Names, name); i >= 0 {
		return g.Graphs[i]
	}
	return ""
}

// Stated reports whether the converter states the model's parts.
func (g *FrontGraphs) Stated() bool { return len(g.Names) != 0 }
