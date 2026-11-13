package oracle

import (
	"math"
	"sort"
)

// ChainStage is one of llama.cpp's samplers as common_sampler_init chains
// them in a samplers list's order. Name is the list's: "penalties",
// "top_n_sigma", "top_k", "typ_p", "top_p", "min_p", "xtc" or
// "temperature". P is top-p, min-p, typical-p, top-n-sigma's n, or XTC's
// threshold where its chance fell (+Inf where it did not); InvT the
// reciprocal temperature as the draw reads it (binary32) and Dyn the
// dynamic temperature's range and exponent; Hist the penalties' window.
type ChainStage struct {
	Name                     string
	K                        int
	P, InvT                  float64
	Dyn                      [2]float64
	Hist                     []int
	Rep, Presence, Frequency float32
}

// Chain runs stages over logits in their order, as llama.cpp's chain runs
// its samplers over the candidates (llama_sampler_*_apply), a token a stage
// removes -Inf: the penalties (Penalize); top-n-sigma (TopNSigma); top-k,
// the k largest (the lower id of two equal); typical-p (typicalCut) over
// the survivors' softmax; top-p, those before the first whose mass up to
// and including it reaches p, it kept, over the survivors' softmax in
// their order; min-p, the logits at least the largest plus log p; XTC
// (xtcCut); the temperature, each survivor's logit times the reciprocal or
// the dynamic one over the survivors' softmax (DynTemp), in binary32. A
// stage that is off leaves the row. It returns the row and how near a cut
// came to moving.
func Chain(logits []float32, stages []ChainStage) ([]float32, float64) {
	v := append([]float32(nil), logits...)
	margin := math.Inf(1)
	out := float32(math.Inf(-1))
	drop := func(keep []bool) {
		for i := range v {
			if !keep[i] {
				v[i] = out
			}
		}
	}
	for _, st := range stages {
		in, order, p := chainSurvivors(v)
		switch st.Name {
		case "penalties":
			Penalize(v, st.Hist, st.Rep, st.Presence, st.Frequency)
		case "top_n_sigma":
			var m float64
			v, m = TopNSigma(v, st.P)
			margin = min(margin, m)
		case "top_k":
			if st.K > 0 && st.K < len(order) {
				for _, i := range order[st.K:] {
					v[i] = out
				}
			}
		case "typ_p":
			if st.P > 0 && st.P < 1 {
				keep, m := typicalCut(p, in, st.P)
				margin = min(margin, m)
				drop(keep)
			}
		case "top_p":
			if st.P < 1 {
				before := 0.0
				for j, i := range order {
					if j > 0 {
						margin = min(margin, math.Abs(before-st.P))
						if before >= st.P {
							v[i] = out
						}
					}
					before += p[i]
				}
			}
		case "min_p":
			if st.P > 0 {
				thr := float64(v[order[0]]) + math.Log(st.P)
				for _, i := range order {
					margin = min(margin, math.Abs(float64(v[i])-thr)/(1+math.Abs(thr)))
					if float64(v[i]) < thr {
						v[i] = out
					}
				}
			}
		case "xtc":
			if !math.IsInf(st.P, 1) {
				kept, m := xtcCut(order, func(i int) float64 { return p[i] }, st.P)
				margin = min(margin, m)
				keep := make([]bool, len(v))
				for _, i := range kept {
					keep[i] = true
				}
				drop(keep)
			}
		case "temperature":
			invT := st.InvT
			if st.Dyn[0] > 0 {
				ps := make([]float64, 0, len(order))
				for _, i := range order {
					ps = append(ps, p[i])
				}
				invT = DynTemp(ps, invT, st.Dyn)
			}
			for _, i := range order {
				v[i] *= float32(invT)
			}
		}
	}
	return v, margin
}

// chainSurvivors is v's finite logits, their ids the largest first (the
// lower of two equal), and their softmax, 0 off them.
func chainSurvivors(v []float32) (in []bool, order []int, p []float64) {
	in = make([]bool, len(v))
	top := math.Inf(-1)
	for i, x := range v {
		if !math.IsInf(float64(x), -1) {
			in[i], order = true, append(order, i)
			top = max(top, float64(x))
		}
	}
	sort.SliceStable(order, func(a, b int) bool { return v[order[a]] > v[order[b]] })
	p = make([]float64, len(v))
	mass := 0.0
	for _, i := range order {
		p[i] = math.Exp(float64(v[i]) - top)
		mass += p[i]
	}
	for _, i := range order {
		p[i] /= mass
	}
	return in, order, p
}
