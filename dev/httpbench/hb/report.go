package hb

import (
	"encoding/json"
	"fmt"
	"io"
	"math"
	"sort"
	"strings"
	"time"
)

// Stats is one set of samples summarised: a level of one arm, or one model of
// it.
type Stats struct {
	Requests int `json:"requests"`
	Errors   int `json:"errors"`

	TTFTp50 F `json:"ttft_p50_ms"`
	TTFTp90 F `json:"ttft_p90_ms"`
	TTFTp99 F `json:"ttft_p99_ms"`
	ITLp50  F `json:"itl_p50_ms"`
	ITLp99  F `json:"itl_p99_ms"`
	// ReqRate is the median request's decode rate (Sample.DecodeRate).
	ReqRate F `json:"request_tok_s_p50"`
	// AggRate is every completion token over the wall time.
	AggRate F `json:"aggregate_tok_s"`
	// Goodput is the requests per second that met both SLOs.
	Goodput F `json:"goodput_req_s"`

	CompletionTokens int `json:"completion_tokens"`
	// AtMax is how many requests generated exactly max_tokens: whether the
	// engine honoured ignore_eos, measured rather than assumed.
	AtMax int `json:"at_max_tokens"`
	// NoUsage is how many streams carried no usage, whose token counts are
	// chunk counts.
	NoUsage int `json:"no_usage"`
}

func ms(d time.Duration) float64 { return float64(d) / float64(time.Millisecond) }

// Summarise reduces samples run over wall.
func Summarise(ss []Sample, wall time.Duration, c Config) Stats {
	var st Stats
	var ttft, itl, rate []float64
	good := 0
	for i := range ss {
		s := &ss[i]
		st.Requests++
		if s.Err != "" {
			st.Errors++
			continue
		}
		st.CompletionTokens += s.CompletionTokens
		if s.CompletionTokens == c.MaxTokens {
			st.AtMax++
		}
		if !s.UsageSeen {
			st.NoUsage++
		}
		if s.Chunks > 0 {
			ttft = append(ttft, ms(s.TTFT))
		}
		for _, g := range s.ITL {
			itl = append(itl, ms(g))
		}
		if r := s.DecodeRate(); !math.IsNaN(r) {
			rate = append(rate, r)
		}
		if s.Chunks > 0 && (c.SLOTTFT == 0 || s.TTFT <= c.SLOTTFT) && (c.SLOTPOT == 0 || s.TPOT() <= c.SLOTPOT) {
			good++
		}
	}
	st.TTFTp50, st.TTFTp90, st.TTFTp99 = F(Percentile(ttft, 50)), F(Percentile(ttft, 90)), F(Percentile(ttft, 99))
	st.ITLp50, st.ITLp99 = F(Percentile(itl, 50)), F(Percentile(itl, 99))
	st.ReqRate = F(Median(rate))
	if wall > 0 {
		st.AggRate = F(float64(st.CompletionTokens) / wall.Seconds())
		st.Goodput = F(float64(good) / wall.Seconds())
	}
	return st
}

func summaryLine(lv Level) string {
	st := Summarise(lv.Samples, lv.Wall, Config{})
	return fmt.Sprintf("%7.1f tok/s  TTFT p50 %8.1f ms  ITL p50 %7.2f ms  errors %d/%d",
		st.AggRate, st.TTFTp50, st.ITLp50, st.Errors, st.Requests)
}

// Row is one arm at one concurrency over every round, pooled.
type Row struct {
	Arm     string           `json:"arm"`
	Level   int              `json:"concurrency"`
	All     Stats            `json:"all"`
	ByModel map[string]Stats `json:"by_model,omitempty"`
	// Rounds is the per-round aggregate rate, the series the ratios are of.
	Rounds []F `json:"round_tok_s"`
}

// Comparison is one arm against the first, at one concurrency: the median of
// the per-round ratios, gated (RULE 2). A ratio above 1 is the arm faster
// (rates) or slower (latencies); each field says which.
type Comparison struct {
	Arm   string `json:"arm"`
	Base  string `json:"base"`
	Level int    `json:"concurrency"`
	// AggRate is arm/base aggregate tok/s: above 1 the arm is faster.
	AggRate RatioGate `json:"aggregate_tok_s"`
	// TTFT and ITL are arm/base p50: above 1 the arm is slower.
	TTFT RatioGate `json:"ttft_p50"`
	ITL  RatioGate `json:"itl_p50"`
}

// Report is what a run is reduced to.
type Report struct {
	Rows        []Row        `json:"rows"`
	Comparisons []Comparison `json:"comparisons,omitempty"`
}

// Reduce builds the report from a run.
func Reduce(r *Result) Report {
	c := r.Config
	type key struct {
		arm   string
		level int
	}
	pooled := map[key][]Sample{}
	walls := map[key]time.Duration{}
	rounds := map[key]map[int]Level{}
	for _, lv := range r.Levels {
		k := key{lv.Arm, lv.Level}
		for i := range lv.Samples {
			lv.Samples[i].Round = lv.Round
		}
		pooled[k] = append(pooled[k], lv.Samples...)
		walls[k] += lv.Wall
		if rounds[k] == nil {
			rounds[k] = map[int]Level{}
		}
		rounds[k][lv.Round] = lv
	}
	var rep Report
	for _, level := range c.Levels {
		for _, a := range c.Arms {
			k := key{a.Name, level}
			if _, ok := pooled[k]; !ok {
				continue
			}
			row := Row{Arm: a.Name, Level: level, All: Summarise(pooled[k], walls[k], c)}
			if len(c.Models) > 1 {
				row.ByModel = map[string]Stats{}
				for _, m := range c.Models {
					var ms []Sample
					for _, s := range pooled[k] {
						if s.Model == m.Name {
							ms = append(ms, s)
						}
					}
					// Each model's rate is over the whole level's wall: its
					// share of the box's output while every model ran.
					row.ByModel[m.Name] = Summarise(ms, walls[k], c)
				}
			}
			for round := 1; round <= c.Rounds; round++ {
				lv, ok := rounds[k][round]
				if !ok {
					row.Rounds = append(row.Rounds, F(nan()))
					continue
				}
				row.Rounds = append(row.Rounds, Summarise(lv.Samples, lv.Wall, c).AggRate)
			}
			rep.Rows = append(rep.Rows, row)
		}
		if len(c.Arms) < 2 {
			continue
		}
		base := c.Arms[0].Name
		for _, a := range c.Arms[1:] {
			cmp := Comparison{Arm: a.Name, Base: base, Level: level}
			var agg, ttft, itl []float64
			for round := 1; round <= c.Rounds; round++ {
				bl, ok1 := rounds[key{base, level}][round]
				al, ok2 := rounds[key{a.Name, level}][round]
				if !ok1 || !ok2 {
					agg, ttft, itl = append(agg, nan()), append(ttft, nan()), append(itl, nan())
					continue
				}
				bs, as := Summarise(bl.Samples, bl.Wall, c), Summarise(al.Samples, al.Wall, c)
				// A round where either arm failed a request is not a ratio of
				// the two engines' work.
				if bs.Errors > 0 || as.Errors > 0 {
					agg, ttft, itl = append(agg, nan()), append(ttft, nan()), append(itl, nan())
					continue
				}
				agg = append(agg, float64(as.AggRate/bs.AggRate))
				ttft = append(ttft, float64(as.TTFTp50/bs.TTFTp50))
				itl = append(itl, float64(as.ITLp50/bs.ITLp50))
			}
			cmp.AggRate, cmp.TTFT, cmp.ITL = Gate(agg), Gate(ttft), Gate(itl)
			rep.Comparisons = append(rep.Comparisons, cmp)
		}
	}
	return rep
}

// Print writes the human table.
func (rep Report) Print(w io.Writer, c Config) {
	fmt.Fprintf(w, "\n%-12s %4s %6s %6s %9s %9s %9s %8s %8s %9s %10s %8s %7s\n",
		"arm", "c", "reqs", "errs", "TTFT p50", "TTFT p90", "TTFT p99", "ITL p50", "ITL p99",
		"req tok/s", "agg tok/s", "goodput", "at max")
	line := func(name string, level int, st Stats) {
		fmt.Fprintf(w, "%-12s %4d %6d %6d %9.1f %9.1f %9.1f %8.2f %8.2f %9.1f %10.1f %8.2f %6.0f%%\n",
			name, level, st.Requests, st.Errors, st.TTFTp50, st.TTFTp90, st.TTFTp99,
			st.ITLp50, st.ITLp99, st.ReqRate, st.AggRate, st.Goodput,
			100*float64(st.AtMax)/float64(max(st.Requests-st.Errors, 1)))
	}
	for _, r := range rep.Rows {
		line(r.Arm, r.Level, r.All)
		names := make([]string, 0, len(r.ByModel))
		for n := range r.ByModel {
			names = append(names, n)
		}
		sort.Strings(names)
		for _, n := range names {
			line("  "+n, r.Level, r.ByModel[n])
		}
		if r.All.NoUsage > 0 {
			fmt.Fprintf(w, "%-12s      %d stream(s) carried no usage: their token counts are chunk counts\n", "", r.All.NoUsage)
		}
	}
	fmt.Fprintf(w, "\nlatencies in ms; goodput in requests/s within TTFT %v and TPOT %v (0 = unbounded); "+
		"\"at max\" is the share that generated exactly max_tokens=%d\n", c.SLOTTFT, c.SLOTPOT, c.MaxTokens)
	if len(rep.Comparisons) == 0 {
		return
	}
	fmt.Fprintf(w, "\nper-round ratios, median (IQR/median, n); refused rows may not be quoted (RULE 2)\n")
	gate := func(g RatioGate) string {
		s := fmt.Sprintf("%.4f (%.3f, n=%d)", g.Median, g.IQR/g.Median, g.N)
		if g.Refused != "" {
			s += " REFUSED: " + g.Refused
		}
		return s
	}
	for _, cmp := range rep.Comparisons {
		fmt.Fprintf(w, "%s / %s  c=%d\n", cmp.Arm, cmp.Base, cmp.Level)
		fmt.Fprintf(w, "  aggregate tok/s  %s   (>1: %s faster)\n", gate(cmp.AggRate), cmp.Arm)
		fmt.Fprintf(w, "  TTFT p50         %s   (>1: %s slower)\n", gate(cmp.TTFT), cmp.Arm)
		fmt.Fprintf(w, "  ITL p50          %s   (>1: %s slower)\n", gate(cmp.ITL), cmp.Arm)
	}
	if aa := strings.TrimSpace(selfControl(c)); aa != "" {
		fmt.Fprintln(w, aa)
	}
}

// selfControl names an A/A run, whose ratios must read 1.0 inside their IQR
// before the same harness's A/B rows mean anything.
func selfControl(c Config) string {
	if len(c.Arms) < 2 {
		return ""
	}
	for _, a := range c.Arms[1:] {
		if a.URL != c.Arms[0].URL {
			return ""
		}
	}
	return "\nA/A self-control: every arm is one endpoint. Each ratio must read 1.0 inside its IQR; " +
		"if it does not, this harness cannot resolve an A/B on this box."
}

// F is a float that encodes NaN (a statistic with no samples) as JSON null,
// which encoding/json otherwise refuses to write.
type F float64

func (f F) MarshalJSON() ([]byte, error) {
	if math.IsNaN(float64(f)) || math.IsInf(float64(f), 0) {
		return []byte("null"), nil
	}
	return json.Marshal(float64(f))
}

func (f *F) UnmarshalJSON(b []byte) error {
	if string(b) == "null" {
		*f = F(nan())
		return nil
	}
	return json.Unmarshal(b, (*float64)(f))
}
