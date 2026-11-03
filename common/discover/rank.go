package discover

import (
	"fmt"
	"math"
	"sort"
	"strconv"
	"strings"

	"github.com/jitllm/jitllm/common/catalog"
	"github.com/jitllm/jitllm/common/session"
	"github.com/jitllm/jitllm/convert/library"
)

// BalanceSteps() are the preference stops, smallest model to largest.
func BalanceSteps() []string {
	return []string{"Fastest", "Faster", "Balanced", "Smarter", "Smartest"}
}

// Params is a model's quoted size as total and active parameters: "8B" is
// 8e9 of each, "30B-A3B" is 30e9 of which 3e9 run per token, "360M" is 3.6e8.
func Params(s string) (total, active float64) {
	parse := func(t string) float64 {
		t = strings.TrimSpace(strings.ToUpper(t))
		mult := 1.0
		switch {
		case strings.HasSuffix(t, "B"):
			mult, t = 1e9, strings.TrimSuffix(t, "B")
		case strings.HasSuffix(t, "M"):
			mult, t = 1e6, strings.TrimSuffix(t, "M")
		}
		v, err := strconv.ParseFloat(t, 64)
		if err != nil {
			return 0
		}
		return v * mult
	}
	parts := strings.SplitN(s, "-A", 2)
	total = parse(parts[0])
	active = total
	if len(parts) == 2 {
		active = parse(parts[1])
	}
	return total, active
}

// Rank orders the library for this machine: models that fit come first,
// nearest the preference -- 0 the smallest that fits, len(BalanceSteps())-1 the
// largest -- and models larger than memory follow, smallest first, because
// they run but read their weights from disk.
func Rank(models []library.Model, mr session.MachineReport, pref int) []int {
	var fit, big []int
	for i, m := range models {
		if EntryFits(catalog.Entry{Size: m.Bytes}, mr) == "paged" {
			big = append(big, i)
		} else {
			fit = append(fit, i)
		}
	}
	size := func(i int) float64 { t, _ := Params(models[i].Params); return t }
	sort.SliceStable(fit, func(a, b int) bool { return size(fit[a]) < size(fit[b]) })
	sort.SliceStable(big, func(a, b int) bool { return size(big[a]) < size(big[b]) })
	if len(fit) > 0 {
		target := float64(pref) / float64(len(BalanceSteps())-1) * float64(len(fit)-1)
		pos := make(map[int]float64, len(fit))
		for k, i := range fit {
			pos[i] = math.Abs(float64(k) - target)
		}
		sort.SliceStable(fit, func(a, b int) bool {
			if pos[fit[a]] != pos[fit[b]] {
				return pos[fit[a]] < pos[fit[b]]
			}
			// A tie goes to the larger model when leaning smart, else the smaller.
			if pref*2 >= len(BalanceSteps())-1 {
				return size(fit[a]) > size(fit[b])
			}
			return size(fit[a]) < size(fit[b])
		})
	}
	return append(fit, big...)
}

// DecodeEfficiency is the share of the measured read bandwidth a decode
// token reaches: the engine decodes close to the wall.
const DecodeEfficiency = 0.8

// Speed estimates decode from the measured bandwidth: a token reads every
// active weight once. A model on the GPU reads faster than the host's wall,
// so its figure is a floor; one larger than memory reads from disk instead.
func Speed(m library.Model, mr session.MachineReport) string {
	if mr.MemWall <= 0 || m.Bytes <= 0 {
		return ""
	}
	total, active := Params(m.Params)
	bytes := float64(m.Bytes)
	if total > 0 {
		bytes *= active / total
	}
	tps := DecodeEfficiency * mr.MemWall / bytes
	switch EntryFits(catalog.Entry{Size: m.Bytes}, mr) {
	case "GPU":
		return fmt.Sprintf("over %.0f tok/s", tps)
	case "RAM":
		return fmt.Sprintf("about %.0f tok/s", tps)
	case "paged":
		return "slow, reads from disk"
	}
	return ""
}

// FitWords is where a model would run, as the row says it.
func FitWords(m library.Model, mr session.MachineReport) string {
	switch EntryFits(catalog.Entry{Size: m.Bytes}, mr) {
	case "GPU":
		return "Fits on the GPU"
	case "RAM":
		return "Fits in memory"
	case "paged":
		return "Larger than memory"
	}
	return ""
}

// FitWordsOf is EntryFits as a row says it.
func FitWordsOf(fit string) string {
	switch fit {
	case "GPU":
		return "Fits on the GPU"
	case "RAM":
		return "Fits in memory"
	case "paged":
		return "Pages from disk"
	}
	return ""
}

// EntryFits says where a model would live on this machine: "GPU" when a
// discrete card holds it with room for the attention cache, "RAM" when the
// engine's memory budget does, "paged" when neither does (it still runs,
// slowly), and "--" until the machine has been probed. A GPU that shares
// system memory is not counted. The size is the file's, which for a GGUF is
// within a few percent of its container.
func EntryFits(e catalog.Entry, mr session.MachineReport) string {
	if !mr.Probed || e.Size <= 0 {
		return "--"
	}
	need := uint64(e.Size)
	for _, g := range mr.GPUs {
		if !g.Unified && g.Mem >= need+need/8 {
			return "GPU"
		}
	}
	if mr.MemBudget >= need {
		return "RAM"
	}
	return "paged"
}
