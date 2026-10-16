package tier

import (
	"fmt"
	"sort"
)

// RecipeChoice is one site's decision on one device, with the reason: which
// recipe the blocks with that shape key run, and why that one -- forced by the
// caller, read from the verdict cache, timed here, or the device's default
// because no challenger resolved faster (docs/design/composer.md). It is what
// a run manifest records, so a row says which variant it measured.
type RecipeChoice struct {
	Device string `json:"device"`
	Site   string `json:"site"`            // "ffn", "actin-qkv", "actin-o", "actin-down"
	Key    string `json:"key"`             // the shape key; "*" where a forced choice covers every key
	Recipe string `json:"recipe"`          // the recipe the key runs
	Source string `json:"source"`          // "forced", "cache", "timed", "untuned"
	Why    string `json:"why"`             // the reason in words
	Blocks int    `json:"blocks"`          // placed blocks with this key at the decision
	Trial  []Arm  `json:"trial,omitempty"` // the timed challengers, where Source is "timed"
}

// Arm is one challenger's trial: its median per-round ratio to the
// incumbent, and whether the trial resolved (IQR/median within the gate).
type Arm struct {
	Recipe   string  `json:"recipe"`
	Ratio    float64 `json:"ratio"`
	Resolved bool    `json:"resolved"`
}

// explainSet records site's decision for key, replacing an earlier one (a
// retune decides the same keys again). Called holding g.mu.
func (g *devTier) explainSet(c RecipeChoice) {
	if g.explain == nil {
		g.explain = map[string]RecipeChoice{}
	}
	c.Device = label(g.dev, g.ord)
	g.explain[c.Site+"\x00"+c.Key] = c
}

// explainDrop forgets every decision of site, before a tune records its own.
func (g *devTier) explainDrop(site string) {
	for k, c := range g.explain {
		if c.Site == site || (site == "actin" && len(c.Site) > 6 && c.Site[:6] == "actin-") {
			delete(g.explain, k)
		}
	}
}

// Explain is every device's recipe decisions in force, by device, site and
// key: the plan explanation a run manifest carries.
func (g *GPU) Explain() []RecipeChoice {
	var out []RecipeChoice
	for _, d := range g.devs {
		d.mu.Lock()
		for _, c := range d.explain {
			out = append(out, c)
		}
		d.mu.Unlock()
	}
	sort.Slice(out, func(i, j int) bool {
		a, b := out[i], out[j]
		if a.Device != b.Device {
			return a.Device < b.Device
		}
		if a.Site != b.Site {
			return a.Site < b.Site
		}
		return a.Key < b.Key
	})
	return out
}

// String is the choice on one line.
func (c RecipeChoice) String() string {
	s := fmt.Sprintf("%s %s %s: %s (%s: %s; %d blocks)", c.Device, c.Site, c.Key, c.Recipe, c.Source, c.Why, c.Blocks)
	for _, a := range c.Trial {
		s += fmt.Sprintf(" [%s %.4f resolved %v]", a.Recipe, a.Ratio, a.Resolved)
	}
	return s
}
