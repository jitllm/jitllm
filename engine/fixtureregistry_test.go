package engine

import (
	"maps"
	"slices"

	"github.com/jitllm/jitllm/model"
)

type fixtureList = []struct {
	name string
	sel  func(m *model.Model) bool
}

// c6Fixtures is the kit's base fixtures and every family that joins its gates.
func c6Fixtures() fixtureList {
	return slices.Concat(c6BaseFixtures(), denseFixtures(), glm4Fixtures(), yarnFixtures())
}

func c6Violations() map[string][]c6Violation {
	m := c6BaseViolations()
	maps.Copy(m, denseViolations())
	m["synth-glm4"] = glm4Violations()
	maps.Copy(m, yarnViolations())
	return m
}

func c6DevViolations() map[string][]c6Violation {
	m := c6BaseDevViolations()
	maps.Copy(m, denseDevViolations())
	m["synth-glm4"] = glm4DevViolations()
	return m
}

func moeFixtures() fixtureList {
	return slices.Concat(moeBaseFixtures(), moe2Fixtures())
}

func moeViolations() map[string][]c6Violation {
	m := moeBaseViolations()
	maps.Copy(m, moe2Violations())
	return m
}

func moeDevViolations() map[string][]c6Violation {
	m := moeBaseDevViolations()
	maps.Copy(m, moe2DevViolations())
	return m
}

func ssmFixtures() fixtureList {
	return slices.Concat(ssmBaseFixtures(), falconh1Fixtures(), granitehybridFixtures(), lfm2Fixtures(),
		mamba1Fixtures(), nemotronhFixtures())
}

func servedModels() []string {
	return append(servedBaseModels(), ssmServed()...)
}

func fixtureNames(fs fixtureList) []string {
	names := make([]string, len(fs))
	for i, fx := range fs {
		names[i] = fx.name
	}
	return names
}

// ssmNotMamba names the fixtures whose recurrent blocks are not Mamba-2's
// (LFM2's short convolutions, Mamba-1's scan): their family's list is the
// whole list.
func ssmNotMamba(name string) bool {
	return slices.Contains(fixtureNames(lfm2Fixtures()), name) || slices.Contains(fixtureNames(mamba1Fixtures()), name)
}

// ssmFamilyViolations is each family's own violations beside the Mamba-2
// ones; device says the list is for a device session (a family may hold back a
// violation the device declines by name rather than runs).
func ssmFamilyViolations(name string) func(device bool) []c6Violation {
	switch name {
	case "synth-mamba1", "synth-mamba1-ragged":
		return func(bool) []c6Violation { return mamba1Violations() }
	case "synth-falconmamba":
		return func(bool) []c6Violation {
			return append(slices.Clone(mamba1Violations()), mamba1NormViolations()...)
		}
	case "synth-jamba":
		return func(bool) []c6Violation {
			return append(append(slices.Clone(mamba1Violations()), mamba1NormViolations()...), jambaViolations()...)
		}
	case "synth-nemotronhmoe":
		return func(device bool) []c6Violation {
			return append(nemotronhViolations(device), nemotronhMoEViolations()...)
		}
	case "synth-lfm2moe":
		return func(device bool) []c6Violation {
			return append(lfm2Violations(device),
				c6Violation{"no selection bias", noSelectionBias},
				c6Violation{"softmax for the sigmoid router", routerSwap(false)})
		}
	}
	switch {
	case slices.Contains(fixtureNames(falconh1Fixtures()), name):
		return falconh1Violations
	case slices.Contains(fixtureNames(granitehybridFixtures()), name):
		return func(bool) []c6Violation { return granitehybridViolations() }
	case slices.Contains(fixtureNames(lfm2Fixtures()), name):
		return lfm2Violations
	case slices.Contains(fixtureNames(nemotronhFixtures()), name):
		return nemotronhViolations
	}
	return nil
}
