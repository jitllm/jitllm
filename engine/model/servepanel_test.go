package model

import "slices"

// servePanel is the serving gate's prompts: crossingPrompt and seven of
// other kinds, so an arm is judged across prompts and not on one.
func servePanel() []string {
	return []string{
		crossingPrompt,
		"In 1854 the physician John Snow mapped the cholera deaths around Broad Street in London and noticed that they " +
			"clustered near a single public water pump. Most of his colleagues still believed that the disease spread through " +
			"bad air, a miasma rising from rotting matter, and they dismissed his map as a coincidence. Snow persuaded the local " +
			"board to remove the pump's handle, and the outbreak, already waning, ended soon after. Years later, the engineer " +
			"Joseph Bazalgette built the sewers that finally separated the city's drinking water from its waste, and the city",
		"func merge(a, b []int) []int {\n\tout := make([]int, 0, len(a)+len(b))\n\tfor len(a) > 0 && len(b) > 0 {\n" +
			"\t\tif a[0] <= b[0] {\n\t\t\tout = append(out, a[0])\n\t\t\ta = a[1:]\n\t\t} else {\n\t\t\tout = append(out, b[0])\n" +
			"\t\t\tb = b[1:]\n\t\t}\n\t}\n\tout = append(out, a...)\n\treturn append(out, b...)\n}\n\n// mergeSort sorts xs by " +
			"splitting it in half, sorting each half and merging the two.\nfunc mergeSort(xs []int) []int {\n\tif len(xs) < 2 {\n" +
			"\t\treturn xs\n\t}\n\tmid := len(xs) / 2\n\treturn",
		"Question: A train leaves the station at 9:15 and travels at 80 kilometres per hour. A second train leaves the same " +
			"station at 10:00 on a parallel track and travels at 100 kilometres per hour. At what time does the second train " +
			"catch up with the first, and how far from the station are they when it does?\nAnswer: Let t be the hours after " +
			"10:00. By then the first train has travelled 0.75 hours at 80 km/h, which is 60 kilometres. The gap closes at 20 " +
			"kilometres per hour, so",
		"Dear Ms. Alvarez,\n\nThank you for your letter of the twelfth regarding the delayed shipment of ceramic tiles to your " +
			"store in Valencia. I have spoken with our warehouse manager, who confirmed that the pallets were held at customs " +
			"because the invoice listed the wrong tariff code. We have corrected the paperwork and the goods were released this " +
			"morning. To make up for the inconvenience, we would like to offer you a discount of ten percent on your next order, " +
			"and we will cover the cost of express delivery. Please let me know",
		"The mitochondrion is often called the powerhouse of the cell, but that phrase hides most of what it does. Besides " +
			"producing ATP through oxidative phosphorylation, it buffers calcium, takes part in programmed cell death, builds " +
			"iron-sulfur clusters and heme, and carries its own small circular genome, inherited almost always from the mother. " +
			"Its inner membrane folds into cristae, which multiply the surface on which the electron transport chain sits. " +
			"Because the organelle descends from an ancient bacterium, its ribosomes resemble",
		"1. Preheat the oven to 200 degrees and line a baking tray with parchment.\n2. Cut the squash into cubes, toss them " +
			"with olive oil, salt and a pinch of chili, and roast for thirty minutes, turning once.\n3. Meanwhile, toast the " +
			"walnuts in a dry pan until they smell nutty, then chop them roughly.\n4. Cook the pasta in plenty of salted water " +
			"until it is just short of tender, and keep a cup of the cooking water.\n5. Melt the butter with the sage leaves " +
			"until the butter turns brown and the leaves crisp.\n6.",
		"She had not meant to stay in the lighthouse for the winter. The keeper had asked her to watch the lamp for a week " +
			"while he visited his sister on the mainland, and then the storms came and the ferry stopped running. Every evening " +
			"she climbed the hundred and twelve steps, trimmed the wick, wound the clockwork that turned the lens, and wrote a " +
			"line in the log. By January she had filled four pages with the weather, the ships she saw, and a gull that came to " +
			"the window each morning. On the first clear day of February, she saw",
	}
}

// panelSteps is the decode each prompt after the first runs.
const panelSteps = 48

// panelBand is the serving gate's statistic over servePanel: the median
// across prompts of each prompt's median per-step ratio, and the worst step.
type panelBand struct {
	meds  []float64
	worst float64
}

func (b *panelBand) add(ratios []float64, worst float64) {
	slices.Sort(ratios)
	b.meds = append(b.meds, ratios[len(ratios)/2])
	b.worst = max(b.worst, worst)
}

func (b *panelBand) median() float64 {
	m := slices.Sorted(slices.Values(b.meds))
	return m[len(m)/2]
}

func (b *panelBand) inside(wide float64) bool {
	return b.median() <= bandRatio && b.worst <= bandWorst*wide
}
