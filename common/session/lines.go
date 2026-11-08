package session

import (
	"fmt"
	"strings"
)

// The one-line readouts of a session, as plain text. Both front ends print
// them, so a figure reads the same in the window and the terminal.

// SessionRateLine renders how fast the last reply was read in and written.
//
// The percentage appears only against a measured wall: memWall is 0 until the
// hardware probe has measured it, and a share of a spec-sheet figure would
// look checkable and not be.
func SessionRateLine(promptTokS, decodeTokS float64, bytesPerTok uint64, memWall float64) string {
	var b strings.Builder
	b.WriteString("prompt ")
	b.WriteString(Rate(promptTokS))
	b.WriteString(" · reply ")
	b.WriteString(Rate(decodeTokS))
	if decodeTokS <= 0 || bytesPerTok == 0 {
		return b.String()
	}
	bps := BytesPerTokenRate(bytesPerTok, decodeTokS)
	b.WriteString(" · ")
	b.WriteString(GBs(bps))
	if memWall > 0 {
		b.WriteString(", ")
		b.WriteString(Percent(bps / memWall))
		b.WriteString(" of ")
		b.WriteString(GBs(memWall))
	}
	return b.String()
}

// SessionPagerLine says a model is paging, or nothing.
func SessionPagerLine(p PagerStat) string {
	// The last turn's evictions, not the running total, which only grows.
	if p.TurnOuts == 0 {
		return ""
	}
	return fmt.Sprintf("paging from disk: %d layer(s) fit in memory, the rest is read as needed (%s so far)",
		p.Frames, Bytes(p.BytesRead))
}

// SessionFooter is the line under the composer: what the last reply cost and
// how much of the context window the conversation holds.
func SessionFooter(promptTokS, decodeTokS float64, bytesPerTok uint64, memWall float64, a Allocation, p PagerStat) string {
	parts := []string{SessionRateLine(promptTokS, decodeTokS, bytesPerTok, memWall)}
	if a.MaxSeq > 0 {
		parts = append(parts, fmt.Sprintf("context %s / %s", Thousands(a.Pos), Thousands(a.MaxSeq)))
	}
	if pg := SessionPagerLine(p); pg != "" {
		parts = append(parts, pg)
	}
	return strings.Join(parts, " · ")
}

// TranscriptText is the whole conversation as plain text.
func TranscriptText(turns []Turn) string {
	var b strings.Builder
	for _, t := range turns {
		who := "you"
		if t.Role == RoleAssistant {
			who = "assistant"
			if t.Model != "" {
				who = t.Model
			}
		}
		fmt.Fprintf(&b, "%s:\n%s\n\n", who, t.Text)
	}
	return b.String()
}

// HostLine is the host's share: page bytes against the page budget, with the
// dense region (embedding and output projection, which never page) named
// separately so a model that fits does not look over its cap.
func HostLine(a Allocation) string {
	if a.NBlocks == 0 {
		return "no model loaded"
	}
	var b strings.Builder
	// Runs and holds are different counts: a device block's page can be held
	// here too, and a host block can be paged out.
	fmt.Fprintf(&b, "runs %d block(s), holds %s of pages", a.CPUBlocks, Bytes(a.HostUsed))
	if a.HostBudget > 0 {
		fmt.Fprintf(&b, " of a %s budget", Bytes(a.HostBudget))
	} else {
		b.WriteString(" (no cap)")
	}
	if a.Dense > 0 {
		fmt.Fprintf(&b, "  + %s dense", Bytes(a.Dense))
	}
	return b.String()
}

// DeviceUseLine is one device's share. It names the pool, since two devices
// on one heap report the same bytes.
func DeviceUseLine(d DeviceUse) string {
	var b strings.Builder
	// Not the name: the row's label carries it, and the caption repeated it.
	fmt.Fprintf(&b, "%d block(s), %s", d.Blocks, Bytes(d.Used))
	if d.Limit > 0 {
		fmt.Fprintf(&b, " of %s", Bytes(d.Limit))
	}
	if d.KV > 0 {
		fmt.Fprintf(&b, "  + %s kv", Bytes(d.KV))
	}
	if d.Shared {
		// An integrated GPU's memory is the host's: subtract, never add.
		b.WriteString("  (host memory, not a second pool)")
	}
	return b.String()
}

// PlacementLine counts where the blocks of the loaded model went. A byte this
// package does not know counts as host, so the card is never over-reported.
func PlacementLine(b []byte) string {
	if len(b) == 0 {
		return "nothing loaded yet -- placement appears here after a load"
	}
	var dev, paged, hostN int
	for _, p := range b {
		switch p := Placement(p); {
		case p.Device() >= 0:
			dev++
		case !p.Resident():
			paged++
		default:
			hostN++
		}
	}
	return fmt.Sprintf("%d of %d block(s) on a device, %d on the host, %d paged out", dev, len(b), hostN, paged)
}

// PagerLine is the container pager's counters.
func PagerLine(p PagerStat) string {
	if p.Frames == 0 && p.BytesRead == 0 {
		return "no page-ins: the budget holds every block"
	}
	return fmt.Sprintf("%d page(s) holding %s, %d in / %d out, %s read over %d request(s)",
		p.Frames, Bytes(p.Resident), p.PageIns, p.PageOuts, Bytes(p.BytesRead), p.Reads)
}
