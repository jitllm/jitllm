package page

import "fmt"

// MinBudget is the floor a derived budget lands on when the device says it has
// almost nothing left. It deliberately holds nothing useful, so a device
// declines blocks in the ordinary way rather than claiming room the card does
// not have.
const MinBudget = 64 << 20

// HostShare is how much of the host's weight budget devices whose memory is the
// host's may hold between them: all of it, less the headroom BudgetFor leaves
// on a card (an eighth, at least 256 MiB, at most 1 GiB and at most half). The
// device and the host are one pool, so a model that fits the host budget
// belongs on the device. Nothing is counted twice: the host loses what the
// device actually holds (tier.GPU.HostReserved), never this ceiling. The headroom is
// what the host side keeps when the device is full -- the pager's frames for
// the blocks the device declined, and its own scratch.
//
// A quarter of the host budget stood here; on a 16 GiB M4 it was 1.92 GiB and
// put the last block and the head of a 2.26 GiB model on the CPU by default.
//
// A per-device budget in the spec (`vulkan:1=8G`) overrides it. An explicit
// -vram (tier.WithBudget) is taken at face value for the pool too, and unified
// devices share that number rather than each getting it.
func HostShare(host, budget, defaultBudget uint64) uint64 {
	if budget > 0 {
		return budget
	}
	if host == 0 {
		// sched.MemBudget returns 0 when it cannot read a limit at all, which
		// is "unknown" and not "unlimited".
		return defaultBudget
	}
	return host - min(max(host/8, 256<<20), MaxHeadroom, host/2)
}

// BudgetFor is what one device may spend on resident weights, and why: free
// memory less headroom, unless the caller asked for a figure: free and total
// are the device's Mem, err its error, which takes defaultBudget.
//
// The headroom is an eighth of free, at least 256 MiB, at most half and at most
// 1 GiB. It covers staging buffers, compiled modules and driver allocations,
// and it is a fraction because `free` is a snapshot of a machine where others
// may allocate next; overshooting VRAM fails rather than degrades. "At most
// half" is for a nearly full card, which then declines blocks per layer. "At
// most 1 GiB" is because an eighth of a big card is a whole block: on a 16 GiB
// card it cost a large mixture two blocks to the host, while the peak beyond
// the charged budget (with the widest batched mixture scratch) was ~0.5 GiB.
func BudgetFor(free, total uint64, err error, ask uint64, src string, defaultBudget uint64) (uint64, string) {
	if ask > 0 {
		return ask, src
	}
	if err != nil {
		return defaultBudget, fmt.Sprintf("the default: %v", err)
	}
	if free == 0 {
		return MinBudget, fmt.Sprintf("the floor: the device reports 0 of %.2f GiB free",
			float64(total)/(1<<30))
	}
	head := min(free/8, MaxHeadroom)
	if head < 256<<20 {
		head = 256 << 20
	}
	if head > free/2 {
		head = free / 2
	}
	limit := free - head
	if limit < MinBudget {
		limit = MinBudget
	}
	return limit, fmt.Sprintf("%.2f GiB free less %.2f GiB headroom",
		float64(free)/(1<<30), float64(head)/(1<<30))
}

// MaxHeadroom caps BudgetFor's headroom; see there.
const MaxHeadroom = 1 << 30
