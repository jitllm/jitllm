package hardware

import (
	"fmt"
	"slices"
	"strings"

	"github.com/jitllm/jitllm/common/session"
)

// The probe as plain lines of text, for every front end that shows it.

// ProbeSummary is the status line a finished probe leaves behind.
func ProbeSummary(r Report) string {
	if r.Err != "" {
		return "hardware probe: " + r.Err
	}
	switch len(r.GPUs) {
	case 0:
		return fmt.Sprintf("%s, %d decode core(s), no GPU -- that is a CPU machine, not an error",
			r.CPU, len(r.Decode))
	case 1:
		return fmt.Sprintf("%s, %d decode core(s), 1 device: %s", r.CPU, len(r.Decode), r.GPUs[0].Name)
	default:
		return fmt.Sprintf("%s, %d decode core(s), %d devices", r.CPU, len(r.Decode), len(r.GPUs))
	}
}

// ShortCPU drops a processor name's brand marks and generation, which the
// card has no room for: "12th Gen Intel(R) Core(TM) i7-1260P" is "i7-1260P".
func ShortCPU(name string) string {
	for _, cut := range []string{"(R)", "(TM)", " CPU", "Intel", "AMD", "Core", "Processor"} {
		name = strings.ReplaceAll(name, cut, "")
	}
	f := strings.Fields(name)
	// "12th Gen" is the generation; the model number already says it.
	if len(f) > 2 && f[1] == "Gen" {
		f = f[2:]
	}
	if i := slices.IndexFunc(f, func(s string) bool { return strings.HasPrefix(s, "@") }); i >= 0 {
		f = f[:i]
	}
	return strings.Join(f, " ")
}

// DeviceStat is a device's API, slots and memory, as one short line.
func DeviceStat(g GPU) string {
	mem := g.MemNote
	if mem == "" {
		mem = fmt.Sprintf("%s of %s free", session.Bytes(g.Free), session.Bytes(g.Total))
	}
	s := strings.ToUpper(g.API) + ", " + mem
	if g.Slots > 0 {
		s += fmt.Sprintf(", %d slots", g.Slots)
	}
	return s
}

func CoreLine(r Report) string {
	if !r.Probed {
		return "reading..."
	}
	s := fmt.Sprintf("%d logical, %d P-core(s)", r.Logical, len(r.PCores))
	if len(r.ECores) > 0 {
		s += fmt.Sprintf(", %d E-core(s)", len(r.ECores))
	}
	if t := len(r.SMT) / max(1, len(r.PCores)); t > 1 {
		s += fmt.Sprintf(", SMT %d per P-core", t)
	} else {
		s += ", no SMT"
	}
	return s
}

func DecodeLine(r Report) string {
	if !r.Probed {
		return "reading..."
	}
	return fmt.Sprintf("%d core(s) %s", len(r.Decode), Ints(r.Decode))
}

func Ints(v []int) string {
	if len(v) == 0 {
		return "[]"
	}
	parts := make([]string, 0, len(v))
	for _, n := range v {
		parts = append(parts, fmt.Sprint(n))
	}
	return "[" + strings.Join(parts, " ") + "]"
}

// DeviceLine is one opened device, as one line.
func DeviceLine(g GPU) string {
	mem := g.MemNote
	if mem == "" {
		mem = fmt.Sprintf("%s of %s free", session.Bytes(g.Free), session.Bytes(g.Total))
	}
	slots := "slots unreported"
	if g.Slots > 0 {
		slots = fmt.Sprintf("%d slot(s)", g.Slots)
	}
	return fmt.Sprintf("%-6s %s   %s, %s", g.API, g.Name, slots, mem)
}

// DeviceNote is what else is true of a device, or "" when nothing is.
func DeviceNote(g GPU) string {
	var parts []string
	if g.Unified {
		parts = append(parts, "shares system memory: it uses the host budget, not memory of its own")
	}
	if g.ImportAlign > 0 {
		if g.Unified {
			// A page-in is then a descriptor update, not a copy.
			parts = append(parts, "loads weights straight from system memory")
		} else {
			parts = append(parts, "can read system memory directly, over the bus")
		}
	}
	return strings.Join(parts, "; ")
}

// VulkanLine is one entry of the Vulkan enumeration, with the index that names
// it in a device spec.
func VulkanLine(v VulkanDev) string {
	s := fmt.Sprintf("vulkan:%d  %s  (%s)", v.Index, v.Name, v.Type)
	switch {
	case !v.Compute:
		s += "  -- no compute queue: cannot run models"
	case v.Software:
		s += "  -- software renderer: used only if named"
	case v.Unified:
		s += "  -- shares system memory"
	}
	if v.Compute && !v.Promises32 {
		s += fmt.Sprintf("  subgroups %d..%d, no 32-lane guarantee", v.Subgroups[0], v.Subgroups[1])
	}
	return s
}

// ReportText is the whole probe as plain text, for the clipboard -- the page
// you send someone when their rate is wrong.
func ReportText(r Report) string {
	if !r.Probed {
		return "jitllm: the machine has not been probed yet"
	}
	var b strings.Builder
	fmt.Fprintf(&b, "host      %s/%s   %s\n", r.GOOS, r.GOARCH, r.GoVersion)
	fmt.Fprintf(&b, "cpu       %s\n", r.CPU)
	fmt.Fprintf(&b, "          %s\n", CoreLine(r))
	fmt.Fprintf(&b, "          decode %s\n", DecodeLine(r))
	if r.RAMTotal > 0 {
		fmt.Fprintf(&b, "memory    %s total\n", session.Bytes(r.RAMTotal))
	}
	fmt.Fprintf(&b, "          budget %s   limit %s   collector cap %s\n",
		session.Bytes(r.MemBudget), session.Bytes(r.MemLimit), session.Bytes(r.GCBudgetCap))
	fmt.Fprintf(&b, "composer  %s\n", r.Form)
	fmt.Fprintf(&b, "          isa %s (%s)\n", r.ISA, r.Arch)
	if len(r.GPUs) == 0 {
		fmt.Fprintf(&b, "gpu       none opened here\n")
	}
	for _, g := range r.GPUs {
		fmt.Fprintf(&b, "gpu       %s\n", DeviceLine(g))
		if n := DeviceNote(g); n != "" {
			fmt.Fprintf(&b, "          %s\n", n)
		}
	}
	if len(r.Vulkan) > 1 {
		for _, v := range r.Vulkan {
			fmt.Fprintf(&b, "          %s\n", VulkanLine(v))
		}
	}
	if r.Err != "" {
		fmt.Fprintf(&b, "error     %s\n", r.Err)
	}
	return b.String()
}

// ShortGPU drops a GPU name's vendor and form factor, which the card has no
// room for: "NVIDIA GeForce RTX 4060 Laptop GPU" is "RTX 4060".
func ShortGPU(name string) string {
	for _, cut := range []string{"NVIDIA ", "GeForce ", "AMD ", "Radeon ", "Intel(R) ", " Laptop GPU", " GPU"} {
		name = strings.ReplaceAll(name, cut, "")
	}
	return strings.TrimSpace(name)
}
