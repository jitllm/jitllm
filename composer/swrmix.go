package composer

import (
	"fmt"
	"math"
	"math/bits"
)

// A channel's bit in a layout, libavutil's AV_CHAN_* and a WAV's
// dwChannelMask alike.
const (
	chFL = 1 << iota
	chFR
	chFC
	chLFE
	chBL
	chBR
	chFLC
	chFRC
	chBC
	chSL
	chSR
	chTC
	chTFL
	chTFC
	chTFR
	chTBL
	chTBC
	chTBR
)

// chNamed is the channels swr's matrix names (rematrix.c's
// NUM_NAMED_CHANNELS); a channel past them mixes into no other.
const chNamed = 18

// defaultLayout is av_channel_layout_default: the first layout of
// libavutil's channel_layout_map with that many channels, 0 for a count no
// layout has.
func defaultLayout(channels int) uint64 {
	const (
		surround = chFL | chFR | chFC
		back51   = surround | chBL | chBR | chLFE
		side51   = surround | chSL | chSR | chLFE
		octagon  = surround | chSL | chSR | chBL | chBC | chBR
		wide     = 1<<31 | 1<<32
	)
	switch channels {
	case 1:
		return chFC
	case 2:
		return chFL | chFR
	case 3:
		return chFL | chFR | chLFE
	case 4:
		return surround | chBC
	case 5:
		return surround | chBL | chBR
	case 6:
		return back51
	case 7:
		return side51 | chBC
	case 8:
		return side51 | chBL | chBR
	case 10:
		return back51 | chTFL | chTFR | chTBL | chTBR
	case 12:
		return side51 | chBL | chBR | chTFL | chTFR | chTBL | chTBR
	case 16:
		return octagon | wide | chTBL | chTBR | chTBC | chTFC | chTFL | chTFR
	}
	return 0
}

// MonoMix is each channel's weight in the mono mix swr makes of a clip of
// channels channels laid out by mask (a WAV's dwChannelMask; 0 for none, the
// count's default layout): swr_build_matrix2 for a mono output at swr's
// default levels -- centre and surround mixed at -3 dB, the LFE not at all
// -- and, the output being float, unnormalised. A layout swr refuses
// (sane_layout) or a count with no default is refused as ffmpeg refuses it.
func MonoMix(channels int, mask uint64) ([]float32, error) {
	if channels < 1 || channels > 64 {
		return nil, fmt.Errorf("composer: a clip of %d channels", channels)
	}
	if mask == 0 || bits.OnesCount64(mask) != channels {
		mask = defaultLayout(channels)
		if mask == 0 {
			return nil, fmt.Errorf("composer: a clip of %d channels has no layout to mix it to mono (ffmpeg's rematrix refuses it)", channels)
		}
	}
	out := make([]float32, channels)
	// One channel without a centre is treated as mono (clean_layout).
	if channels == 1 {
		out[0] = 1
		return out, nil
	}
	pair := func(l, r uint64) bool { return mask&(l|r) == 0 || mask&(l|r) == l|r }
	if mask&(chFL|chFR|chFC) == 0 || !pair(chFL, chFR) || !pair(chSL, chSR) || !pair(chBL, chBR) ||
		!pair(chFLC, chFRC) || !pair(chTFL, chTFR) {
		return nil, fmt.Errorf("composer: swr does not mix a clip laid out as %#x", mask)
	}
	const (
		sqrt1_2 = math.Sqrt2 / 2
		clev    = sqrt1_2
		slev    = sqrt1_2
	)
	var m [chNamed]float64
	if mask&chFC != 0 {
		m[2] = 1
	}
	add := func(bit uint64, w float64) {
		m[bits.TrailingZeros64(bit)] += w
	}
	if mask&(chFL|chFR) != 0 {
		add(chFL, sqrt1_2)
		add(chFR, sqrt1_2)
		if mask&chFC != 0 {
			m[2] = clev * math.Sqrt2
		}
	}
	if mask&chBC != 0 {
		add(chBC, slev*sqrt1_2)
	}
	if mask&chBL != 0 {
		add(chBL, slev*sqrt1_2)
		add(chBR, slev*sqrt1_2)
	}
	if mask&chSL != 0 {
		add(chSL, slev*sqrt1_2)
		add(chSR, slev*sqrt1_2)
	}
	if mask&chFLC != 0 {
		add(chFLC, sqrt1_2)
		add(chFRC, sqrt1_2)
	}
	if mask&chTFL != 0 {
		add(chTFL, sqrt1_2)
		add(chTFR, sqrt1_2)
	}
	i := 0
	for ch := range 64 {
		if mask&(1<<ch) == 0 {
			continue
		}
		if ch < chNamed {
			out[i] = float32(m[ch])
		}
		i++
	}
	return out, nil
}
