//go:build jitllmfault || jitllmtest

package tier

var pagedFaultNames = map[string]pagedFaultID{
	"window":     pagedWindow,
	"alias":      pagedAlias,
	"softwidth":  pagedSoftwidth,
	"session":    pagedSession,
	"passclip":   pagedPassclip,
	"winrelease": pagedWinrelease,
	"stale":      pagedStale,
	"passfold":   pagedPassfold,
	"stream":     pagedStreamed,
}

var recFaultNames = map[string]recFaultID{
	"slot":   recSlot,
	"parity": recParity,
	"resize": recResize,
	"chain":  recChain,
}

var rowsFaultNames = map[string]rowsFaultID{
	"headone":        rowsHeadone,
	"nocap":          rowsNocap,
	"swatab":         rowsSwatab,
	"noswa":          rowsNoswa,
	"nope":           rowsNope,
	"notemp":         rowsNotemp,
	"nowin":          rowsNowin,
	"ragtab":         rowsRagtab,
	"streamcutrows":  rowsStreamcutrows,
	"evictselection": rowsEvictselection,
	"decodealloc":    rowsDecodealloc,
	"pieceouter":     rowsPieceouter,
}

var roomFaultNames = map[string]roomFaultID{
	"nozeros":         roomNozeros,
	"nozeros-nohost":  roomNozerosNohost,
	"narrow":          roomNarrow,
	"narrow-lie":      roomNarrowLie,
	"noscratch":       roomNoscratch,
	"noscratch-split": roomNoscratchSplit,
	"noseat":          roomNoseat,
	"noseat-lie":      roomNoseatLie,
}

var moeFaultNames = map[string]moeFaultID{
	"floatsrc":       moeFloatsrc,
	"downsrc":        moeDownsrc,
	"wout":           moeWout,
	"prefillwave":    moePrefillwave,
	"prefillunnamed": moePrefillunnamed,
	"prefillalloc":   moePrefillalloc,
	"splitrowshost":  moeSplitrowshost,
}

var peerFaultNames = map[string]peerFaultID{
	"wrong": peerWrong,
}
