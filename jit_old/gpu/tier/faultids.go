package tier

// The named deliberate violations of the fault gates (RULE 10), one enum per
// family. A read site compares one small integer, never a string; the names
// below are parsed only where a test arms a fault (hooks_on.go), and their
// meanings are documented beside that table in fault.go. Zero is disarmed.

type pagedFaultID uint8

const (
	pagedNone pagedFaultID = iota
	pagedWindow
	pagedAlias
	pagedSoftwidth
	pagedSession
	pagedPassclip
	pagedWinrelease
	pagedStale
	pagedPassfold
	pagedStreamed
)

type recFaultID uint8

const (
	recNone recFaultID = iota
	recSlot
	recParity
	recResize
	recChain
)

type rowsFaultID uint8

const (
	rowsNone rowsFaultID = iota
	rowsHeadone
	rowsNocap
	rowsSwatab
	rowsNoswa
	rowsNope
	rowsNotemp
	rowsNowin
	rowsRagtab
	rowsStreamcutrows
	rowsEvictselection
	rowsDecodealloc
	rowsPieceouter
)

type roomFaultID uint8

const (
	roomNone roomFaultID = iota
	roomNozeros
	roomNozerosNohost
	roomNarrow
	roomNarrowLie
	roomNoscratch
	roomNoscratchSplit
	roomNoseat
	roomNoseatLie
)

type moeFaultID uint8

const (
	moeNone moeFaultID = iota
	moeFloatsrc
	moeDownsrc
	moeWout
	moePrefillwave
	moePrefillunnamed
	moePrefillalloc
	moeSplitrowshost
)

type peerFaultID uint8

const (
	peerNone peerFaultID = iota
	peerWrong
)
