package tier

// addNormBroken makes a folded add sum x2 with itself instead of with the
// FFN's output: the violation the fold's gate must see (BreakAddNormForTest,
// jitllmtest builds only).
var addNormBroken bool
