package convert

import _ "embed"

// kimiK3ChatTemplate is Kimi K3's chat as its release renders it --
// encoding_k3.build_chat_segments, which tokenization_kimi.py's
// apply_chat_template calls with thinking on and thinking_effort "max" --
// transcribed for system, user and assistant turns and the generation prompt.
// The release ships that renderer as Python and no Jinja template, so a
// conversion finds none (hfTemplates) and the class states this one; a tool
// turn raises rather than rendering untranscribed. Held to the renderer's own
// text by TestKimiK3ChatTemplateMatchesItsRenderer.
//
//go:embed kimik3_chat.jinja
var kimiK3ChatTemplate string
