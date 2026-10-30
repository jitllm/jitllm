// lcpplogits teacher-forces llama.cpp over the token ids in a file and writes
// the logits of the prompt's last row and of each step after it, float32 rows.
// Build: cc -O2 -o lcpplogits lcpplogits.c -I$LLAMA/include -I$LLAMA/ggml/include -L$BIN -lllama -Wl,-rpath,'$ORIGIN'
// Usage: lcpplogits model.gguf ngl ids.bin nprompt out.bin
#include <stdio.h>
#include <stdlib.h>
#include "llama.h"

int main(int argc, char **argv) {
	if (argc != 6) {
		fprintf(stderr, "usage: %s model.gguf ngl ids.bin nprompt out.bin\n", argv[0]);
		return 2;
	}
	FILE *f = fopen(argv[3], "rb");
	if (!f) {
		perror(argv[3]);
		return 1;
	}
	fseek(f, 0, SEEK_END);
	int n = (int)(ftell(f) / sizeof(llama_token));
	fseek(f, 0, SEEK_SET);
	llama_token *ids = malloc(n * sizeof(llama_token));
	if (fread(ids, sizeof(llama_token), n, f) != (size_t)n) {
		fprintf(stderr, "%s: short read\n", argv[3]);
		return 1;
	}
	fclose(f);
	int np = atoi(argv[4]);
	if (np < 1 || np > n) {
		fprintf(stderr, "nprompt %d outside 1..%d\n", np, n);
		return 2;
	}
	llama_backend_init();
	struct llama_model_params mp = llama_model_default_params();
	mp.n_gpu_layers = atoi(argv[2]);
	struct llama_model *model = llama_model_load_from_file(argv[1], mp);
	if (!model) {
		fprintf(stderr, "%s: load failed\n", argv[1]);
		return 1;
	}
	struct llama_context_params cp = llama_context_default_params();
	cp.n_ctx = n + 8;
	cp.n_batch = n + 8;
	cp.n_ubatch = n + 8;
	struct llama_context *ctx = llama_init_from_model(model, cp);
	if (!ctx) {
		fprintf(stderr, "context failed\n");
		return 1;
	}
	int nv = llama_vocab_n_tokens(llama_model_get_vocab(model));
	FILE *out = fopen(argv[5], "wb");
	if (!out) {
		perror(argv[5]);
		return 1;
	}
	if (llama_decode(ctx, llama_batch_get_one(ids, np)) != 0) {
		fprintf(stderr, "prompt decode failed\n");
		return 1;
	}
	fwrite(llama_get_logits_ith(ctx, -1), sizeof(float), nv, out);
	for (int i = np; i < n; i++) {
		if (llama_decode(ctx, llama_batch_get_one(ids + i, 1)) != 0) {
			fprintf(stderr, "step %d decode failed\n", i);
			return 1;
		}
		fwrite(llama_get_logits_ith(ctx, -1), sizeof(float), nv, out);
	}
	fclose(out);
	llama_free(ctx);
	llama_model_free(model);
	llama_backend_free();
	return 0;
}
