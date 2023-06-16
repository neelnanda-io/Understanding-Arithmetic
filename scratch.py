# %%
import os
os.environ["TRANSFORMERS_CACHE"] = "/workspace/cache/"
# %%
from neel.imports import *
from neel_plotly import *

# %%
SEED = 42
torch.set_grad_enabled(False)
# %%
test_model = HookedTransformer.from_pretrained("solu-1l")
# %%
model: HookedTransformer = HookedTransformer.from_pretrained("gpt-j", device="cpu")
model: HookedTransformer = model.to(torch.bfloat16).to("cuda")
# %%
n_layers = model.cfg.n_layers
d_model = model.cfg.d_model
n_heads = model.cfg.n_heads
d_head = model.cfg.d_head
d_mlp = model.cfg.d_mlp
d_vocab = model.cfg.d_vocab

# %%
print(model.to_str_tokens("23481762358723658732568235"))
print(model.to_str_tokens("123+456=579"))
print(model.to_str_tokens("123+456=579\n913+72=985\n218+276=494"))

utils.test_prompt("123+456=579\n913+72=985\n218+276=", "494", model, prepend_space_to_answer=False)
# %%
num_tokens_per_digit = [len(model.to_str_tokens(str(i), prepend_bos=False)) for i in range(1000)]
line(num_tokens_per_digit, title="Number of tokens per digit")
# %%
single_token_numbers = torch.arange(520)
number_token_index = torch.tensor([model.to_single_token(str(i)) for i in range(520)])
line(number_token_index, title="Token index for numbers")

# %%
number_embeddings = model.W_E[number_token_index[100:]].float()
i_am_a_number = number_embeddings.mean(0)
number_embeddings_centered = number_embeddings - i_am_a_number
U, S, Vh = number_embeddings_centered.svd()
line(S, title="Singular values of W_E")
line(U[:, :20].T, title="First 20 rows of U")
line((U[:, :20].T) * S[:20, None], title="First 20 rows of U (Weighted_by_S)")
# %%
by_digit = einops.rearrange(number_embeddings_centered, "(other ten) d_model -> other ten d_model", ten=10)
U2, S2, Vh2 = by_digit.mean(0).svd()
line(S2)
line(U2.T)
line(U2.T @ S2[:, None])
# %%
def fourier_basis(n):
    terms = []
    labels = []
    terms.append(torch.ones(n))
    labels.append("const")
    for i in range(1, n//2):
        terms.append(torch.cos(torch.arange(n) * 2 * torch.pi / n * i))
        terms.append(torch.sin(torch.arange(n) * 2 * torch.pi / n * i))
        labels.append(f"cos_{i}")
        labels.append(f"sin_{i}")
    if n%2 == 0:
        terms.append(torch.tensor([(1 if j%2==0 else -1) for j in range(n)]))
        labels.append(f"sign")
    terms = torch.stack(terms)
    terms = terms / terms.norm(dim=-1, keepdim=True)
    return terms.cuda(), labels

fourier_terms, fourier_labels = fourier_basis(10)
imshow(fourier_terms @ fourier_terms.T, title="Pairwise dot product (should be orthogonal)")
imshow(fourier_terms @ U2, y=fourier_labels, xaxis="Singular Vectors", yaxis="Fourier Terms", title="Singular Vectors of digit average in the Fourier basis")
# imshow(fourier_terms.T @ U2)
line(U2.T, title="Singular Vectors of by digit")
# %%
U, S, Vh = number_embeddings_centered.svd()
line(S, title="Singular values of W_E")
line(U[:, :10].T, title="First 10 rows of U")
line((U[:, :10].T) * S[:10, None], title="First 10 rows of U (Weighted_by_S)")
fourier_terms_420, fourier_labels_420 = fourier_basis(420)
line((U[:, :10].T @ fourier_terms_420) * S[:10, None], title="Top 10 singular vectors in Fourier Space (420)", x=fourier_labels_420, xaxis="Fourier Component")

fourier_terms_100, fourier_labels_100 = fourier_basis(100)
U100, S100, Vh100 = number_embeddings_centered[:100].svd()
line((U100[:, :10].T @ fourier_terms_100) * S[:10, None], title="Top 10 singular vectors in Fourier Space (100)", x=fourier_labels_100, xaxis="Fourier Component")
# %%
line((fourier_terms_100 @ number_embeddings_centered[:100]).norm(dim=-1), x=fourier_labels_100, title="Norm of each Fourier Component (100 to 200)")
fourier_terms_400, fourier_labels_400 = fourier_basis(400)

line((fourier_terms_400 @ number_embeddings_centered[:400]).norm(dim=-1), x=fourier_labels_400, title="Norm of each Fourier Component (100 to 500)")
# %%
def make_zero_shot_prompt(num_prompts=10, range_nums=(100, 200)):
    x = np.random.randint(*range_nums, size=(num_prompts))
    y = np.random.randint(*range_nums, size=(num_prompts))
    z = x+y
    prompts = [f"{x[i]}+{y[i]}=" for i in range(num_prompts)]
    answers = [f"{z[i]}" for i in range(num_prompts)]
    return prompts, answers, torch.tensor(x, device="cuda"), torch.tensor(y, device="cuda"), torch.tensor(z, device="cuda")


def make_n_shot_prompt(num_prompts=10, num_shots=2, range_nums=(100, 200), seed=42):
    np.random.seed(seed)
    full_prompts = [""] * num_prompts
    for i in range(num_shots):
        p, a, _, _, _ = make_zero_shot_prompt(num_prompts, range_nums)
        for j in range(num_prompts):
            full_prompts[j] = full_prompts[j] + f"{p[j]}{a[j]}\n"
    prompts, answers, x, y, z = make_zero_shot_prompt(num_prompts, range_nums)
    full_prompts = [f"{full_prompts[i]}{prompts[i]}" for i in range(num_prompts)]
    return full_prompts, prompts, answers, x, y, z


num_prompts = 256
num_shot = 2
full_prompts, prompts, answers, x, y, z = make_n_shot_prompt(num_prompts, num_shot)
print(full_prompts, prompts, answers, x, y, z)
# %%
tokens = model.to_tokens(full_prompts)
print("tokens.shape", tokens.shape)
logits, cache = model.run_with_cache(tokens)
log_probs = logits.log_softmax(dim=-1)
answer_tokens = torch.tensor([model.to_single_token(a) for a in answers]).cuda()
plps = (log_probs[torch.arange(num_prompts).cuda(), -1, answer_tokens])
line(plps, x=prompts)

is_top_answer = (log_probs[:, -1, :].argmax(dim=-1) == answer_tokens)
scatter(x=is_top_answer, y=plps, title="Is top answer vs. log prob of top answer", yaxis="correct log prob", xaxis="is top answer")

# %%
filter = is_top_answer & (plps > -0.7)
print(filter.sum())
# %%
full_prompts_filt = [i for c, i in enumerate(full_prompts) if filter[c].item()]
prompts_filt = [i for c, i in enumerate(prompts) if filter[c].item()]
answers_filt = [i for c, i in enumerate(answers) if filter[c].item()]
x_filt = [i for c, i in enumerate(x) if filter[c].item()]
y_filt = [i for c, i in enumerate(y) if filter[c].item()]
z_filt = [i for c, i in enumerate(z) if filter[c].item()]
num_prompts_filt = filter.sum()
tokens_filt = tokens[filter]
answer_tokens_filt = answer_tokens[filter]
print("tokens_filt.shape", tokens_filt.shape)
# %%
# Residual Stream Patching
clean_tokens = tokens_filt.clone()
clean_logits, clean_cache = model.run_with_cache(clean_tokens)
clean_answer = answer_tokens_filt.clone()
torch.manual_seed(SEED)











# %%
torch.manual_seed(SEED+1)
corr_x = torch.randint(100, 200, (num_prompts_filt,)).cuda()
corr_x_index = torch.tensor([model.to_single_token(str(i.item())) for i in corr_x])
corr_x_answer = torch.tensor([model.to_single_token(str(i.item() + j.item())) for i, j in zip(corr_x, y_filt)]).cuda()
corr_x_tokens = tokens_filt.clone()
X_POSITION = -4
corr_x_tokens[:, X_POSITION] = corr_x_index

corr_x_logits, corr_x_cache = model.run_with_cache(corr_x_tokens)
corr_x_log_probs = corr_x_logits.log_softmax(dim=-1)
corr_x_plps = (corr_x_log_probs[torch.arange(num_prompts_filt).cuda(), -1, corr_x_answer])
line(corr_x_plps)
print("Corr X prompt 0")
print(model.to_string(corr_x_tokens[0]))
print("Clean Prompt 0")
print(model.to_string(clean_tokens[0]))
print("Corr X Answer 0")
print(model.to_string(corr_x_answer[0]))

# %%
torch.manual_seed(SEED+2)
corr_y = torch.randint(100, 200, (num_prompts_filt,)).cuda()
corr_y_index = torch.tensor([model.to_single_token(str(i.item())) for i in corr_y])
corr_y_answer = torch.tensor([model.to_single_token(str(i.item() + j.item())) for i, j in zip(corr_y, x_filt)]).cuda()
corr_y_tokens = tokens_filt.clone()
Y_POSITION = -2
corr_y_tokens[:, Y_POSITION] = corr_y_index

corr_y_logits, corr_y_cache = model.run_with_cache(corr_y_tokens)
corr_y_log_probs = corr_y_logits.log_softmax(dim=-1)
corr_y_plps = (corr_y_log_probs[torch.arange(num_prompts_filt).cuda(), -1, corr_y_answer])
line(corr_y_plps, title="corr_y_plps")

print("Corr y prompt 0")
print(model.to_string(corr_y_tokens[0]))
print("Clean Prompt 0")
print(model.to_string(clean_tokens[0]))
print("Corr y Answer 0")
print(model.to_string(corr_y_answer[0]))
# %%
corr_both_answer = torch.tensor([model.to_single_token(str(i.item() + j.item())) for i, j in zip(corr_x, corr_y)]).cuda()
corr_both_tokens = tokens_filt.clone()
Y_POSITION = -2
corr_both_tokens[:, X_POSITION] = corr_x_index
corr_both_tokens[:, Y_POSITION] = corr_y_index

corr_both_logits, corr_both_cache = model.run_with_cache(corr_both_tokens)
corr_both_log_probs = corr_both_logits.log_softmax(dim=-1)
corr_both_plps = (corr_both_log_probs[torch.arange(num_prompts_filt).cuda(), -1, corr_both_answer])
line(corr_both_plps, title="corr_both_plps")

print("Corr both prompt 0")
print(model.to_string(corr_both_tokens[0]))
print("Clean Prompt 0")
print(model.to_string(clean_tokens[0]))
print("Corr both Answer 0")
print(model.to_string(corr_both_answer[0]))

# %%
# Define patching metric
# Clean log prob
def metric(patched_logits, per_token=False):
    patched_log_probs = patched_logits[:, -1, :].float().log_softmax(dim=-1)
    batch = len(patched_log_probs)
    plps = -patched_log_probs[torch.arange(batch, device="cuda"), answer_tokens_filt]
    if per_token:
        return plps
    else:
        return plps.mean()
clean_baseline = (metric(clean_logits))
print("clean_baseline", clean_baseline)
corr_x_baseline = (metric(corr_x_logits))
print("corr_x_baseline", corr_x_baseline)
corr_y_baseline = (metric(corr_y_logits))
print("corr_y_baseline", corr_y_baseline)
corr_both_baseline = (metric(corr_both_logits))
print("corr_both_baseline", corr_both_baseline)

# %%
POS_LABELS = ["X", "+", "Y", "="]

def resid_patch_hook(resid_pre, hook, pos, layer):
    resid_pre[:, pos, :] = clean_cache["resid_pre", layer][:, pos, :]
    return resid_pre


# %%
corr_x_resid_patching = torch.zeros((4, n_layers)).cuda()
for pos in range(-4, 0):
    for layer in range(n_layers):
        patched_logits = model.run_with_hooks(corr_x_tokens, fwd_hooks = [(
            utils.get_act_name("resid_pre", layer),
            partial(resid_patch_hook, pos=pos, layer=layer),)
        ])
        corr_x_resid_patching[pos, layer] = metric(patched_logits)

OFFSET_LOG_PROB = 6
imshow((OFFSET_LOG_PROB - corr_x_resid_patching), title="corr_x_resid_patching (6 - clean_log_prob)", y=POS_LABELS, yaxis="Position", xaxis="Layer", zmin=0, zmax=OFFSET_LOG_PROB, color_continuous_scale="Blues")
# %%
corr_y_resid_patching = torch.zeros((4, n_layers)).cuda()
for pos in range(-4, 0):
    for layer in range(n_layers):
        patched_logits = model.run_with_hooks(corr_y_tokens, fwd_hooks = [(
            utils.get_act_name("resid_pre", layer),
            partial(resid_patch_hook, pos=pos, layer=layer),)
        ])
        corr_y_resid_patching[pos, layer] = metric(patched_logits)

OFFSET_LOG_PROB = 6
imshow((OFFSET_LOG_PROB - corr_y_resid_patching), title="corr_y_resid_patching (6 - clean_log_prob)", y=POS_LABELS, yaxis="Position", xaxis="Layer", zmin=0, zmax=OFFSET_LOG_PROB, color_continuous_scale="Blues")

# %%
corr_both_resid_patching = torch.zeros((4, n_layers)).cuda()
for pos in range(-4, 0):
    for layer in range(n_layers):
        patched_logits = model.run_with_hooks(corr_both_tokens, fwd_hooks = [(
            utils.get_act_name("resid_pre", layer),
            partial(resid_patch_hook, pos=pos, layer=layer),)
        ])
        corr_both_resid_patching[pos, layer] = metric(patched_logits)

OFFSET_LOG_PROB = 6
imshow((OFFSET_LOG_PROB - corr_both_resid_patching), title="corr_both_resid_patching (6 - clean_log_prob)", y=POS_LABELS, yaxis="Position", xaxis="Layer", zmin=0, zmax=OFFSET_LOG_PROB, color_continuous_scale="Blues")
# %%



# %%
def resid_patch_hook(resid_pre, hook, pos, layer):
    resid_pre[:, pos, :] = clean_cache["resid_pre", layer][:, pos, :]
    return resid_pre

corr_tokens = torch.stack([corr_x_tokens, corr_y_tokens, corr_both_tokens])
corr_labels = ["corr_x", "corr_y", "corr_both"]
corr_resid_patching = torch.zeros((3, 4, n_layers)).cuda()
for i, (lab, toks) in tqdm.tqdm(enumerate(zip(corr_labels, corr_tokens))):
    for pos in range(-4, 0):
        for layer in range(n_layers):
            patched_logits = model.run_with_hooks(toks, fwd_hooks = [(
                utils.get_act_name("resid_pre", layer),
                partial(resid_patch_hook, pos=pos, layer=layer),)
            ])
            corr_resid_patching[i, pos, layer] = metric(patched_logits)

OFFSET_LOG_PROB = 6
imshow((OFFSET_LOG_PROB - corr_resid_patching), title="corr_resid_patching (6 - clean_log_prob)", y=POS_LABELS, yaxis="Position", xaxis="Layer", zmin=0, zmax=OFFSET_LOG_PROB, color_continuous_scale="Blues", facet_col=0, facet_labels=corr_labels)
# %%
def mlp_patch_hook(mlp_out, hook, pos, layer):
    mlp_out[:, pos, :] = clean_cache["mlp_out", layer][:, pos, :]
    return mlp_out

corr_tokens = torch.stack([corr_x_tokens, corr_y_tokens, corr_both_tokens])
corr_labels = ["corr_x", "corr_y", "corr_both"]
corr_mlp_patching = torch.zeros((3, 4, n_layers)).cuda()
for i, (lab, toks) in tqdm.tqdm(enumerate(zip(corr_labels, corr_tokens))):
    for pos in range(-4, 0):
        for layer in range(n_layers):
            patched_logits = model.run_with_hooks(toks, fwd_hooks = [(
                utils.get_act_name("mlp_out", layer),
                partial(mlp_patch_hook, pos=pos, layer=layer),)
            ])
            corr_mlp_patching[i, pos, layer] = metric(patched_logits)

imshow(-corr_mlp_patching, title="corr_mlp_patching (Clean Log Prob)", y=POS_LABELS, yaxis="Position", xaxis="Layer", facet_col=0, facet_labels=corr_labels, zmin=-15, zmax=0, color_continuous_scale="Blues")


# %%
def attn_patch_hook(attn_out, hook, pos, layer):
    attn_out[:, pos, :] = clean_cache["attn_out", layer][:, pos, :]
    return attn_out

corr_tokens = torch.stack([corr_x_tokens, corr_y_tokens, corr_both_tokens])
corr_labels = ["corr_x", "corr_y", "corr_both"]
corr_attn_patching = torch.zeros((3, 4, n_layers)).cuda()
for i, (lab, toks) in tqdm.tqdm(enumerate(zip(corr_labels, corr_tokens))):
    for pos in range(-4, 0):
        for layer in range(n_layers):
            patched_logits = model.run_with_hooks(toks, fwd_hooks = [(
                utils.get_act_name("attn_out", layer),
                partial(attn_patch_hook, pos=pos, layer=layer),)
            ])
            corr_attn_patching[i, pos, layer] = metric(patched_logits)

OFFSET_LOG_PROB = 6
imshow(-corr_attn_patching, title="corr_attn_patching (Clean Log Prob)", y=POS_LABELS, yaxis="Position", xaxis="Layer", facet_col=0, facet_labels=corr_labels, zmin=-15, zmax=0, color_continuous_scale="Blues")
# %%

for i in range(3):
    patched_logits = model.run_with_hooks(
        corr_tokens[i],
        fwd_hooks = [
            (
                utils.get_act_name("mlp_out", layer),
                partial(mlp_patch_hook, layer=layer, pos=-1)
            )
            for layer in range(19, 20)
        ]
    )

    print(corr_labels[i])
    print(metric(patched_logits))
    print("Acc:", (patched_logits[:, -1, :].argmax(dim=-1)==clean_answer).float().mean())
# %%
clean_resid_stack, resid_labels = clean_cache.get_full_resid_decomposition(expand_neurons=False, apply_ln=True, pos_slice=-1, return_labels=True)
W_U_clean = model.W_U[:, clean_answer]
clean_dla = (clean_resid_stack * W_U_clean.T.float()).sum(-1).mean(-1)
line(clean_dla, x=resid_labels, title="Direct Logit Attribution Clean")
# %%
dla_102 = (clean_resid_stack @ model.W_U[:, model.to_single_token("102")].float()).mean(-1)
line(dla_102, x=resid_labels, title="Direct Logit Attribution to 102")

# %%
line((clean_resid_stack[:, :, None, :].to(torch.bfloat16) @ (model.blocks[24].attn.OV[10] @ W_U_clean[:, :].T[:, :, None])).AB.squeeze().mean(-1), x=resid_labels, title="DLA mediated via head L24H10")
# %%
i_am_a_number_U = model.W_U[:, [model.to_single_token(str(i)) for i in range(200, 400)]].mean(-1)
# %%
W_U_clean_centered = model.W_U[:, clean_answer] - i_am_a_number_U[:, None]
clean_dla_centered = (clean_resid_stack * W_U_clean_centered.T.float()).sum(-1).mean(-1)
line(clean_dla_centered, x=resid_labels, title="Direct Logit Attribution Clean (Minus Mean)")
scatter(x=clean_dla, y=clean_dla_centered, hover=resid_labels, xaxis="Clean DLA", yaxis="Clean DLA (Minus Mean)", title="Direct Logit Attribution Clean (Minus Mean)", include_diag=True)

# %%
units_dla = []
for i, ans in enumerate(clean_answer):
    a = int(model.to_string(ans))
    unit = a % 10
    i_am_a_unit = model.W_U[:, [unit+10*j for j in range(20, 40)]].mean(-1)
    units_dla.append(clean_resid_stack[:, i, :] @ (W_U_clean[:, i] - i_am_a_unit).float())
units_dla = torch.stack(units_dla, dim=-1)
line(units_dla.mean(-1), x=resid_labels, title="Direct Logit Attribution Clean (Minus Units)")
scatter(x=clean_dla_centered, y=units_dla.mean(-1), hover=resid_labels, xaxis="Clean DLA Centered", yaxis="Units DLA", title="Direct Logit Attribution Clean (Minus Units)", include_diag=True)

# %%
tens_dla = []
for i, ans in enumerate(clean_answer):
    a = int(model.to_string(ans))
    ten = (a//10) % 10
    i_am_a_ten = model.W_U[:, [j for j in range(200, 400) if ((j//10) % 10)==ten]].mean(-1)
    tens_dla.append(clean_resid_stack[:, i, :] @ (W_U_clean[:, i] - i_am_a_ten).float())
tens_dla = torch.stack(tens_dla, dim=-1)
line(tens_dla.mean(-1), x=resid_labels, title="Direct Logit Attribution Clean (Minus tens)")
scatter(x=clean_dla_centered, y=tens_dla.mean(-1), hover=resid_labels, xaxis="Clean DLA Centered", yaxis="Tens DLA", title="Direct Logit Attribution Clean (Minus Units)", include_diag=True)

# %%
hundreds_dla = []
for i, ans in enumerate(clean_answer):
    a = int(model.to_string(ans))
    hundred = a//100
    i_am_a_hundred = model.W_U[:, [j for j in range(200, 400) if ((j//100))==hundred]].mean(-1)
    hundreds_dla.append(clean_resid_stack[:, i, :] @ (W_U_clean[:, i] - i_am_a_hundred).float())
hundreds_dla = torch.stack(hundreds_dla, dim=-1)
line(hundreds_dla.mean(-1), x=resid_labels, title="Direct Logit Attribution Clean (Minus hundreds)")
scatter(x=clean_dla_centered, y=hundreds_dla.mean(-1), hover=resid_labels, xaxis="Clean DLA Centered", yaxis="hundreds DLA", title="Direct Logit Attribution Clean (Minus Units)", include_diag=True)
# %%
neuron_dlas = []
neuron_layer_range = torch.arange(17, 28)
for layer in neuron_layer_range.tolist():

    mlp_acts = clean_cache["post", layer][:, -1, :]

    W_outU = model.blocks[layer].mlp.W_out @ (W_U_clean_centered / clean_cache["scale"][:, -1, :].T)

    neuron_dla = einops.einsum(W_outU, mlp_acts, "d_mlp d_model, d_model batch -> d_mlp")/len(clean_tokens)
    neuron_dlas.append(neuron_dla)
histogram(torch.stack(neuron_dlas).T, barmode="overlay")
# %%
px.histogram(to_numpy(neuron_dlas[2]), hover_name=np.arange(d_mlp), marginal="rug")
# %%
fourier_U = (fourier_terms_400 @ model.W_U[:, number_token_index[100:500]].T.float()).T
start_layer = 17
end_layer = 28

W_out_stack = torch.stack([model.blocks[l].mlp.W_out for l in range(start_layer, end_layer)])
W_out_stack_normed = W_out_stack / W_out_stack.norm(dim=-1, keepdim=True)
print(W_out_stack_normed.shape)
W_out_fourier = W_out_stack_normed.float() @ fourier_U
print(W_out_fourier.shape)
# %%
line(W_out_fourier[:, :, 8]/fourier_U[:, 8].norm(), line_labels=[str(i) for i in range(start_layer, end_layer)])
# %%

line(W_out_fourier.std(-1), line_labels=[str(i) for i in range(start_layer, end_layer)], title="W_out Fourier Std")
line((W_out_fourier.max(-1).values - W_out_fourier.min(-1).values), line_labels=[str(i) for i in range(start_layer, end_layer)], title="W_out Fourier Range")


# %%
for l in range(start_layer, 20):
    scatter(x=W_out_fourier[l-start_layer].std(-1), y=neuron_dlas[l-start_layer], xaxis="Std", yaxis="DLA", title=f"Layer {l} DLA vs W_out Fourier Std")
# %%
def only_diff_unit(x, y):
    hundred = x//100 == y//100
    tens = (x//10)%10 == (y//10)%10
    unit = x%10 == y%10
    return hundred & tens
print("(106, 107)", only_diff_unit(106, 107), )
print("(116, 106)", only_diff_unit(116, 106), )
print("(116, 216)", only_diff_unit(116, 216), )

def only_diff_tens(x, y):
    hundred = x//100 == y//100
    tens = (x//10)%10 == (y//10)%10
    unit = x%10 == y%10
    return hundred & unit
print("(106, 107)", only_diff_tens(106, 107), )
print("(116, 106)", only_diff_tens(116, 106), )
print("(116, 216)", only_diff_tens(116, 216), )

def only_diff_hundreds(x, y):
    hundred = x//100 == y//100
    tens = (x//10)%10 == (y//10)%10
    unit = x%10 == y%10
    return tens & unit
print("(106, 107)", only_diff_hundreds(106, 107), )
print("(116, 106)", only_diff_hundreds(116, 106), )
print("(116, 216)", only_diff_hundreds(116, 216), )
# # %%
# units_dla = []
# for i, ans in enumerate(clean_answer):
#     a = int(model.to_string(ans))
#     unit_relevant = model.W_U[:, [j for j in range(200, 400) if only_diff_unit(j, a)]].mean(-1)
#     units_dla.append(clean_resid_stack[-30:, i, :] @ (W_U_clean[:, i] - unit_relevant).float())
# units_dla = torch.stack(units_dla, dim=-1)
# line(units_dla.mean(-1), x=resid_labels[-30:], title="Direct Logit Attribution Clean (Minus Units)")
# scatter(x=clean_dla_centered[-30:], y=units_dla.mean(-1), hover=resid_labels[-30:], xaxis="Clean DLA Centered", yaxis="Units DLA", title="Direct Logit Attribution Clean (Minus Units)", include_diag=True)
# # %%
# tens_dla = []
# for i, ans in enumerate(clean_answer):
#     a = int(model.to_string(ans))
#     unit_relevant = model.W_U[:, [j for j in range(200, 400) if only_diff_unit(j, a)]].mean(-1)
#     tens_dla.append(clean_resid_stack[-30:, i, :] @ (W_U_clean[:, i] - unit_relevant).float())
# tens_dla = torch.stack(tens_dla, dim=-1)
# line(tens_dla.mean(-1), x=resid_labels[-30:], title="Direct Logit Attribution Clean (Minus tens)")
# scatter(x=clean_dla_centered[-30:], y=tens_dla.mean(-1), hover=resid_labels[-30:], xaxis="Clean DLA Centered", yaxis="tens DLA", title="Direct Logit Attribution Clean (Minus tens)", include_diag=True)
# # %%
# hundreds_dla = []
# for i, ans in enumerate(clean_answer):
#     a = int(model.to_string(ans))
#     unit_relevant = model.W_U[:, [j for j in range(200, 400) if only_diff_unit(j, a)]].mean(-1)
#     hundreds_dla.append(clean_resid_stack[-30:, i, :] @ (W_U_clean[:, i] - unit_relevant).float())
# hundreds_dla = torch.stack(hundreds_dla, dim=-1)
# line(hundreds_dla.mean(-1), x=resid_labels[-30:], title="Direct Logit Attribution Clean (Minus hundreds)")
# scatter(x=clean_dla_centered[-30:], y=hundreds_dla.mean(-1), hover=resid_labels[-30:], xaxis="Clean DLA Centered", yaxis="hundreds DLA", title="Direct Logit Attribution Clean (Minus hundreds)", include_diag=True)











# %%
W_U_clean = model.W_U[:, clean_answer]
# %%
start_layer = 17
end_layer = 22
mlp_acts = torch.stack([clean_cache["post", layer][:, -1, :] for layer in range(start_layer, end_layer)], dim=1)
print(f"{mlp_acts.shape=}")
W_out_stack = torch.stack([model.blocks[layer].mlp.W_out for layer in range(start_layer, end_layer)], dim=0)

print(f"{W_out_stack.shape=}")
LAYER_LABELS = [str(i) for i in range(start_layer, end_layer)]
# %%

# %%
def only_diff_unit(x, y):
    hundred = x//100 == y//100
    tens = (x//10)%10 == (y//10)%10
    unit = x%10 == y%10
    return hundred & tens
print("(106, 107)", only_diff_unit(106, 107), )
print("(116, 106)", only_diff_unit(116, 106), )
print("(116, 216)", only_diff_unit(116, 216), )

def only_diff_tens(x, y):
    hundred = x//100 == y//100
    tens = (x//10)%10 == (y//10)%10
    unit = x%10 == y%10
    return hundred & unit
print("(106, 107)", only_diff_tens(106, 107), )
print("(116, 106)", only_diff_tens(116, 106), )
print("(116, 216)", only_diff_tens(116, 216), )

def only_diff_hundreds(x, y):
    hundred = x//100 == y//100
    tens = (x//10)%10 == (y//10)%10
    unit = x%10 == y%10
    return tens & unit
print("(106, 107)", only_diff_hundreds(106, 107), )
print("(116, 106)", only_diff_hundreds(116, 106), )
print("(116, 216)", only_diff_hundreds(116, 216), )

tens_dla = []
for i, ans in enumerate(clean_answer):
    a = int(model.to_string(ans))
    ten_relevant = model.W_U[:, [model.to_single_token(str(j)) for j in range(200, 400) if only_diff_tens(j, a)]].mean(-1)
    unembed_vec = W_U_clean[:, i] - ten_relevant
    tens_dla.append((W_out_stack @ unembed_vec) * mlp_acts[i])
tens_dla = torch.stack(tens_dla, dim=-1)
line(tens_dla.mean(-1), title="Direct Logit Attribution Clean (Minus tens)", line_labels = LAYER_LABELS)
# %%
ni = 3998
l = 16
nlabel = f"L{l}N{ni}"
wout = W_out_stack[l - start_layer, ni]
line(wout @ model.W_U[:, number_token_index], title=f"{nlabel} Direct Unembed")
line((wout @ model.W_U[:, number_token_index[100:500]]).float() @ fourier_terms_400.T, title=f"{nlabel} Fourier Unembed", x=fourier_labels_400)

line([wout @ model.W_U[:, number_token_index[100+i:500:10]].mean(dim=-1) for i in range(10)], title='Weight DLA by digit')
# line(fourier_basis(10)[0])

line(clean_cache["post", l][:, -1, ni], x=prompts_filt, title=f"{nlabel} MLP Activations")

df = pd.DataFrame({
    "act": clean_cache["post", l][:, -1, ni].tolist(),
    "unit": [i.item()%10 for i in z_filt],
    "ten": [(i.item()//10)%10 for i in z_filt],
    "hundred": [i.item()//100 for i in z_filt],
    # "both_mod3": (((x_filt % 3)==0) & ((y_filt % 3)==0)).tolist(),
})

# px.box(df, x="both_mod3", y="act").show()
px.box(df, x="unit", y="act", title=f"{nlabel} activation by unit").show()
px.box(df, x="hundred", y="act", title=f"{nlabel} activation by hundred").show()
px.box(df, x="ten", y="act", title=f"{nlabel} activation by ten").show()


# line(einops.rearrange(wout @ model.W_U[:, number_token_index], "(m x) -> x m", x=2), line_labels=["even", "odd"], title=f"The Parity Neuron {nlabel}")

# %%
ni = 13991
l = 19
df = pd.DataFrame({
    "act": clean_cache["post", l][:, -1, ni].tolist(),
    "unit": [i.item()%10 for i in z_filt],
    "ten": [(i.item()//10)%10 for i in z_filt],
    "hundred": [i.item()//100 for i in z_filt],
    "both_mod3": to_numpy(((torch.tensor(x_filt) % 3)==0) & ((torch.tensor(y_filt) % 3)==0)),
})

px.box(df, x="both_mod3", y="act", title="The 3|x & 3|y neuron").show()

# %%
ni = 5406
l = 19
df = pd.DataFrame({
    "act": clean_cache["post", l][:, -1, ni].tolist(),
    "unit": [i.item()%10 for i in z_filt],
    "ten": [(i.item()//10)%10 for i in z_filt],
    "hundred": [i.item()//100 for i in z_filt],
    "mod4": (torch.tensor(z_filt) % 4).tolist(),
    "mod8": (torch.tensor(z_filt) % 8).tolist(),
    "mod12": (torch.tensor(z_filt) % 12).tolist(),
    "mod16": (torch.tensor(z_filt) % 16).tolist(),
})

px.box(df, x="mod4", y="act", title="L19N5406").show()
px.box(df, x="mod8", y="act", title="L19N5406").show()
px.box(df, x="mod12", y="act", title="L19N5406").show()
px.box(df, x="mod16", y="act", title="L19N5406").show()

acts = clean_cache["post", l][:, -1, ni]
indices = acts.argsort(descending=True).tolist()
for i in indices[:30]:
    print(acts[i].item())
    print(x_filt[i].item() % 4, y_filt[i].item() % 4, z_filt[i].item() % 4)
    # print(prompts_filt[i])
    # print(answers_filt[i])

# %%
df = pd.DataFrame({
    "act": clean_cache["post", l][:, -1, ni].tolist(),
    "unit": [i.item()%10 for i in z_filt],
    "ten": [(i.item()//10)%10 for i in z_filt],
    "hundred": [i.item()//100 for i in z_filt],
})

px.box(df, x="unit", y="act").show()
px.box(df, x="hundred", y="act").show()
px.box(df, x="ten", y="act").show()
# %%
l=19
ni=12978
df = pd.DataFrame({
    "act": clean_cache["post", l][:, -1, ni].tolist(),
    "mod3": [i.item()%3 for i in z_filt],
    
})

px.box(df, x="mod3", y="act", title=f"L{l}N{ni} The not a multiple of 3 neuron").show()
# px.box(df, x="hundred", y="act").show()
# px.box(df, x="ten", y="act").show()
# %%
neuron_df = pd.DataFrame(
    {
        "L":[l for l in range(start_layer, end_layer) for ni in range(d_mlp)],
        "N":[ni for l in range(start_layer, end_layer) for ni in range(d_mlp)],
    }
)
tens_affecting_score = tens_dla.mean(-1)
neuron_df["tens_affecting"] = tens_affecting_score.flatten().tolist()
# %%
fourier_U = model.W_U[:, number_token_index[100:500]].float() @ fourier_terms_400.T
cos_133_vec = fourier_U[:, fourier_labels_400.index("cos_133")]
cos_133_score = W_out_stack.float() @ cos_133_vec
neuron_df["cos_133_score"] = cos_133_score.flatten().tolist()

px.histogram(neuron_df, title="Cosine 133 Score Distribution", x="cos_133_score", marginal="rug", hover_data=["L", "N"]).show()
# line((wout @ , title=f"{nlabel} Fourier Unembed", x=fourier_labels_400)
# cos_133_score = 



px.scatter(neuron_df, x="cos_133_score", y="tens_affecting", hover_data=["L", "N"], title="Cosine 133 Score vs Tens Affecting").show()




# %%
units_dla = []
for i, ans in enumerate(clean_answer):
    a = int(model.to_string(ans))
    unit = a % 10
    i_am_a_unit = model.W_U[:, [unit+10*j for j in range(20, 40)]].mean(-1)
    units_dla.append(clean_resid_stack[:, i, :] @ (W_U_clean[:, i] - i_am_a_unit).float())
units_dla = torch.stack(units_dla, dim=-1)
line(units_dla.mean(-1), x=resid_labels, title="Direct Logit Attribution Clean (Minus Units)")
scatter(x=clean_dla_centered, y=units_dla.mean(-1), hover=resid_labels, xaxis="Clean DLA Centered", yaxis="Units DLA", title="Direct Logit Attribution Clean (Minus Units)", include_diag=True)

# %%
tens_dla = []
for i, ans in enumerate(clean_answer):
    a = int(model.to_string(ans))
    ten = (a//10) % 10
    i_am_a_ten = model.W_U[:, [j for j in range(200, 400) if ((j//10) % 10)==ten]].mean(-1)
    tens_dla.append(clean_resid_stack[:, i, :] @ (W_U_clean[:, i] - i_am_a_ten).float())
tens_dla = torch.stack(tens_dla, dim=-1)
line(tens_dla.mean(-1), x=resid_labels, title="Direct Logit Attribution Clean (Minus tens)")
scatter(x=clean_dla_centered, y=tens_dla.mean(-1), hover=resid_labels, xaxis="Clean DLA Centered", yaxis="Tens DLA", title="Direct Logit Attribution Clean (Minus Units)", include_diag=True)

# %%
hundreds_dla = []
for i, ans in enumerate(clean_answer):
    a = int(model.to_string(ans))
    hundred = a//100
    i_am_a_hundred = model.W_U[:, [j for j in range(200, 400) if ((j//100))==hundred]].mean(-1)
    hundreds_dla.append(clean_resid_stack[:, i, :] @ (W_U_clean[:, i] - i_am_a_hundred).float())
hundreds_dla = torch.stack(hundreds_dla, dim=-1)
line(hundreds_dla.mean(-1), x=resid_labels, title="Direct Logit Attribution Clean (Minus hundreds)")
scatter(x=clean_dla_centered, y=hundreds_dla.mean(-1), hover=resid_labels, xaxis="Clean DLA Centered", yaxis="hundreds DLA", title="Direct Logit Attribution Clean (Minus Units)", include_diag=True)

# %%
# line(einops.rearrange(wout @ model.W_U[:, number_token_index], "(m x) -> x m", x=2), line_labels=["even", "odd"], title=f"The Parity Neuron {nlabel}")
# %%

for l, ni in [
    (20, 12987),
    (16, 3998)
]:
    clean_resid_stack, resid_labels = clean_cache.decompose_resid(l, pos_slice=-1, return_labels=True, apply_ln=True, mlp_input=True)
    line([clean_resid_stack[:, torch.tensor(z_filt) % 2 ==0].mean(1) @ model.blocks[l].mlp.W_in[:, ni], clean_resid_stack[:, torch.tensor(z_filt) % 2 ==1].mean(1) @ model.blocks[l].mlp.W_in[:, ni]], x=resid_labels, title=f"L{l}N{ni} DNA", line_labels=["even", "odd"])
# %%
is_even = torch.tensor(z_filt) % 2 ==0
is_odd = torch.tensor(z_filt) % 2 ==1
l16_acts = clean_cache["post", 16][:, -1, :]
W_out16 = model.blocks[16].mlp.W_out @ model.blocks[l].mlp.W_in[:, ni]
print(l16_acts.shape, W_out16.shape)
# %%
scatter(x=l16_acts[is_even].mean(0), y=l16_acts[is_odd].mean(0), color=W_out16, xaxis="Even", yaxis="Odd", color_name="DNA to parity", hover=np.arange(d_mlp), title="DNA of L16 Neurons to Parity Neuron")
line(W_out16)
# %%
# top_l16_neurons = [3998, 12088, 7036, 11021, 16144, 11537, 2535, 1176]
# for ni in top_l16_neurons:
    