# %%
from neel.imports import *
from neel_plotly import *

# %%
model = HookedTransformer.from_pretrained("pythia-70m-v0")
torch.set_grad_enabled(False)

# %%
data = load_dataset("NeelNanda/pile-10k", split="train")
# %%
tokenized_data = utils.tokenize_and_concatenate(data, model.tokenizer, max_length=32)
tokenized_data = tokenized_data.shuffle()
print(len(tokenized_data))
all_tokens = tokenized_data["tokens"].cuda()
# %%
cached_mlps = []


def store(mlp_post, hook):
    cached_mlps.append(mlp_post[:, -1, :].detach().cpu().numpy())


# batch_size = 20000
# model.add_hook("blocks.1.mlp.hook_post", store)
# _ = model(all_tokens[:batch_size], stop_at_layer=2)
# print(cached_mlps[0].shape)

# %%
normal_tokens = []
for i in range(1000, 25000):
    if model.W_E[i].norm(dim=-1) > 0.65:
        if model.to_string(i).strip().isalpha():
            normal_tokens.append(i)
normal_tokens = torch.tensor(normal_tokens).cuda()
# normal_tokens = torch.arange(1000, 20000).cuda()[model.W_E[torch.arange(1000, 20000)].norm(dim=-1) > 0.65]
batch_size = len(normal_tokens)

# %%
x = 100
y = 200

base_tokens_xy = all_tokens[:batch_size].clone()
base_tokens_xb = all_tokens[batch_size : batch_size * 2].clone()
base_tokens_ay = all_tokens[batch_size * 2 : batch_size * 3].clone()

base_tokens_xy[:, -2] = x
base_tokens_xb[:, -2] = x
base_tokens_xy[:, -1] = y
base_tokens_ay[:, -1] = y

base_tokens_xb[:, -1] = normal_tokens
base_tokens_ay[:, -2] = normal_tokens

# %%
model.reset_hooks()
cached_mlps = {}


def store(mlp_post, hook, label):
    cached_mlps[label] = mlp_post[:, -1, :].detach().cpu().numpy()


model.run_with_hooks(
    base_tokens_xy,
    stop_at_layer=2,
    fwd_hooks=[("blocks.1.mlp.hook_post", partial(store, label="base_tokens_xy"))],
)
model.run_with_hooks(
    base_tokens_xb,
    stop_at_layer=2,
    fwd_hooks=[("blocks.1.mlp.hook_post", partial(store, label="base_tokens_xb"))],
)
model.run_with_hooks(
    base_tokens_ay,
    stop_at_layer=2,
    fwd_hooks=[("blocks.1.mlp.hook_post", partial(store, label="base_tokens_ay"))],
)
# %%
xy_vec = cached_mlps["base_tokens_xy"]
xb_vec = cached_mlps["base_tokens_xb"]
ay_vec = cached_mlps["base_tokens_ay"]

# %%
neuron_df = pd.DataFrame({
    "xy": xy_vec.mean(0),
    "xb": xb_vec.mean(0),
    "ay": ay_vec.mean(0),
    "index": np.arange(model.cfg.d_mlp)
})
neuron_df["diff"] = neuron_df["xy"] - np.maximum(neuron_df["xb"], neuron_df["ay"])
line(neuron_df["diff"].values)
histogram(neuron_df["diff"])
import scipy
print(scipy.stats.kurtosis(neuron_df["diff"]))
# %%
top_neurons = neuron_df["diff"].sort_values(ascending=False).index[:5].values
for ni in top_neurons:
    histogram(np.stack([xy_vec[:, ni], xb_vec[:, ni], ay_vec[:, ni]]).T, title=f"Neuron {ni}", barmode="overlay", line_labels=["XY", "XB", "AY"])

# %%
v = xy_vec.mean(0)
xy_vec_norm = xy_vec / np.linalg.norm(xy_vec, axis=-1, keepdims=True)
print(np.linalg.norm(v))
histogram(np.linalg.norm(xy_vec, axis=-1))
histogram(xy_vec_norm @ (v / np.linalg.norm(v)))
# %%
