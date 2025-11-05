---
title: Unified Attention
---
[](){ #unified_attn }

## Overview

Unified Attention is a new attention backend introduced in vllm-gaudi plugin. It brings several benefits when compared to previous approaches:

1. properly handles shared blocks in case of contiguous kv-cache
1. allows running both prefill and decode tokens in a single batch (aka mixed batches)
1. allows running all query tokens 'flattened', i.e. without the need of a separate seq_len dimension

The name comes from the fact that it combines several previous algorithms into a single implementation. Since it's a completely new attention backend, not all features are yet supported.

## Main idea

To get the main idea behind the algorithm, let's work on a concrete example. Assuming:

* block_size=4
* 4 samples in a batch, out of which 2 are prefills and 2 are decodes, with query lengths=[8, 4, 1, 1] and context lengths=[0, 4, 6, 4]
* we're using scaled dot product attention:
$$\text{Attention}(Q, K, V, B) = \text{softmax}\left( s \cdot QK^\top + B \right) V$$

![](../assets/unified_attn/block_table.png)

We can observe two things:

1. some of the blocks are only used by a single token, and some are shared
2. some of the key values have been just calculated and are available alongside queries and don't need to be fetched from the cache

In a naive implementation we would just multiply whole query times key and value and use appropriate bias to mask unused fields, but that would be very inneficient especially for decodes where usually we have only a single token per sample in a batch and there's almost no overlap between used blocks. We could slice the query and key into chunks and multiply only those regions that have relevant data, but that's currently difficult to achieve due to technical reasons. Instead we can divide the work into 3 separate parts and merge the results at the end.

![](../assets/unified_attn/block_table_annotated.png)

## Splitting softmax

The main trick that unified attention utilizes is spliting and merging softmax values. Softmax is defined as:
$$\text{softmax}(x_i) = \frac{e^{x_i-c}}{\sum_{j} e^{x_j-c}}, c = max(x_i)$$
The problem here lies in the denominator as it contains the sum of all terms. Fortunately we can split the calculation into two separate softmax and then readjust the results and combine them. Let's say we have:
$$z_1, z_2\text{ - local softmax results} \\ c_1, c_2 \text{ - local maxima} \\ s_1, s_2 \text{ - local sums}$$
We can then calculate:
$$c = max(c_1, c_2) \\ adj_i = e^{c_i-c} \\ s = s_1 *adj_1 + s_2* adj_2\\ z_i\prime = \frac{z_i*s_i*adj_i}{s} $$

This way we can calculate parts of softmax and later readjust and recombine the values into the final result. There are two other tricks that we can use. Since we're going to divide by the global sum anyway we can skip dividing by local sums followed by multiplying by local sums during readjustment and keep intermediate 'softmax' values without division. Additionally since readjustment is multiplication by a constant we can utilize the facts that:
$$(sA)B=s(AB) \\ [A; B; C+D] \times [A; C+D; E] = [A; B; C] \times [A; C; E] + [A; B; D] \times [A; D; E] = [A; B; E]$$
and move softmax readjustment after multiplication by V in attention calculation.

## Causal Attention

Causal attention is used to calculate attention values between currently computed Q, K and V. Since we data has been recently calculated, we don't need to fetch it from kv-cache. Prompt lengths are usually much longer then max_num_seqs. This means, in practice, we don't need to distinguish which tokens are used in prompts and which in decodes and use the whole Q relying on attn bias to mask out unnecessary tokens. Since we're using all query tokens one after another it works similarily to merged prefill feature. Here's an example how the computed causal bias might look like:

![](../assets/unified_attn/causal.png)

One optimization that is used here is that we can divide query into equal slices that use different lengths of key:

![](../assets/unified_attn/causal_sliced.png)

This way we can skip parts of the computation where index(key) > index(query). In the current implementation slice size is constant and is set to 512 based on experimental results.

## Shared Attention

Shared attention is used in cases where a context block is used by multiple tokens. This is usually the case when we have a prompt with parts of the context cached or in case of decode when multiple samples share a common prefix. Since shared blocks are used more then once we fetch them all and we multiply them with all the query tokens. Usually the number of shared blocks is relatively small compared to whole kv-cache that's why it's better to fetch them instead of relying on tricks like contiguous_pa. The main difficulty is creating the shared_bias.

## Unique Attention

Since we know that each block is used by upmost one token, we can use two optimizations:

1. compute attention per block instead of per query token
1. use a contiguous slice of kv-cache instead of fetching individual blocks

First optimization allows better handling of batches with large differences between sequence lengths. For example, if we have two samples in a batch, using [4, 12] context blocks respectively, instead of padding the block_table to the highest number of blocks we can use flattened list of blocks. This way the amount of compute we need scales with the sum of blocks_used instead of bs * max(num_blocks). This is a simplified diagram that shows how it works

![](../../docs/assets/unified_attn/unique.png)

The main difficulty in this approach is that several blocks might be used in a single query token and thus we cannot compute softmax directly. Fortunately we can utlize the same approach to calculate softmax in parts and then readjust.

Second optimization comes from the fact that in case of decodes, most of the time we only need to fetch the block once and since we're going to fetch most of the kv-cache anyway we might just use a contiguous chunk instead. This optimization is optional from unified attention's algorithm point of view, but currently unified batch creation assumes it's turned on by default.

## Merging intermediate values

There are 3 optional code paths that unified attention code can take:

1. causal attn
1. shared attn
1. unique attn

Each of those code paths returns a triplet either with (local_attn, local_max, local_sum) or with (None, None, None) in case that path is skipped. The last step is to combine partial values, readjust them and combine them together using the previously described method.

## Unified/Mixed Batches

One of the main benefits of unified attention is that it doesn't distinguish between prompt and decode tokens and the whole attention pass can be computed by a single function without breaking synapse graphs. This means that we no longer need to do any kind of preprocessing of scheduler output (like sorting and separating prompts and decodes). Which code paths should be active in unified attention is based on the presence of a particular bias tensor in attention metadata:

* causal_bias => causal attention is enabled
* shared_bias => shared attention is enabled
* unique_bias => unique attention is enabled

This means that there are 8 possible code paths that we can take. This is reflected when printing particular configuration that is being run. For example phase string of "csu" means that all 3 code paths are used whereas '--u' means that only unique attention is being run.

Most of the model forward code relies only on query_len. Two other dimensions come into play when calculating unified attention - num_shared_blocks and num_unique_blocks. In reality, when contiguous_pa is enabled for unified attention (which is currently enforced) num_unique_blocks is kv-cache slice size that we need to use (this depends on max(block_id) currently in use). Last missing part is whether to include causal_attn or not. This depends on existence of prompt samples in the batch. If there's at least a single prompt in the batch we enable causal attention. The last part beside model forward is dependent on the number of logits that we want to fetch as not all token logits should be passed to sampler. This is usually padded to max_num_seqs, but the code allows creating a more detailed bucketing scheme in the future.

To sum up, a single model execute can be characterized by a following tuple:

*(phase, query_len, num_shared_blocks, num_unique_blocks, num_logits)*
