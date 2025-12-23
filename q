[33mcommit f9cee6f547e4d5bc292195748f33efc642df3ced[m[33m ([m[1;36mHEAD[m[33m -> [m[1;32mslokesha/Update_qwen_from_v0.14.1[m[33m)[m
Author: Michał Kuligowski <michal.kuligowski@intel.com>
Date:   Wed Feb 4 08:53:47 2026 +0100

    Initializatrion profiling noop (#916)
    
    Signed-off-by: Michał Kuligowski <michal.kuligowski@intel.com>

[33mcommit f4126da365645acfad0a020e7ce1431e07b5be0d[m[33m ([m[1;31mslokesha/slokesha/Update_qwen_from_v0.14.1[m[33m)[m
Merge: 29aa3d3 a238387
Author: Spurthi Lokeshappa <spurthi.lokeshappa@intel.com>
Date:   Sun Feb 8 19:13:45 2026 -0800

    Merge branch 'vllm-project:main' into slokesha/Update_qwen_from_v0.14.1

[33mcommit a2383875ed7f66cd51e05bf9e4a78c8cd471d375[m[33m ([m[1;31morigin/main[m[33m, [m[1;31morigin/HEAD[m[33m)[m
Author: Youlei Yang <youlei.yang@intel.com>
Date:   Sat Feb 7 07:09:53 2026 +0800

    Set device according to local rank (#788)
    
    ### Motivation
    For a typical node with 8xGaudi2E HPUs, the devices are break into two
    groups with 4 HPUs connected with top board each. Current random mapping
    between `local_rank` and `module_id` will cause HCCL failure for
    `world_size>4` cases.
    
    ### Changes
    - Set device according to local rank.
    - Use `pyhlml` to set `HABANA_VISIBLE_MODULES` to available modules.
    This is necessary if multiple cases with `world_size=1/2/4` wants to run
    on the same node simultaneously or the available `module_ids` are not
    start with 0.
    
    ---------
    
    Signed-off-by: Youlei Yang <youlei.yang@intel.com>
    Co-authored-by: Agata Dobrzyniewicz <160237065+adobrzyn@users.noreply.github.com>
    Co-authored-by: Michał Kuligowski <michal.kuligowski@intel.com>

[33mcommit 29aa3d31321ed01b70e6ead6c65c028630a7d1e3[m
Merge: 78168e6 5907666
Author: Spurthi Lokeshappa <spurthi.lokeshappa@intel.com>
Date:   Fri Feb 6 14:03:12 2026 -0800

    Merge branch 'libinta/remove_gather_scatter' into slokesha/Update_qwen_from_v0.14.1

[33mcommit 590766646b6839bde89bd3dd50c2059324a8db08[m
Merge: 020936a 67416b1
Author: Spurthi Lokeshappa <spurthi.lokeshappa@intel.com>
Date:   Fri Feb 6 12:07:07 2026 -0800

    Merge branch 'vllm-project:main' into libinta/remove_gather_scatter

[33mcommit 67416b1af67e3b9f1750db52a74ab82789309c14[m
Author: Jan Wieczorek <jwieczorek@habana.ai>
Date:   Fri Feb 6 14:36:05 2026 +0100

    [GAUDISW-244575] Adapt OnlineDefragmenter and CacheSwapUtils for torc… (#889)
    
    …h.compile
    
    Because of the double entrypoint of CacheSwapUtils (forward and swap
    functions) torch.compile would process module and forward function while
    swap's self would refer to unwrapped module. That results in the
    function not being run as compiled Changes made in this patch:
    - Hide CacheSwapUtils entirely in OnlineDefragmenter. Let it be
    responsible for calling the module correctly
     - Moved warmup_defragmenter to defragmenter itself
    - Removed initialize function of OnlineDefragmenter, fully initialize
    object in __init__
     - Adapted unit tests for new implementations
    
    ---------
    
    Signed-off-by: Jan Wieczorek <jwieczorek@habana.ai>
    Co-authored-by: Kamil Kaczor <kamil.kaczor@intel.com>
    Co-authored-by: Michał Kuligowski <michal.kuligowski@intel.com>

[33mcommit 6d15fdc0deb0d1ed7af03b2afb9762c461f820ab[m
Author: Michal Muszynski <141021743+mmuszynskihabana@users.noreply.github.com>
Date:   Fri Feb 6 13:13:04 2026 +0100

    [GAUDISW-244821] Modify ubi docker to support both internal and external builds (#923)
    
    - parametrize all that's different internally and externally, e.g.
    locations of RPM packages, pypi indexes
    - add libomp package which is now required during pytorch modules
    installation
    - add support for RHEL 9.4 builds, handle minor differences between 9.4
    and 9.6
    
    Signed-off-by: Michal Muszynski <mmuszynski@habana.ai>

[33mcommit f51f12c610e344bbada3f805083818f5b4f8fc5b[m
Author: Agata Dobrzyniewicz <160237065+adobrzyn@users.noreply.github.com>
Date:   Fri Feb 6 12:47:33 2026 +0100

    New testowners (#944)
    
    Signed-off-by: Agata Dobrzyniewicz <adobrzyniewicz@habana.ai>

[33mcommit 89feb5d48176811f4196192fc66dede1103d6872[m
Author: Tanner Voas <tanner.voas@intel.com>
Date:   Fri Feb 6 18:57:46 2026 +0800

    Fix torch.compile crash in sampler by removing NumPy dependency in tensor padding (#893)
    
    ## Description
    This PR resolves a blocking ```DispatchKeySet``` mismatch error
    encountered when running ```vLLM``` on ```HPU``` with
    ```torch.compile``` enabled and ```repetition_penalty != 1.0```.
    
    ```sh
    [multiproc_executor.py:822] AssertionError: Guard check failed: 14/0: tensor '__from_numpy(__stack0)' dispatch key set mismatch. expected DispatchKeySet(CPU, BackendSelect, ADInplaceOrView), actual DispatchKeySet(CPU, BackendSelect)
    ```
    
    The root cause was the usage of
    [vllm.utils.torch_utils.make_tensor_with_pad](vscode-file://vscode-app/c:/Users/tvoas/AppData/Local/Programs/Microsoft%20VS%20Code/resources/app/out/vs/code/electron-sandbox/workbench/workbench.html),
    which internally relies on NumPy for padding operations. When compiled
    graphs (HPU backend) encounter tensors created via
    [torch.from_numpy](vscode-file://vscode-app/c:/Users/tvoas/AppData/Local/Programs/Microsoft%20VS%20Code/resources/app/out/vs/code/electron-sandbox/workbench/workbench.html),
    the compiler's dispatch logic fails to reconcile the dispatch keys,
    leading to the following crash during sampling:
    
    ## Changes
    Patched ```make_tensor_with_pad``` In ```utils.py```, we now override
    the core ```make_tensor_with_pad``` function with an HPU-specific
    implementation. This implementation uses pure PyTorch operations
    (```torch.nn.utils.rnn.pad_sequence```) instead of NumPy to ensure
    compatibility with ```torch.compile```.
    Updated the existing but unused ```make_tensor_with_pad_align``` to also
    use the pure PyTorch approach, removing the need for
    ```make_ndarray_with_pad_align```.
    
    ## Validation
    Verified on ```HPU``` using ```Qwen2.5-VL-7B-Instruct```.
    
    Test Case: Randomly generated multimodal prompts with varying
    concurrencies.
    Result: The crash is resolved, and the model generates output
    successfully with repetition penalties applied.
    Performance: No measurable performance regression observed compared to
    the previous NumPy implementation.
    
    Signed-off-by: Tanner Voas <tanner.voas@intel.com>
    Co-authored-by: Michał Kuligowski <michal.kuligowski@intel.com>

[33mcommit 03a8b831224b51eac43a2593e02df25378676368[m
Author: Yi Liu <yi4.liu@intel.com>
Date:   Fri Feb 6 18:49:05 2026 +0800

    [CT] Add FP8 GQA Support (#874)
    
    Depends on https://github.com/vllm-project/vllm-gaudi/pull/929
    
    - Local test
    ```bash
    vllm ({'pretrained': '/mnt/disk5/hf_models/Qwen3-8B-FP8_STATIC-FP8-Attn-LLMC-Test-Only/', 'tensor_parallel_size': 8, 'max_model_len': 4096, 'max_num_seqs': 64, 'gpu_memory_utilization': 0.85, 'dtype': 'bfloat16', 'max_gen_toks': 2048, 'enable_prefix_caching': False, 'max_num_batched_tokens': 32768, 'kv_cache_dtype': 'fp8_inc'}), gen_kwargs: ({}), limit: None, num_fewshot: None, batch_size: 128
    # |Tasks|Version|     Filter     |n-shot|  Metric   |   |Value |   |Stderr|
    # |-----|------:|----------------|-----:|-----------|---|-----:|---|-----:|
    # |gsm8k|      3|flexible-extract|     5|exact_match|↑  |0.8999|±  |0.0083|
    # |     |       |strict-match    |     5|exact_match|↑  |0.8999|±  |0.0083|
    ```
    cc @hshen14 @thuang6
    
    ---------
    
    Signed-off-by: yiliu30 <yi4.liu@intel.com>

[33mcommit 2b082e54ce916dc547a55e585fe6c23d086de6e9[m
Author: Yupeng Zhang <yupeng.zhang@intel.com>
Date:   Fri Feb 6 18:34:35 2026 +0800

    improve model weight loading speed (#807)
    
    This commit introduces the with_thread_limits decorator function. The
    decorator temporarily adjusts OpenMP and PyTorch thread settings based
    on available CPU cores to speed up model weight loading, and restores to
    original value after weights loading.
    
    ---------
    
    Signed-off-by: Yupeng Zhang <yupeng.zhang@intel.com>
    
    Signed-off-by: yupengzh-intel <yupeng.zhang@intel.com>
    Co-authored-by: Agata Dobrzyniewicz <160237065+adobrzyn@users.noreply.github.com>

[33mcommit 020936a0c25ceab7510d4ae8ff223d3cee6d1187[m
Merge: 9abae33 8ed93a6
Author: Agata Dobrzyniewicz <160237065+adobrzyn@users.noreply.github.com>
Date:   Fri Feb 6 11:33:18 2026 +0100

    Merge branch 'main' into libinta/remove_gather_scatter

[33mcommit 8ed93a6ca0e57673fcb62271b941a706ff59b34f[m
Author: Kamil Kaczor <kamil.kaczor@intel.com>
Date:   Fri Feb 6 11:24:32 2026 +0100

    Fix sampler & TP>1 recompilations (#935)
    
    In sampler we've been warming up bs 0 and at the same time due to > 1
    and providing 1.0 parameter we've skipped some cases.
    
    In tp>1 vocab is split between workers and we've been warming up using
    the same range, which corresponds to tp=1. This caused in warmup for
    rank > 1 users to warmup non-local vocab access path vs local access
    path we see in runtime where each rank uses correct vocab.
    
    ---------
    
    Signed-off-by: Kamil Kaczor <kamil.kaczor@intel.com>

[33mcommit 78168e61893c7e0cbc4b4702601196ab14a82b99[m
Merge: 8d92d88 175572b
Author: Spurthi Lokeshappa <spurthi.lokeshappa@intel.com>
Date:   Thu Feb 5 17:16:08 2026 -0800

    Merge branch 'vllm-project:main' into slokesha/Update_qwen_from_v0.14.1

[33mcommit 9abae33f771f54826d41240ab28c1f1fd14ac907[m
Merge: ef9f0b3 175572b
Author: Spurthi Lokeshappa <spurthi.lokeshappa@intel.com>
Date:   Thu Feb 5 17:05:25 2026 -0800

    Merge branch 'vllm-project:main' into libinta/remove_gather_scatter

[33mcommit 175572bb2f66ce772deb62447d8b3c41c809a386[m
Author: Yi Liu <yi4.liu@intel.com>
Date:   Fri Feb 6 00:01:08 2026 +0800

    [CT] Fix CT Config to honor `fp8_inc` KV cache dtype (#929)
    
    Adapt the update in https://github.com/vllm-project/vllm/pull/30141
    
    ```python
            # llm-compressor mdls need to set cache_dtype to "fp8" manually.
            if getattr(quant_config, "kv_cache_scheme", None) is not None:
                kv_cache_dtype = "fp8"
                calculate_kv_scales = False
                if cache_config is not None:
                    cache_config.cache_dtype = "fp8"
                    cache_config.calculate_kv_scales = False
    
            self.kv_cache_torch_dtype = kv_cache_dtype_str_to_dtype(
                kv_cache_dtype, vllm_config.model_config
            )
            self.kv_cache_dtype = kv_cache_dtype
    ```
    
    
    cc @hshen14 @thuang6 @lkk12014402
    
    ---------
    
    Signed-off-by: yiliu30 <yi4.liu@intel.com>

[33mcommit 8d92d887ad6f0c30e57a7959527921aed3c92dc6[m
Merge: 6376480 ef9f0b3
Author: Spurthi Lokeshappa <spurthi.lokeshappa@intel.com>
Date:   Wed Feb 4 11:29:55 2026 -0800

    Merge branch 'libinta/remove_gather_scatter' into slokesha/Update_qwen_from_v0.14.1

[33mcommit ef9f0b3fa644e23c804ac3884cd40df8b123b0da[m
Merge: 7b9a1aa 333907d
Author: Spurthi Lokeshappa <spurthi.lokeshappa@intel.com>
Date:   Wed Feb 4 11:18:11 2026 -0800

    Merge branch 'main' into libinta/remove_gather_scatter
    
    Signed-off-by: Spurthi Lokeshappa <spurthi.lokeshappa@intel.com>

[33mcommit 333907d16a5f1b4639b98adb4067adcd0400325b[m[33m ([m[1;31mslokesha/main[m[33m)[m
Author: lkk <33276950+lkk12014402@users.noreply.github.com>
Date:   Thu Feb 5 02:47:47 2026 +0800

    support loading q_scale and using fp8_fused_sdpa for mla prefill. (#909)
    
    1. support loading q_scale
    2. use `fp8_fused_sdpa` op
    
    ---------
    
    Signed-off-by: lkk12014402 <kaokao.lv@intel.com>
    Signed-off-by: lkk <33276950+lkk12014402@users.noreply.github.com>

[33mcommit 4362ee77d0c7226cb8bd52fa2c7d3c5c878d2d75[m
Author: Jakub Byczkowski <jbyczkowski@habana.ai>
Date:   Wed Feb 4 10:00:53 2026 +0100

    Hpu granite 4.0-h small implementation (#897)
    
    Signed-off-by: Jakub Byczkowski <jbyczkowski@habana.ai>
    Signed-off-by: Michał Kuligowski <michal.kuligowski@intel.com>
    Co-authored-by: Michał Kuligowski <michal.kuligowski@intel.com>
    Co-authored-by: Katarzyna Fojcik <kfojcik@habana.ai>
    Co-authored-by: Krzysztof Smusz <ksmusz@habana.ai>
    Co-authored-by: Jozef Mamza <jmamzax@habana.ai>

[33mcommit 723b9601520e5c3ac79ab3549ac427df5445f319[m
Author: Patryk Wolsza <patryk.wolsza@intel.com>
Date:   Wed Feb 4 09:59:52 2026 +0100

    Update compatibility matrix, and refine installation instructions (#920)
    
    Signed-off-by: PatrykWo <patryk.wolsza@intel.com>

[33mcommit e8e7596b103bd435ab1cc06d0a24ad536f93eefb[m
Author: Nir David <nir1.david@intel.com>
Date:   Wed Feb 4 08:34:43 2026 +0200

    [GAUDISW-245785] - Fix measurement config file generation in calibrate_model.sh scripts (#853)
    
    Signed-off-by: Nir David <ndavid@habana.ai>
    Co-authored-by: Iryna Boiko <iryna.boiko@intel.com>

[33mcommit 6376480ec2c6bb9afefec67822f626948a315f47[m
Merge: daeb39b 922a1b8
Author: Spurthi Lokeshappa <spurthi.lokeshappa@intel.com>
Date:   Tue Feb 3 10:46:08 2026 -0800

    Merge branch 'vllm-project:main' into slokesha/Update_qwen_from_v0.14.1

[33mcommit 7b9a1aab5aa94fb6c59f61e5a2c9cba180f82151[m
Merge: 44abc77 922a1b8
Author: Spurthi Lokeshappa <spurthi.lokeshappa@intel.com>
Date:   Tue Feb 3 10:23:51 2026 -0800

    Merge branch 'main' into libinta/remove_gather_scatter

[33mcommit 922a1b8a6fbc0fef642bd5d93f1b613c4a9339e6[m
Author: Tomasz Zielinski <85164140+tzielinski-habana@users.noreply.github.com>
Date:   Tue Feb 3 15:43:54 2026 +0100

    [FIX_FOR_VLLM_LATEST] Fix for hourly KeyError: <PlatformEnum.OOT: 6> (#917)
    
    This pull request introduces support for Habana Gaudi (HPU) devices in
    the FP8 scaled matrix multiplication (scaled_mm) kernels by registering
    new HPU-specific kernel classes. It also enables a previously
    commented-out test for per-tensor scaling with the
    RedHatAI/Meta-Llama-3-8B-Instruct-FP8 model.
    
    **Habana Gaudi FP8 kernel support:**
    - Added `HPUPerTensorTorchFP8ScaledMMLinearKernel` and
    `HPUChannelWiseTorchFP8ScaledMMLinearKernel` classes that extend the
    existing PyTorch FP8 kernel classes and always report as supported,
    enabling FP8 scaled matrix multiplication on HPU devices.
    - Registered these new HPU-specific kernels with the `scaled_mm` kernel
    registry for the `PlatformEnum.OOT` platform, ensuring that the system
    uses the correct kernels on HPU hardware.
    
    **Testing improvements:**
    - Enabled the per-tensor scaling test for the
    RedHatAI/Meta-Llama-3-8B-Instruct-FP8 model by uncommenting and
    activating the test invocation in `ci_gsm8k_tests.sh`.
    
    Signed-off-by: tzielinski-habana <tomasz.zielinski@intel.com>

[33mcommit 95bbd363a1fa0d78c8dc1e157d7c46dd8af16d54[m
Author: Linoy Buchnik <linoybu@gmail.com>
Date:   Tue Feb 3 10:26:14 2026 +0200

    [GAUDISW-245117] add b2b matmul (#770)
    
    Signed-off-by: linoy buchnik <lbuchnik@habana.ai>
    Signed-off-by: Linoy Buchnik <linoybu@gmail.com>
    Co-authored-by: Copilot <175728472+Copilot@users.noreply.github.com>

[33mcommit daeb39bf69ed4307c4bd08ffc0d1e4b9fbf45c47[m
Author: slokesha <spurthi.lokeshappa@intel.com>
Date:   Tue Feb 3 00:00:08 2026 +0000

     Fixed MultiModalprofiler Import failure
    
    Signed-off-by: slokesha <spurthi.lokeshappa@intel.com>

[33mcommit 9aa7f9b33a3e2e06177bc1a708841a4f93258a87[m
Author: Seunghyuk Park (shepark) <seunghyuk.h.park@intel.com>
Date:   Mon Feb 2 00:16:05 2026 -0800

    Update qwen2_5_vl attention forward (#908)
    
    * Prevent cu_seqlens/mask mix-ups that can trigger performance
    regressions or incorrect attention behavior.
    * Remove the lens = (cu_seqlens[1:] - cu_seqlens[:-1]).tolist()
    computation from the Qwen2.5 path.
    
    This calculation is not required for Qwen2.5 and was causing a
    performance regression after PR
    https://github.com/vllm-project/vllm-gaudi/pull/884. Removing it
    restores the previous performance without changing model behavior.

[33mcommit fb9a0f87468ad155f70dcfbbba6346beaa8a4c9d[m
Author: slokesha <spurthi.lokeshappa@intel.com>
Date:   Mon Feb 2 22:36:38 2026 +0000

    Resolved conflict in HPU_model_runner
    
    Signed-off-by: slokesha <spurthi.lokeshappa@intel.com>

[33mcommit acd2563a8307d94accff364f5a4cd58c17f155ee[m
Author: Libin Tang <libin.tang@intel.com>
Date:   Fri Jan 30 00:56:52 2026 -0800

    Qwen3vl accuracy fixes (#884)
    
    for qwen3 vl, there is accuracy issue with multi-images within 1
    request, this PR is to fix that. After fix, there are 3 paths for vision
    attention depending on the images count inside 1 request
    1. single image, use fusedsdpa without attn mask
    3. multi-images with threshold use fusedsdpa without attn_mask one by
    one
    This pr also enables qwen3vl moe
    
    ---------
    
    Signed-off-by: slokesha <slokeshappa@habana.ai>
    Signed-off-by: Jakub Byczkowski <jbyczkowski@habana.ai>
    Signed-off-by: Agata Dobrzyniewicz <adobrzyniewicz@habana.ai>
    Signed-off-by: Radoslaw Smyrek <radoslawx.smyrek@intel.com>
    Signed-off-by: linoy buchnik <lbuchnik@habana.ai>
    Signed-off-by: Iryna Boiko <iboiko@habana.ai>
    Signed-off-by: Artur Fierka <artur.fierka@intel.com>
    Signed-off-by: Luca Calabria <luca.calabria@intel.com>
    Co-authored-by: Seunghyuk Park <separk@habana.ai>
    Co-authored-by: Jakub Byczkowski <jbyczkowski@habana.ai>
    Co-authored-by: Agata Dobrzyniewicz <160237065+adobrzyn@users.noreply.github.com>
    Co-authored-by: Chendi.Xue <chendi.xue@intel.com>
    Co-authored-by: Radosław Smyrek <radoslawx.smyrek@intel.com>
    Co-authored-by: Linoy Buchnik <linoybu@gmail.com>
    Co-authored-by: Iryna Boiko <iboiko@habana.ai>
    Co-authored-by: Artur Fierka <artur.fierka@intel.com>
    Co-authored-by: Luca Calabria <luca.calabria@intel.com>
    Co-authored-by: github-actions[bot] <github-actions[bot]@users.noreply.github.com>
    Co-authored-by: slokesha <slokeshappa@habana.ai>
    Co-authored-by: Seunghyuk Park (shepark) <seunghyuk.h.park@intel.com>
    Co-authored-by: Michał Kuligowski <michal.kuligowski@intel.com>
    Co-authored-by: Katarzyna Fojcik <kfojcik@habana.ai>
    Co-authored-by: Krzysztof Smusz <ksmusz@habana.ai>
    Co-authored-by: Jozef Mamza <jmamzax@habana.ai>

[33mcommit 44abc7737b84c3e1b8994916a103d9a7d612c0ff[m
Merge: 5a44e7f 1e9013b
Author: Spurthi Lokeshappa <spurthi.lokeshappa@intel.com>
Date:   Mon Feb 2 13:23:53 2026 -0800

    Merge branch 'main' into libinta/remove_gather_scatter

[33mcommit 1e9013b66e3431b22b6dd73b1470db3ab0629e36[m[33m ([m[1;31mlibinta/main[m[33m)[m
Author: Soila Kavulya <soila.p.kavulya@intel.com>
Date:   Mon Feb 2 13:01:35 2026 -0800

    Support for modelopt FP8 quantization format for dense models (#890)
    
    Signed-off-by: Soila Kavulya <soila.p.kavulya@intel.com>
    Signed-off-by: Kavulya, Soila P <soila.p.kavulya@intel.com>

[33mcommit a36f7fe742bd8b46f896a4c6b1d002b860c3d7b9[m
Author: Konrad Zawora <kzawora@icloud.com>
Date:   Mon Feb 2 12:21:34 2026 +0100

    yeet myself out of CODEOWNERS (#905)
    
    it's been truly great, but it's time for me to go 🫡
    
    love,
    Konrad
    
    Signed-off-by: Konrad Zawora <kzawora@habana.ai>

[33mcommit 6c13d8adcba50ce102ea1a7817b2106e4da38af0[m
Author: git config -lDudi Lester <160421192+dudilester@users.noreply.github.com>
Date:   Thu Jan 29 17:01:58 2026 +0200

    [GAUDISW-245686] Add dynamic quantization configuration file example (#838)
    
    [GAUDISW-245448] Add dynamic quantization configuration file example
    that includes kv cache
    
    Signed-off-by: Dudi Lester <dlester@habana.ai>
    Co-authored-by: Michał Kuligowski <michal.kuligowski@intel.com>

[33mcommit 3a7195edf1c2fee4db57e1d46be87293c1df5dcd[m
Author: Iryna Boiko <iboiko@habana.ai>
Date:   Thu Jan 29 14:27:51 2026 +0100

    [FIX_FOR_VLLM_LATEST] Refactor for #30623 and small fix for #32238 (#876)
    
    Signed-off-by: Iryna Boiko <iboiko@habana.ai>
    Co-authored-by: Agata Dobrzyniewicz <160237065+adobrzyn@users.noreply.github.com>

[33mcommit 24a98217e8a7d19cf01db952b194fc9532917c61[m
Author: Iryna Boiko <iboiko@habana.ai>
Date:   Thu Jan 29 14:23:50 2026 +0100

    Jenkins CI fix for Mistral (#840)
    
    Signed-off-by: Iryna Boiko <iboiko@habana.ai>
    Co-authored-by: Michał Kuligowski <michal.kuligowski@intel.com>

[33mcommit d6989ceda255f5aa1b2334103d0dd5eb736d6b26[m
Author: Jakub Byczkowski <jbyczkowski@habana.ai>
Date:   Thu Jan 29 05:58:22 2026 +0100

    Correct sliding window enabling
    
    Currently is_interleaved just checks for at least two different layers
    in the model. Sliding window is used when multiple different attention
    layers are present in one model. For Granite 4.0-h small thats an issue
    since we have one type of attention layers and one type of mamba layer.
    Sliding window should be disabled. For Granite 4.0 it's sufficient to
    check if sliding_window size is defined. Proper fix would be to change
    is_interleaved function in the upstream. Alternativelly maybe a
    different function needs to be used to check for sliding window
    enabling.
    
    Signed-off-by: Jakub Byczkowski <jbyczkowski@habana.ai>
    Co-authored-by: Michał Kuligowski <michal.kuligowski@intel.com>

[33mcommit d54f4c258e60904d863d9c4c540885ff443803ce[m
Author: Yeonsil Yoon <yeon.sil.yoon@intel.com>
Date:   Wed Jan 28 07:29:41 2026 -0800

    Enable support for prefill side kv_layout and block_size update (#867)
    
    1. update example to support prefill HND and agreed_block_size
    2. enable prefill side kv_layout and block_size update
    
    Port https://github.com/vllm-project/vllm/pull/30448 to vllm-gaudi
    
    ---------
    
    Signed-off-by: Chendi Xue <chendi.xue@intel.com>
    Signed-off-by: Yeonsil Yoon <yeon.sil.yoon@intel.com>

[33mcommit 5a44e7f34eec8f766193ce9d200a550b6530e38e[m
Merge: 79357b9 6de6d5a
Author: Iryna Boiko <iboiko@habana.ai>
Date:   Wed Jan 28 16:12:38 2026 +0100

    Merge branch 'main' into libinta/remove_gather_scatter

[33mcommit 6de6d5a9de54e2b5b6f2bd671668f9c3fc1f62b2[m
Author: Konrad Zawora <kzawora@habana.ai>
Date:   Wed Jan 28 09:32:35 2026 +0100

    Draft: Add FlashAttention online merge in Unified Attention  (#785)
    
    Further experiments on top of
    https://github.com/vllm-project/vllm-gaudi/pull/784 - I wanted to check
    if we can avoid some OOMs by performing FlashAttention rescaling online
    rather than after computing all the parts - should save us memory on
    some intermediate buffers. Accuracy is surprisingly okay-ish, but I
    haven't tested this too thouroughly.
    
    ---------
    
    Signed-off-by: Konrad Zawora <kzawora@habana.ai>
    Co-authored-by: Agata Dobrzyniewicz <160237065+adobrzyn@users.noreply.github.com>

[33mcommit 60beb0f9cae6eaf9a2039999b62f7aaf13878106[m
Author: Jakub Byczkowski <jbyczkowski@habana.ai>
Date:   Wed Jan 28 08:02:05 2026 +0100

    Implement bucket corrector for Mamba chunk size (#886)
    
    Due to MambaMixer2 implementation requirements, all buckets used for
    mamba must be a multiple of mamba chunk size.
    
    Signed-off-by: Jakub Byczkowski <jbyczkowski@habana.ai>

[33mcommit 4c0e6ff6c542a94a86cac22dfa8d466bf3d0b976[m
Author: Agata Dobrzyniewicz <160237065+adobrzyn@users.noreply.github.com>
Date:   Mon Jan 26 15:36:38 2026 +0100

    Revert "skip HPU graphs for long prefills" (#850)
    
    Reverts vllm-project/vllm-gaudi#780
    
    ---------
    
    Signed-off-by: Agata Dobrzyniewicz <adobrzyniewicz@habana.ai>
    Co-authored-by: Chendi.Xue <chendi.xue@intel.com>

[33mcommit 79357b92f28ef52bac4c095b41ffb22a70a2719f[m
Merge: f46b48d f8dea36
Author: Iryna Boiko <iboiko@habana.ai>
Date:   Mon Jan 26 10:03:08 2026 +0100

    Merge branch 'main' into libinta/remove_gather_scatter

[33mcommit f8dea36f6c6be335a1cbce1b2ca86e5230cf6b34[m
Author: git config -lDudi Lester <160421192+dudilester@users.noreply.github.com>
Date:   Mon Jan 26 10:23:34 2026 +0200

    Fix HPU model runner profile_run to work with dynamic kv-cache scales (#852)
    
    Signed-off-by: Dudi Lester <dlester@habana.ai>
    Co-authored-by: Kamil Kaczor <kamil.kaczor@intel.com>

[33mcommit b689eb99fd8d8ba6ce824dfd96b5fa07d8d7be3e[m
Author: Artur Fierka <artur.fierka@intel.com>
Date:   Mon Jan 26 08:13:14 2026 +0100

    Fix Llama4 shape mismatch for 32k+ context window (#842) (#855)
    
    Llama4 for `max_model_len > 32k` enable temperature adjustment
    https://github.com/vllm-project/vllm/blob/main/vllm/model_executor/models/llama4.py#L719.
    Enabled adjustment causes tensor `q` shape modification from 2D to 3D:
    https://github.com/vllm-project/vllm/blob/main/vllm/model_executor/models/llama4.py#L307.
    This tensor is passing to `UnqnatizedFusedMoEMetod -> forward`:
    https://github.com/vllm-project/vllm-gaudi/blob/main/vllm_gaudi/ops/hpu_fused_moe.py#L163
    causing invalid reshaping - we trying to return a 3D `output.view` based
    on 2D output tensor.
    
    Found that following PR introduced the bug: #680 and #684
    
    Cherry-picked from `releases/v0.13.0`
    
    ---------
    
    Signed-off-by: Artur Fierka <artur.fierka@intel.com>

[33mcommit f46b48db065599597e8ba5f674a1c612ca5cb349[m
Author: Libin Tang <litang@habana.ai>
Date:   Fri Jan 23 09:28:44 2026 -0800

    Update qwen2.5-vl-7b.yaml to revert change

[33mcommit 150cf7abe82a583bbf4d353c1ea22d9003aca7b2[m
Merge: 4cf5cb1 7e97f22
Author: Libin Tang <libin.tang@intel.com>
Date:   Fri Jan 23 09:01:25 2026 -0800

    Merge branch 'main' into libinta/remove_gather_scatter

[33mcommit 7e97f2259667303557b39776fb6e817af2b18d7a[m
Author: Katarzyna Fojcik <kfojcik@habana.ai>
Date:   Fri Jan 23 12:13:40 2026 +0100

    Add support for chunked attention (#821)
    
    Cherry-pick of
    
    https://github.com/vllm-project/vllm-gaudi/commit/6e1be4e0295e79ef260aa4f00411b542aeeed21f
    but adapted to recent changes in
    https://github.com/vllm-project/vllm-gaudi/pull/526
    
    ---------
    
    Signed-off-by: Katarzyna Fojcik <kfojcik@habana.ai>

[33mcommit ce495dfbb5e5325b9c140a3a1cb8ad09a2745e6a[m
Author: Miłosz Grunwald <milosz.grunwald@intel.com>
Date:   Fri Jan 23 11:42:00 2026 +0100

    Remove unused test utils (#864)
    
    Signed-off-by: Milosz Grunwald <milosz.grunwald@intel.com>

[33mcommit 4cf5cb1baaae7f5ba1cb2db8b2fb82b5dc287d6b[m
Author: Libin Tang <libin.tang@intel.com>
Date:   Fri Jan 23 00:00:11 2026 -0800

    precommit fix

[33mcommit ec827b8a93383c410b26b2eeb20db44787bfdfa4[m
Author: Libin Tang <libin.tang@intel.com>
Date:   Thu Jan 22 23:52:39 2026 -0800

    add more mm bucket

[33mcommit f0613fdce6666c495c12c2467a0152823e8c3ab5[m
Author: Libin Tang <libin.tang@intel.com>
Date:   Thu Jan 22 17:29:51 2026 -0800

    precommit fix

[33mcommit 091c5fe3d7bc575339ae223461e355d9cc139184[m
Author: Libin Tang <libin.tang@intel.com>
Date:   Thu Jan 22 17:21:45 2026 -0800

    precommit fix

[33mcommit 913176ae052158d496da2d775c418b85a6204d93[m
Author: Libin Tang <libin.tang@intel.com>
Date:   Thu Jan 22 16:58:54 2026 -0800

    fix qwen2.5vl unified attn test failure

[33mcommit 90971645e11cc1bd2abdb40aeb99a23bfc80ac4b[m
Merge: 7757e80 b7c6409
Author: Libin Tang <litang@habana.ai>
Date:   Thu Jan 22 08:03:56 2026 -0800

    Merge branch 'main' into libinta/remove_gather_scatter

[33mcommit b7c640994f64076fecf173ba3648256c87949b64[m
Author: Xinyu Chen <xinyu1.chen@intel.com>
Date:   Thu Jan 22 22:52:30 2026 +0800

    DP: Fix for torch.compile (#722)
    
    Signed-off-by: Xinyu Chen <xinyu1.chen@intel.com>
    Co-authored-by: Yaser Afshar <yaser.afshar@intel.com>
    Co-authored-by: Chendi.Xue <chendi.xue@intel.com>

[33mcommit d453ed94e2d88f8e6bddb6f643fb11c4537a4ca2[m
Author: Tanner Voas <tanner.voas@intel.com>
Date:   Thu Jan 22 20:52:19 2026 +0800

    Resolve qwen25 vl accuracy regression (#831)
    
    ## Motivation
    Qwen2.5-VL models have lower accuracy than expected, and this accuracy
    regressed due to PR #698 (commit
    18105cc511a3db222363797465ef2732a7306442 on main). This PR introduces
    too changes to boost accuracy on Qwen2.5-VL-7B-Instruct on MMMU dataset
    from ~42% to 51%. The accuracy matches that seen on GPU version of vLLM
    (build 0.13.0) under similar test conditions.
    
    ## Changes
    - First change is a fix for the regression. The attn_mask was not being
    used in HPUQwen2_5_VisionBlock.
    - The second change is enabling fp32_softmax for qwen2_5_vl models.
    
    ---------
    
    Signed-off-by: Tanner Voas <tanner.voas@intel.com>

[33mcommit 7757e80d6af1d2f7150f57ec448d42aa3fa7c995[m
Author: Libin Tang <litang@habana.ai>
Date:   Thu Jan 22 00:01:57 2026 -0800

    Update hpu_model_runner.py for precommit fix

[33mcommit 3dd1f5cf3d7cd7836c3994ddd794c686b356545a[m
Author: Libin Tang <litang@habana.ai>
Date:   Wed Jan 21 23:59:18 2026 -0800

    Update hpu_model_runner.py for precommit fix

[33mcommit 02c239bff07cae16c203aef2c10c8682560fe33a[m
Author: Libin Tang <litang@habana.ai>
Date:   Wed Jan 21 23:54:00 2026 -0800

    Update hpu_model_runner.py for precommit fix

[33mcommit b4f2e6c3933406080c0df304bee4886796463dd7[m
Author: Libin Tang <litang@habana.ai>
Date:   Wed Jan 21 23:46:50 2026 -0800

    Update hpu_model_runner.py for precommit fix

[33mcommit 3ff7e804b7e1863c7dcd079473bbba36232c4aad[m
Merge: 9be0056 9ce14a2
Author: Libin Tang <litang@habana.ai>
Date:   Wed Jan 21 23:40:35 2026 -0800

    Merge branch 'main' into libinta/remove_gather_scatter

[33mcommit 9be005620c376f299439e41f2d65352c52fcb06e[m
Author: Libin Tang <litang@habana.ai>
Date:   Wed Jan 21 23:39:57 2026 -0800

    Update hpu_model_runner.py for precommit fix

[33mcommit 9ce14a24daf0f68177a8a396b9ff183d32fe7cb5[m
Author: Iryna Boiko <iboiko@habana.ai>
Date:   Thu Jan 22 08:36:31 2026 +0100

    Fix for #32077 (#851)
    
    Signed-off-by: Iryna Boiko <iboiko@habana.ai>
    Co-authored-by: Chendi.Xue <chendi.xue@intel.com>

[33mcommit 9db6b783ca119ff4b047a7210d42891a86ea3291[m
Author: Libin Tang <litang@habana.ai>
Date:   Wed Jan 21 11:20:00 2026 -0800

    Update ops.py with removing uncessary change

[33mcommit 07f40c9170b10d0418f3b3bfb60cd4c308869238[m
Author: Libin Tang <libin.tang@intel.com>
Date:   Wed Jan 21 08:59:03 2026 -0800

    add back warmup with ratio and video warmup

[33mcommit 5fdf23721e72dff157baac386b29ac54eeac5ebf[m
Merge: 0df1f20 2d1a7a7
Author: Libin Tang <litang@habana.ai>
Date:   Wed Jan 21 07:21:47 2026 -0800

    Merge branch 'main' into libinta/remove_gather_scatter

[33mcommit 2d1a7a7a4e080e710f8cbe3dfa418e57a1480d49[m
Author: Jakub Sochacki <97886316+jakub-sochacki@users.noreply.github.com>
Date:   Wed Jan 21 16:20:00 2026 +0100

    KV cache sharing for HPU (#834)
    
    Adds support for cross-layer KV cache sharing on HPU, enabling models
    like Gemma-3n that share KV cache between layers to run on Gaudi.
    
    **Changes**
    - hpu_attn.py: Store kv_sharing_target_layer_name and skip KV cache
    writes for sharing layers
    - hpu_model_runner.py: Track shared layers, validate config, and set up
    tensor sharing during initialization
    - test_hpu_model_runner.py: Enable KV sharing unit tests
    
    **Expected Benefits**
    Reduced KV cache memory usage for models with layer sharing
    Lower TTFT for long-context scenarios in supported models (e.g.,
    Gemma-3n)
    
    **Testing**
    Unit tests pass
    E2E validation with a KV-sharing model (e.g., Gemma-3n) pending
    
    ---------
    
    Signed-off-by: jakub-sochacki <jakub.sochacki@intel.com>
    Co-authored-by: jakub-sochacki <jakub.sochacki@intel.com>

[33mcommit 3108ed8241828ed91dcc3654602d3f079ae98fb1[m
Author: Linoy Buchnik <linoybu@gmail.com>
Date:   Wed Jan 21 15:07:55 2026 +0200

    [GAUDISW-245665] fix diverge from vllm in multiModalBudget (#837)
    
    Signed-off-by: linoy buchnik <lbuchnik@habana.ai>
    Signed-off-by: Iryna Boiko <iboiko@habana.ai>
    Co-authored-by: Iryna Boiko <iboiko@habana.ai>

[33mcommit 0df1f2021a668022e9ab0e49bea5c3f7f61ea4ab[m
Merge: e370a49 0321b6c
Author: Iryna Boiko <iboiko@habana.ai>
Date:   Wed Jan 21 10:33:28 2026 +0100

    Merge branch 'main' into libinta/remove_gather_scatter

[33mcommit 0321b6c65c360f447d4af45c1eaa3d452873bbf5[m
Author: Radosław Smyrek <radoslawx.smyrek@intel.com>
Date:   Wed Jan 21 10:17:30 2026 +0100

    Interleaved sliding window fix (#805)
    
    Following reasoning stated in PR:
    https://github.com/vllm-project/vllm-gaudi/pull/616
    
    Signed-off-by: Radoslaw Smyrek <radoslawx.smyrek@intel.com>

[33mcommit e370a49d5e969df49403e37c25f24abdd522fce6[m
Merge: 79d90a4 f8cb8d2
Author: Iryna Boiko <iboiko@habana.ai>
Date:   Wed Jan 21 09:33:29 2026 +0100

    Merge branch 'main' into libinta/remove_gather_scatter

[33mcommit f8cb8d28b1f18800616c62006cfedbfcf43e68c9[m
Author: Spurthi Lokeshappa <spurthi.lokeshappa@intel.com>
Date:   Wed Jan 21 00:33:00 2026 -0800

    Added Qwen3 Test (#736)
    
    Adds  Qwen3 model test case for image
    
    Signed-off-by: slokesha <slokeshappa@habana.ai>
    Co-authored-by: Iryna Boiko <iboiko@habana.ai>

[33mcommit 472ed884dfea6a370c152dfafa3bbb054e320e43[m
Author: git config -lDudi Lester <160421192+dudilester@users.noreply.github.com>
Date:   Tue Jan 20 14:03:52 2026 +0200

    [GAUDISW-244752] add dynamic scale for V-Cache on Hiddden dim (#749)
    
    Signed-off-by: Dudi Lester <dlester@habana.ai>

[33mcommit 7c5fd58c39c4342548d0aed8a2dda30a0205824e[m
Author: Patryk Wolsza <patryk.wolsza@intel.com>
Date:   Tue Jan 20 12:47:25 2026 +0100

    Update configurations for Bielik-4.5B model integration  (#804)
    
    The PR adds tuned and validated configurations for the Bielik family
    model and integrates to Docker functionality.
    
    ---------
    
    Signed-off-by: PatrykWo <patryk.wolsza@intel.com>

[33mcommit fc96a30c285a125def8f8d53eb551f6979eec3b1[m
Author: Youlei Yang <youlei.yang@intel.com>
Date:   Tue Jan 20 16:59:26 2026 +0800

    fix calibration for fp8 MoE models (#832)
    
    - remove arg `quantization` for step-2 and step-4.
    - pass `--expert-parallel` if `-u` passed.
    - no arg after `-u` flag.
    
    ---------
    
    Signed-off-by: Youlei Yang <youlei.yang@intel.com>

[33mcommit 4bbe529df9508cf07aa5c024297933f1c8e2d9b4[m
Author: Youlei Yang <youlei.yang@intel.com>
Date:   Tue Jan 20 15:19:59 2026 +0800

    fix empty buckets issue for enforce eager mode (#761)
    
    Current implementation skip calling `self.model_runner.warmup_model()`
    for `enforce_eager=True` leads to empty bucket lists in the bucket
    manager, and the following `find_bucket` calls will get fallback
    buckets.
    
    The buckets are generated by the following calls in
    `self.model_runner.warmup_model()`.
    
    https://github.com/vllm-project/vllm-gaudi/blob/cc37f1f221ecb2733d02f3c3f138cc697f0acaac/vllm_gaudi/v1/worker/hpu_model_runner.py#L4581-L4596
    
    And the actual warmup will be skipped for `enforce_eager=True` according
    to
    
    https://github.com/vllm-project/vllm-gaudi/blob/cc37f1f221ecb2733d02f3c3f138cc697f0acaac/vllm_gaudi/v1/worker/hpu_model_runner.py#L4670-L4695
    
    So the `self.model_runner.warmup_model()` cannot be skipped when
    `enforce_eager=True` and no actual warmup in this case as expected.
    
    Signed-off-by: Youlei Yang <youlei.yang@intel.com>

[33mcommit a280ae9e7cb98322b018502eb33e130554e05677[m
Author: Patryk Wolsza <patryk.wolsza@intel.com>
Date:   Tue Jan 20 01:16:10 2026 +0100

    Add conditional runner selection based on PR title for discover_runne… (#841)
    
    …r job
    
    The purpose of this PR is to ensure that PRs containing
    [FIX_FOR_VLLM_LATEST] in the title are executed on the node where the
    hourly runs take place.
    
    Signed-off-by: PatrykWo <patryk.wolsza@intel.com>

[33mcommit 79d90a416bdf487676cf78ebe4906d725dbe4c01[m
Author: Libin Tang <litang@habana.ai>
Date:   Mon Jan 19 14:22:46 2026 -0800

    Update qwen3_vl.py for precommit fix

[33mcommit 4089adff19fc67b650bfa0bd06642a10b3780d3f[m
Author: Libin Tang <litang@habana.ai>
Date:   Mon Jan 19 14:06:50 2026 -0800

    Update qwen3_vl.py for precommit fix

[33mcommit 46facad06aa8cbc5dcdd31af1b76d719436b499d[m
Merge: e23e6d2 1770639
Author: Libin Tang <litang@habana.ai>
Date:   Mon Jan 19 14:00:41 2026 -0800

    Merge branch 'main' into libinta/remove_gather_scatter
    
    Signed-off-by: Libin Tang <litang@habana.ai>

[33mcommit 17706394744ad9128f76e4e95997b06084e518dd[m
Author: Agata Dobrzyniewicz <160237065+adobrzyn@users.noreply.github.com>
Date:   Mon Jan 19 12:48:46 2026 +0100

    Exponential max decode blocks fix for non-contiguous pa scenario (#818)
    
    Signed-off-by: Agata Dobrzyniewicz <adobrzyniewicz@habana.ai>
    Signed-off-by: Agata Dobrzyniewicz <160237065+adobrzyn@users.noreply.github.com>
    Co-authored-by: Iryna Boiko <iboiko@habana.ai>

[33mcommit 7011e318e8185a5d77aa255e75f6fbd61fff4637[m
Author: Spurthi Lokeshappa <spurthi.lokeshappa@intel.com>
Date:   Mon Jan 19 01:37:07 2026 -0800

    Enable HPU Fused SDPA for Qwen3-VL vision attention using attention masks (#787)
    
    Qwen3-VL vision attention is updated to use FusedSDPA.apply directly
    when the query sequence length is within the supported fused range
    (q_len ≤ 65536).
    This removes the per-block Q/K/V attention loop and enables the
    optimized HPU fused SDPA kernel for vision attention.
    
    The change aligns Qwen3-VL with the optimized path already used by
    Qwen2.5-VL on Gaudi, improving efficiency while preserving identical
    model outputs.
    
    ---------
    
    Signed-off-by: slokesha <slokeshappa@habana.ai>
    Signed-off-by: Spurthi Lokeshappa <slokeshappa@habana.ai>
    Signed-off-by: slokesha <spurthi.lokeshappa@intel.com>

[33mcommit a823690b35733984570e551c9da6888b1a2530a3[m
Author: Yi Liu <yi4.liu@intel.com>
Date:   Mon Jan 19 16:48:01 2026 +0800

    Fix INC patch for new version (#829)
    
    The issue was fixed in synpase 1.24(INC 3.6), and the
    `supported_dynamic_ops ` was removed.
    
    ---------
    
    Signed-off-by: yiliu30 <yi4.liu@intel.com>

[33mcommit 97841322c9d46e1623c1c5882d19aae5b7a176f5[m
Author: Tanner Voas <tanner.voas@intel.com>
Date:   Mon Jan 19 15:57:54 2026 +0800

    Resolve crash when using caching with mm models (#823)
    
    ## Motivation
    scheduler_output.scheduled_encoder_inputs is not encoding all the
    features of the scheduled requests, resulting in RTEs when we attempt to
    gather the encoded MM features and encounter a cache miss.
    
    ## Changes
    Rather than checking scheduler_output.scheduled_encoder_inputs we ensure
    all the features are cached (in _execute_mm_encoder) as this is what the
    following function (_gather_mm_embeddings) expects. This resolves the
    encountered crash with no noticeable decrease in performance or accuracy
    (evaluated on MMMU dataset).
    
    ----
    Below is the error that is encountered without this change.
    
    ```bash
    (EngineCore_DP0 pid=262909) ERROR 01-13 14:23:52 [dump_input.py:81] Dumping scheduler stats: SchedulerStats(num_running_reqs=79, num_waiting_reqs=19, step_counter=0, current_wave=0, kv_cache_usage=0.013820035543915421, prefix_cache_stats=PrefixCacheStats(reset=False, requests=31, queries=16549, hits=0, preempted_requests=0, preempted_queries=0, preempted_hits=0), connector_prefix_cache_stats=None, kv_cache_eviction_events=[], spec_decoding_stats=None, kv_connector_stats=None, waiting_lora_adapters={}, running_lora_adapters={}, cudagraph_stats=None, perf_stats=None)
    (EngineCore_DP0 pid=262909) ERROR 01-13 14:23:52 [core.py:881] EngineCore encountered a fatal error.
    (EngineCore_DP0 pid=262909) ERROR 01-13 14:23:52 [core.py:881] Traceback (most recent call last):
    (EngineCore_DP0 pid=262909) ERROR 01-13 14:23:52 [core.py:881]   File "/workspace/vllm-project/vllm/v1/engine/core.py", line 872, in run_engine_core
    (EngineCore_DP0 pid=262909) ERROR 01-13 14:23:52 [core.py:881]     engine_core.run_busy_loop()
    (EngineCore_DP0 pid=262909) ERROR 01-13 14:23:52 [core.py:881]   File "/workspace/vllm-project/vllm/v1/engine/core.py", line 899, in run_busy_loop
    (EngineCore_DP0 pid=262909) ERROR 01-13 14:23:52 [core.py:881]     self._process_engine_step()
    (EngineCore_DP0 pid=262909) ERROR 01-13 14:23:52 [core.py:881]   File "/workspace/vllm-project/vllm/v1/engine/core.py", line 932, in _process_engine_step
    (EngineCore_DP0 pid=262909) ERROR 01-13 14:23:52 [core.py:881]     outputs, model_executed = self.step_fn()
    (EngineCore_DP0 pid=262909) ERROR 01-13 14:23:52 [core.py:881]                               ^^^^^^^^^^^^^^
    (EngineCore_DP0 pid=262909) ERROR 01-13 14:23:52 [core.py:881]   File "/workspace/vllm-project/vllm/v1/engine/core.py", line 462, in step_with_batch_queue
    (EngineCore_DP0 pid=262909) ERROR 01-13 14:23:52 [core.py:881]     model_output = future.result()
    (EngineCore_DP0 pid=262909) ERROR 01-13 14:23:52 [core.py:881]                    ^^^^^^^^^^^^^^^
    (EngineCore_DP0 pid=262909) ERROR 01-13 14:23:52 [core.py:881]   File "/workspace/vllm-project/vllm/v1/executor/multiproc_executor.py", line 80, in result
    (EngineCore_DP0 pid=262909) ERROR 01-13 14:23:52 [core.py:881]     return super().result()
    (EngineCore_DP0 pid=262909) ERROR 01-13 14:23:52 [core.py:881]            ^^^^^^^^^^^^^^^^
    (EngineCore_DP0 pid=262909) ERROR 01-13 14:23:52 [core.py:881]   File "/usr/lib/python3.12/concurrent/futures/_base.py", line 449, in result
    (EngineCore_DP0 pid=262909) ERROR 01-13 14:23:52 [core.py:881]     return self.__get_result()
    (EngineCore_DP0 pid=262909) ERROR 01-13 14:23:52 [core.py:881]            ^^^^^^^^^^^^^^^^^^^
    (EngineCore_DP0 pid=262909) ERROR 01-13 14:23:52 [core.py:881]   File "/usr/lib/python3.12/concurrent/futures/_base.py", line 401, in __get_result
    (EngineCore_DP0 pid=262909) ERROR 01-13 14:23:52 [core.py:881]     raise self._exception
    (EngineCore_DP0 pid=262909) ERROR 01-13 14:23:52 [core.py:881]   File "/workspace/vllm-project/vllm/v1/executor/multiproc_executor.py", line 84, in wait_for_response
    (EngineCore_DP0 pid=262909) ERROR 01-13 14:23:52 [core.py:881]     response = self.aggregate(get_response())
    (EngineCore_DP0 pid=262909) ERROR 01-13 14:23:52 [core.py:881]                               ^^^^^^^^^^^^^^
    (EngineCore_DP0 pid=262909) ERROR 01-13 14:23:52 [core.py:881]   File "/workspace/vllm-project/vllm/v1/executor/multiproc_executor.py", line 342, in get_response
    (EngineCore_DP0 pid=262909) ERROR 01-13 14:23:52 [core.py:881]     raise RuntimeError(
    (EngineCore_DP0 pid=262909) ERROR 01-13 14:23:52 [core.py:881] RuntimeError: Worker failed with error 'Encoder cache miss for 9def2746268d93d5b729bcf246ccb1ddfa4af2391dff4854a97c2c5c99e7967b.', please check the stack trace above for the root cause
    ```
    
    Signed-off-by: Tanner Voas <tanner.voas@intel.com>

[33mcommit 7a9d05d219ab98ba4b624975623f2209e99de496[m
Author: iLeGend <youzhi.jin@intel.com>
Date:   Sat Jan 17 00:03:56 2026 +0800

    Fix dummy_mm_item TypeError when warmup MM model  (#822)
    
    Failed in multimodal model init because `dummy_mm_data['image'][0] `
    need be `MultiModalKwargsItem `
    
    
    vllm serve Qwen/Qwen2.5-VL-7B-Instruct --trust-remote-code
    
    ```shell
    (EngineCore_DP0 pid=743478) ERROR 01-15 06:22:21 [core.py:936]   File "/mnt/ceph1/youzhi/vllm-gaudi/vllm_gaudi/v1/worker/hpu_model_runner.py", line 4750, in warmup_model
    (EngineCore_DP0 pid=743478) ERROR 01-15 06:22:21 [core.py:936]     self.warmup_multimodal_graphs(self.get_model().vision_bucket_manager.multimodal_buckets)
    (EngineCore_DP0 pid=743478) ERROR 01-15 06:22:21 [core.py:936]   File "/mnt/ceph1/youzhi/vllm-gaudi/vllm_gaudi/v1/worker/hpu_model_runner.py", line 4633, in warmup_multimodal_graphs
    (EngineCore_DP0 pid=743478) ERROR 01-15 06:22:21 [core.py:936]     batched_dummy_mm_inputs = self._get_mm_dummy_batch(modality, img_arg, ratio_w, ratio_h)
    (EngineCore_DP0 pid=743478) ERROR 01-15 06:22:21 [core.py:936]   File "/mnt/ceph1/youzhi/vllm-gaudi/vllm_gaudi/v1/worker/hpu_model_runner.py", line 4596, in _get_mm_dummy_batch
    (EngineCore_DP0 pid=743478) ERROR 01-15 06:22:21 [core.py:936]     dummy_mm_item = dummy_mm_data['image'][0]
    (EngineCore_DP0 pid=743478) ERROR 01-15 06:22:21 [core.py:936] TypeError: 'ProcessorInputs' object is not subscriptable
    ```
    @iboiko-habana could you review or give some help? thx
    
    Signed-off-by: Jin, Youzhi <youzhi.jin@intel.com>

[33mcommit efe5f7885722407600addfee3375c5623ef309d6[m
Author: Monika Helfer <monika.helfer@intel.com>
Date:   Fri Jan 16 14:51:45 2026 +0100

    Doc updates cherry-picked from 0.13.0 (#799)
    
    Signed-off-by: mhelf-intel <monika.helfer@intel.com>
    Signed-off-by: PatrykWo <patryk.wolsza@intel.com>
    Co-authored-by: PatrykWo <patryk.wolsza@intel.com>

[33mcommit f2de16963546829fc62e84246162bc4bd2dcd1ad[m
Author: Patryk Wolsza <patryk.wolsza@intel.com>
Date:   Fri Jan 16 14:50:57 2026 +0100

    fix ubi docker: use --nobest flag to resolve boost dependency conflic… (#810)
    
    …ts (#803)
    
    Switched from --best to --nobest in DNF install command to allow
    installation of compatible boost versions when the latest version has
    dependency conflicts with boost-program-options.
    
    Signed-off-by: Adam Ghandoura <adam.ghandoura@intel.com>
    Co-authored-by: Adam Ghandoura <adam.ghandoura@intel.com>

[33mcommit e23e6d2077b4f86645b78859f6b265be4726b7ad[m
Author: Libin Tang <litang@habana.ai>
Date:   Thu Jan 15 17:41:30 2026 -0800

    Update hpu_model_runner.py to match with upstream for MultiModalBudget

[33mcommit 40d7635620c3d149e3187c5ceae13d5348024011[m
Author: Libin Tang <litang@habana.ai>
Date:   Thu Jan 15 15:06:09 2026 -0800

    Update interfaces.py for precommit fix

[33mcommit db1054888d60227cae0ed42d881cd7c3c031d397[m
Author: Libin Tang <libin.tang@intel.com>
Date:   Thu Jan 15 11:36:14 2026 -0800

    fix precommit issue

[33mcommit 017164137517c477acc02d075df5c97529acd5d0[m
Merge: 48a96db dbb090c
Author: Iryna Boiko <iboiko@habana.ai>
Date:   Thu Jan 15 20:29:43 2026 +0100

    Merge branch 'main' into libinta/remove_gather_scatter

[33mcommit dbb090ca0f7ff46ae5d8c98f46e07c28cced916e[m
Author: Iryna Boiko <iboiko@habana.ai>
Date:   Thu Jan 15 20:27:44 2026 +0100

    disable async scheduler when spec decode is on for hpu_model_runner (#825)
    
    @xuechendi fix
    
    Signed-off-by: Iryna Boiko <iboiko@habana.ai>

[33mcommit 48a96db5280cb4d1413173694cf3ce274dc5c55e[m
Author: Libin Tang <libin.tang@intel.com>
Date:   Thu Jan 15 11:26:53 2026 -0800

    fix test failure

[33mcommit a7ad06eb03efd85e239a6aa0b7d6d18d9ff6671b[m
Author: Paweł Olejniczak <polejniczakx@habana.ai>
Date:   Thu Jan 15 14:19:17 2026 +0100

    Use actual block count for bucketing in contiguous PA mode (#792)
    
    Fix block_bucket_size calculation to use max(block_list) + 1 instead of
    max(actual_need, tensor_size) in contiguous PA mode. This aligns runtime
    bucket selection with warmup bucket generation, preventing "not
    warmed-up" errors and runtime bucket compilation.
    
    Signed-off-by: Paweł Olejniczak <polejniczakx@habana.ai>
    Co-authored-by: Iryna Boiko <iboiko@habana.ai>

[33mcommit 2f060aca3c72e9ba2c1d6158d4928f1adc8b1a23[m
Author: Patryk Wolsza <patryk.wolsza@intel.com>
Date:   Thu Jan 15 12:54:45 2026 +0100

    Update CODEOWNERS (#808)
    
    Signed-off-by: Patryk Wolsza <patryk.wolsza@intel.com>

[33mcommit 6502061059ecca9b94a92c254570b48ab4864f2f[m
Merge: a394b9a 6fe8e49
Author: Iryna Boiko <iboiko@habana.ai>
Date:   Thu Jan 15 10:11:20 2026 +0100

    Merge branch 'main' into libinta/remove_gather_scatter

[33mcommit 6fe8e49817f61ee640ff01307162c18dc57f29da[m
Author: Iryna Boiko <iboiko@habana.ai>
Date:   Thu Jan 15 10:01:40 2026 +0100

    Upgrade transformers>= 4.56.0, <5 (#767)
    
    Signed-off-by: Iryna Boiko <iboiko@habana.ai>

[33mcommit a40b0902a5b74df2571dd6873ff349583274415a[m
Author: Xiaochang Wu <xiaochang.wu@intel.com>
Date:   Thu Jan 15 14:44:03 2026 +0800

    Implement profile_run method in HPU model runner (#775)
    
    - Add comprehensive profile_run implementation to replace placeholder
    - Setup dummy KV caches using bind_kv_cache for proper memory
    initialization
    - Use existing _prepare_dummy_scenario infrastructure for profiling
    - Support unified attention
    
    ---------
    
    Signed-off-by: Xiaochang Wu <xiaochang.wu@intel.com>
    Co-authored-by: Copilot <175728472+Copilot@users.noreply.github.com>
    Co-authored-by: Iryna Boiko <iboiko@habana.ai>

[33mcommit 38558ede14d740c4f02fa7f8b2f356a504743864[m
Author: Agata Dobrzyniewicz <160237065+adobrzyn@users.noreply.github.com>
Date:   Wed Jan 14 21:58:48 2026 +0100

    No num seqs over max in fallback buckets (#816)
    
    Signed-off-by: Agata Dobrzyniewicz <adobrzyniewicz@habana.ai>
    Co-authored-by: Iryna Boiko <iboiko@habana.ai>

[33mcommit 383a43c4dc54004fc9b944b6bb2c0ae94c9173fc[m
Author: Iryna Boiko <iboiko@habana.ai>
Date:   Wed Jan 14 16:49:21 2026 +0100

    [FIX_FOR_VLLM_LATEST] fixes for #31747, #30519, #32003, #31916 and test cases disablement for #31998 and #32254 (#797)
    
    Signed-off-by: Iryna Boiko <iboiko@habana.ai>

[33mcommit daa84f871790f5a8681c2c12e1218dcdd6acf629[m
Author: Yi Liu <yi4.liu@intel.com>
Date:   Wed Jan 14 21:49:05 2026 +0800

    Add `MoeMatmul` to dynamic op support list (#817)
    
    Signed-off-by: yiliu30 <yi4.liu@intel.com>

[33mcommit 91d11d6c97976e56b72c6fd67f5f416e7b7a28cc[m[33m ([m[1;33mtag: [m[1;33mv0.14.0+test[m[33m)[m
Author: Shiv Kaul <shiv.kaul@intel.com>
Date:   Tue Jan 13 00:58:13 2026 -0800

    modify conv3d permute (#794)
    
    Switch order of dimensions in hpu conv3d to allow Qwen2.5-VL in
    torch.compile mode
    
    Signed-off-by: Shiv Kaul <shiv.kaul@intel.com>

[33mcommit 9ca98eac535ee24862e0005e1cc52a1c6562903e[m
Author: Patryk Wolsza <patryk.wolsza@intel.com>
Date:   Tue Jan 13 09:19:10 2026 +0100

    Update Dockerfiles and workflows for v1.23.0 release, including PyTor… (#802)
    
    …ch version bump to 2.9.0
    
    Signed-off-by: PatrykWo <patryk.wolsza@intel.com>

[33mcommit 872795d5803058d08d1290aa5c2f9192f152164c[m
Author: hlin99 <tony.lin@intel.com>
Date:   Tue Jan 13 15:15:42 2026 +0800

    Prefill batching logic to handle chunked prefill/prefix caching for HPU (#753)
    
    Logic to handle chunked prefill/prefix caching for HPU
    Due to HPU padding constraints, batching requests with existing
    history (ctx != 0) causes excessive memory usage, as the entire
    batch must be padded to the longest context, leading to OOM.
    
    This patch enforces a batch size of 1 for prefill operations when
    ctx != 0. Although this sacrifices some throughput in corenr cases,
    it effectively eliminates the OOM risk.
    
    Signed-off-by: Tony Lin <tony.lin@intel.com>

[33mcommit 1d8a8555befb541f1c06b5f0f6de9dd0902cc02c[m
Author: Konrad Zawora <kzawora@habana.ai>
Date:   Mon Jan 12 11:32:23 2026 +0100

    Unified Attention - multi-step low-level profiling (#791)
    
    This PR adds low-level profiling capabilities to unified attention. It
    works similarly to non-unified attention profiling
    (`VLLM_PROFILE_PROMPT` & `VLLM_PROFILE_DECODE` env vars), but now you
    can use `VLLM_PROFILE_UNIFIED` to pass unified buckets you want to
    profile - plural, because now chaining buckets works fine. You can pass
    `VLLM_PROFILE_UNIFIED=512,128,384,1` or something like
    `VLLM_PROFILE_UNIFIED="[(512,64,384,1),(512,128,384,1)]"` to profile
    execution of two or more buckets sequentially. I implemented this
    because I needed to profile memory reuse across buckets and it was super
    hard to do on actual benchmarks.
    
    ---------
    
    Signed-off-by: Konrad Zawora <kzawora@habana.ai>

[33mcommit c885554b2acfebefe1250b3dc094683e36c5a7d1[m
Author: Daniel Huang <pilotflyer824@gmail.com>
Date:   Fri Jan 9 17:10:46 2026 -0800

    Add ucx test (#711)
    
    Adds pd nixl UCX test to CI. This is dependent on
    https://github.com/intel-staging/ucx/pull/1
    
    ---------
    
    Signed-off-by: Daniel Huang <daniel1.huang@intel.com>

[33mcommit fad27f3603985fc948c8c13d0113eb01624765a4[m
Author: Neelesh Gokhale <neelesh.gokhale@intel.com>
Date:   Fri Jan 9 20:48:13 2026 +0530

    Fix Mixtral 8x22B benchmark error, Add EXTRA_BENCH_ARGS (#796)
    
    Fixes issue "AttributeError: 'MistralTokenizer' object has no attribute
    'chat_template'" with mixtral 8x22 benchmarking.
    Add EXTRA_BENCH_ARGS to bench similar to serve.
    
    Signed-off-by: Neelesh Gokhale <neelesh.gokhale@intel.com>

[33mcommit 4d05a35ed30dc20730d93bea658897821f7c9a3b[m
Author: Agata Dobrzyniewicz <160237065+adobrzyn@users.noreply.github.com>
Date:   Fri Jan 9 15:04:00 2026 +0100

    Exponential max number in range not over bmax (#795)
    
    Signed-off-by: Agata Dobrzyniewicz <adobrzyniewicz@habana.ai>
    Signed-off-by: Agata Dobrzyniewicz <160237065+adobrzyn@users.noreply.github.com>
    Co-authored-by: Copilot <175728472+Copilot@users.noreply.github.com>

[33mcommit 5c608a6f4accd02f51ca0563830410f9f2282f82[m
Author: Vidya Galli <vidya.s.galli@intel.com>
Date:   Fri Jan 9 00:39:31 2026 -0800

    Fix for Llama4 static quantization (#707)
    
    Copy of approved pull #430 but with signed commit
    Same as vllm-hpu-extension PR
    https://github.com/HabanaAI/vllm-hpu-extension/pull/329
    
    Command
    ```
    PT_HPU_LAZY_MODE=0 ./calibrate_model.sh \
    -m meta-llama/Llama-4-Maverick-17B-128E-Instruct \
    -d <>/mlperf_inference/llama2/processed-data.pkl \
    -o /eager_output  -b 128 -t 8 -l 4096
    ```
    
    Error without this fix
    ```
    1/4 Preparing calibration dataset
    Calling add_step_closure function does not have any effect. It's lazy mode only functionality. (warning logged once)
    Calling mark_step function does not have any effect. It's lazy mode only functionality. (warning logged once)
    Calling iter_mark_step function does not have any effect. It's lazy mode only functionality. (warning logged once)
    Loading source dataset: /mnt/weka/data/mlperf_inference/llama2/processed-data.pkl
    Creating calibration dataset...
    Traceback (most recent call last):
      File "/root/work/calibration/step-1-prepare-calibration-dataset.py", line 93, in <module>
        main(args)
      File "/root/work/calibration/step-1-prepare-calibration-dataset.py", line 61, in main
        tmp_input = tokenizer.apply_chat_template(tmp_conversation,
                    ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
    AttributeError:
    ```
    
    Signed-off-by: Vidya Galli <vidya.s.galli@intel.com>

[33mcommit 705284768c80435d70f3557c9b17cc4ad5cbd26e[m
Author: Monika Helfer <monika.helfer@intel.com>
Date:   Fri Jan 9 08:37:13 2026 +0100

    Fix the docker image path (#691)
    
    Fixed the docker image path and replaced the actual Gaudi version in the
    FAQ document with a variable to avoid manual editing with each release.
    
    ---------
    
    Signed-off-by: mhelf-intel <monika.helfer@intel.com>
    Co-authored-by: Patryk Wolsza <patryk.wolsza@intel.com>
    Co-authored-by: Michał Kuligowski <michal.kuligowski@intel.com>

[33mcommit a394b9a0cc3557abfd991d401d2b4d8e798f510d[m
Merge: 8a9efd1 fe069da
Author: Libin Tang <libin.tang@intel.com>
Date:   Thu Jan 8 15:14:10 2026 -0800

    Merge branch 'main' into libinta/remove_gather_scatter

[33mcommit fe069daf7f5da7ab12e15dfcace3371188e65495[m
Author: Shiv Kaul <shiv.kaul@intel.com>
Date:   Thu Jan 8 03:25:10 2026 -0800

    create HPUConv3D class, which replaces unfold with view. (#786)
    
    Replace unfold with view in Conv3dLayer, as it fallbacks to cpu
    otherwise. Needed for Qwen2.5-VL-7B
    
    ---------
    
    Signed-off-by: Shiv Kaul <shiv.kaul@intel.com>
    Signed-off-by: Iryna Boiko <iboiko@habana.ai>
    Co-authored-by: Iryna Boiko <iboiko@habana.ai>

[33mcommit b208bbd80c973dce73853f31920a8478e7bfba6b[m
Author: Youlei Yang <youlei.yang@intel.com>
Date:   Thu Jan 8 16:24:23 2026 +0800

    skip HPU graphs for long prefills (#780)
    
    - Set the batched tokens threshold to skip HPUgraph to
    `max_num_batched_tokens` if `max_cudagraph_capture_size` is not set.
    - Include the context tokens while calculating the batched tokens.
    
    ---------
    
    Signed-off-by: Youlei Yang <youlei.yang@intel.com>

[33mcommit 19be0a20f67a8f6a09783605fa4bfd63ef69c1df[m
Author: Agata Dobrzyniewicz <160237065+adobrzyn@users.noreply.github.com>
Date:   Thu Jan 8 08:45:10 2026 +0100

    WA shared bias in UA (#727)
    
    Signed-off-by: Agata Dobrzyniewicz <adobrzyniewicz@habana.ai>
    Signed-off-by: Agata Dobrzyniewicz <160237065+adobrzyn@users.noreply.github.com>

[33mcommit 1e8fb125cf9ac6c025d07fb48a2ef8a040fda524[m
Author: Paweł Olejniczak <polejniczakx@habana.ai>
Date:   Wed Jan 7 22:28:02 2026 +0100

    [FIX_FOR_VLLM_LATEST] Fix block_size used in eagle (#773)
    
    Fix for upstream changes:
    https://github.com/vllm-project/vllm/pull/31540/files
    
    ---------
    
    Signed-off-by: Paweł Olejniczak <polejniczakx@habana.ai>
    Co-authored-by: Iryna Boiko <iboiko@habana.ai>

[33mcommit 86997786153a2744d880ce026da9b61c6db3310a[m
Author: Yi Liu <yi4.liu@intel.com>
Date:   Thu Jan 8 03:24:27 2026 +0800

    Load KV scales for FP8 MLA (#763)
    
    - lazy mode w/ scale=1.0
    ```
    # vllm (pretrained=/mnt/disk5/hf_models/DeepSeek-V2-Lite-Chat-FP8_STATIC-fp8-kv-2,tensor_parallel_size=8,enable_expert_parallel=True,max_model_len=4096,max_num_seqs=64,gpu_memory_utilization=0.85,dtype=bfloat16,max_gen_toks=2048,enable_prefix_caching=False,max_num_batched_tokens=32768,kv_cache_dtype=fp8_inc), gen_kwargs: (None), limit: None, num_fewshot: None, batch_size: 64
    # |Tasks|Version|     Filter     |n-shot|  Metric   |   |Value |   |Stderr|
    # |-----|------:|----------------|-----:|-----------|---|-----:|---|-----:|
    # |gsm8k|      3|flexible-extract|     5|exact_match|↑  |0.6422|±  |0.0132|
    # |     |       |strict-match    |     5|exact_match|↑  |0.6399|±  |0.0132|
    ```
    
    - Lazy mode with the scale loaded from the checkpoint, which captured
    during the calibration process.
    ```bash
    vllm (pretrained=/mnt/disk5/hf_models/DeepSeek-V2-Lite-Chat-FP8_STATIC-fp8-kv-2,tensor_parallel_size=8,enable_expert_parallel=True,max_model_len=4096,max_num_seqs=64,gpu_memory_utilization=0.85,dtype=bfloat16,max_gen_toks=2048,enable_prefix_caching=False,max_num_batched_tokens=32768,kv_cache_dtype=fp8_inc), gen_kwargs: (None), limit: None, num_fewshot: None, batch_size: 128
    |Tasks|Version|     Filter     |n-shot|  Metric   |   |Value |   |Stderr|
    |-----|------:|----------------|-----:|-----------|---|-----:|---|-----:|
    |gsm8k|      3|flexible-extract|     5|exact_match|↑  |0.6513|±  |0.0131|
    |     |       |strict-match    |     5|exact_match|↑  |0.6505|±  |0.0131|
    ```
    Test model:
    https://huggingface.co/INC4AI/DeepSeek-V2-Lite-Chat-BF16-FP8-STATIC-FP8-KV-TEST-ONLY
    
    cc @hshen14 @thuang6
    
    ---------
    
    Signed-off-by: yiliu30 <yi4.liu@intel.com>

[33mcommit 007825544c4f00aecde210d223419b44c8876df2[m
Author: Konrad Zawora <kzawora@habana.ai>
Date:   Wed Jan 7 14:05:47 2026 +0100

    [Bugfix] Handle spec decode optionals in unified batch (#782)
    
    Spec decode unified batch buffers are provided as `Optional`s with
    `None` value as defaults, yet the code assumes them to be very
    non-optional, with direct calls to `get_cumsum_and_arange`,
    `prepare_spec_decode_inputs_fn` and `scheduled_spec_decode_tokens.get`.
    This PR properly handles the scenario where extra spec decode buffers
    are not provided.
    
    Signed-off-by: Konrad Zawora <kzawora@habana.ai>

[33mcommit 8a9efd1bdf040e74cc59f608376721070e6b6b5f[m
Merge: fe67f98 187a37d
Author: Libin Tang <libin.tang@intel.com>
Date:   Mon Jan 5 07:56:41 2026 -0800

    Merge branch 'main' into libinta/remove_gather_scatter

[33mcommit 187a37da8574cbb5a97e6be6147f69523e3cee05[m
Author: Harish Subramony <hsubramony@habana.ai>
Date:   Mon Jan 5 04:42:39 2026 -0800

    Update lmcache examples (#748)
    
    Signed-off-by: Harish Subramony <hsubramony@habana.ai>

[33mcommit 25e637c329122b13ce41890a5b375fb539b26d4a[m
Author: Paweł Olejniczak <polejniczakx@habana.ai>
Date:   Mon Jan 5 12:22:52 2026 +0100

    Fix repetition penalty crash in decode phase (#769)
    
    The sampler crashes when repetition penalties are used because
    make_selective_sampling_metadata() sets prompt_token_ids=None during
    skip_copy=True (decode phase), but penalties require prompt tokens. Add
    caching mechanism for prompt_token_ids to reuse the tensor when
    skip_copy=True and penalties are needed. Cache is invalidated when batch
    composition changes. Fixes repetition penalty support while preserving
    skip_copy performance optimization.
    
    Signed-off-by: Paweł Olejniczak <polejniczakx@habana.ai>

[33mcommit fe67f989fd4602a35ec55c182988e91bcad33097[m
Merge: bb3ac24 c799e57
Author: Libin Tang <libin.tang@intel.com>
Date:   Sun Jan 4 20:04:04 2026 -0800

    Merge branch 'main' into libinta/remove_gather_scatter

[33mcommit c799e5775742c06232b3bba44e793456fe5185cb[m
Author: Katarzyna Fojcik <katarzyna.fojcik@intel.com>
Date:   Fri Jan 2 14:20:38 2026 +0100

    [GAUDISW-244336] Add missing long ctx prompt buckets (#739)
    
    To generate all missing buckets for long ctx, first we generate ctx
    range up to max ctx for all queries;
    later we generate additional buckets for max ctx per query.
    
    ---------
    
    Signed-off-by: Katarzyna Fojcik <kfojcik@habana.ai>
    Co-authored-by: Iryna Boiko <iboiko@habana.ai>

[33mcommit 4520274ed5290b5a83cc39ca12c6834e63a8f74a[m
Author: Iryna Boiko <iboiko@habana.ai>
Date:   Wed Dec 31 08:32:28 2025 +0100

    [FIX_FOR_VLLM_LATEST] Fix structured_output after use_async_scheduling default usage in #27614 (#768)
    
    Signed-off-by: Iryna Boiko <iboiko@habana.ai>

[33mcommit 129fe5e3e59b4616de95d625dd3672d54320cef1[m
Author: Monika Helfer <monika.helfer@intel.com>
Date:   Tue Dec 30 15:23:30 2025 +0100

    Documentation: Fix missing back navigation arrow on mobile devices (#766)
    
    This change fixes an issue on mobile devices where the back navigation
    arrow was hidden, preventing users from navigating to higher-level
    documents. The rule responsible for this behavior is now applied only to
    desktop viewports, ensuring the arrow displays correctly in the menu bar
    on narrow screens.
    Before:
    <img width="300" alt="image"
    src="https://github.com/user-attachments/assets/869e4be7-2d95-4034-be5a-bd2ae893ed80"
    />
    After:
    <img width="300" alt="image"
    src="https://github.com/user-attachments/assets/4d06e119-901d-4adb-895b-14448e302857"
    />
    
    Signed-off-by: mhelf-intel <monika.helfer@intel.com>

[33mcommit bb3ac24cfd0b58b08f1225caffcbbdedf3fd53a5[m
Author: Iryna Boiko <iboiko@habana.ai>
Date:   Tue Dec 30 11:27:58 2025 +0100

    Update qwen3_vl.py
    
    update for pre-commit error:
    vllm_gaudi/models/qwen3_vl.py:62:25: F821 Undefined name `_require_is_multimodal`

[33mcommit 327a9ccc98451a98377a87d2efd31ed69681d26e[m
Merge: 495643a 4fdb716
Author: Iryna Boiko <iboiko@habana.ai>
Date:   Tue Dec 30 09:10:57 2025 +0100

    Merge branch 'main' into libinta/remove_gather_scatter

[33mcommit 4fdb71629f71e644d58284af012283bb7725cc5a[m
Author: Tianmu Li <tianmu.li@intel.com>
Date:   Tue Dec 30 00:04:00 2025 -0800

    Fix async_scheduling + batched prefill (#740)
    
    More robust handling of dummy logit position when dealing with chunked
    prompt.
    
    ---------
    
    Signed-off-by: Tianmu Li <tianmu.li@intel.com>

[33mcommit 495643a9f1b06df073846590582cbace69e050c0[m
Merge: 568b4eb f439c1a
Author: Libin Tang <libin.tang@intel.com>
Date:   Mon Dec 29 19:32:52 2025 -0800

    Merge branch 'main' into libinta/remove_gather_scatter

[33mcommit f439c1a99df6309a2e00ff36ef513b09f0bf6237[m
Author: Iryna Boiko <iboiko@habana.ai>
Date:   Mon Dec 29 13:18:06 2025 +0100

    [FIX_FOR_VLLM_LATEST] tokenizer fix for #31285 (#764)
    
    Signed-off-by: Iryna Boiko <iboiko@habana.ai>

[33mcommit 568b4eb7047f86621d57ffba9f353d89aa9b3dcf[m
Merge: 625d9c2 f9dc033
Author: Iryna Boiko <iboiko@habana.ai>
Date:   Mon Dec 29 10:23:01 2025 +0100

    Merge branch 'main' into libinta/remove_gather_scatter

[33mcommit 625d9c28c0451f876d82d15a4b7774cc1113126f[m
Author: Iryna Boiko <iboiko@habana.ai>
Date:   Mon Dec 29 10:22:29 2025 +0100

    Update qwen3_vl.py
    
    format

[33mcommit bff3cf58f11d7f2678daf8f129b6792f6a4f73d3[m
Author: Iryna Boiko <iboiko@habana.ai>
Date:   Mon Dec 29 10:21:47 2025 +0100

    Update qwen3_vl.py
    
    Format update

[33mcommit f9dc033e68a1210727e4cdc4876ab827cae877d9[m
Author: Radosław Smyrek <radoslawx.smyrek@intel.com>
Date:   Mon Dec 29 09:54:53 2025 +0100

    [GAUDISW-243560] Monkey-patching _get_attn_scale for the Llama4Attention layer (#758)
    
    Signed-off-by: Radoslaw Smyrek <radoslawx.smyrek@intel.com>

[33mcommit 7c6329efc6589bef0121e6a59e45e92857bc94ee[m
Author: Libin Tang <libin.tang@intel.com>
Date:   Wed Dec 24 23:04:56 2025 -0800

    precommit fix and fix use_window_sdpa

[33mcommit c6526de66ec87f384aebb2d335741d84c00cc643[m
Author: Libin Tang <libin.tang@intel.com>
Date:   Tue Dec 23 22:30:14 2025 -0800

    precomit fix

[33mcommit bdff63f3ea773655fb7412a5b42d657f1219d191[m
Merge: 49d7633 cc37f1f
Author: Libin Tang <libin.tang@intel.com>
Date:   Tue Dec 23 15:08:24 2025 -0800

    Merge branch 'main' into libinta/remove_gather_scatter

[33mcommit 49d76333802cf5d2d8900f90f907d58a86771d0f[m
Author: Libin Tang <libin.tang@intel.com>
Date:   Tue Dec 23 15:03:52 2025 -0800

    add qwen3_vl.py functions

[33mcommit 9d8e2723f370874f40bd8e99944f98af379d76b1[m
Author: Libin Tang <libin.tang@intel.com>
Date:   Tue Dec 23 14:11:41 2025 -0800

    Pick model runner change related to PR30475.
    Also overwrite qwen3_vl function to use _merge_multimodal_embeddings
    with index copy.

[33mcommit cc37f1f221ecb2733d02f3c3f138cc697f0acaac[m
Author: Iryna Boiko <iboiko@habana.ai>
Date:   Tue Dec 23 14:49:54 2025 +0100

    Fix for PR30684 (#757)
    
    Signed-off-by: Iryna Boiko <iboiko@habana.ai>

[33mcommit afe3e6d989835debccf996fec8c88e339b44e680[m
Author: Agata Dobrzyniewicz <160237065+adobrzyn@users.noreply.github.com>
Date:   Tue Dec 23 13:21:18 2025 +0100

    Change neural version (#754)
    
    Both 306 and 330 are not in public neural version - need to move it
    
    ---------
    
    Signed-off-by: Agata Dobrzyniewicz <adobrzyniewicz@habana.ai>

[33mcommit ac9cb19a7792167d9e69c41dbe4ac2642c6a436b[m
Author: Iryna Boiko <iboiko@habana.ai>
Date:   Tue Dec 23 09:26:54 2025 +0100

    [FIX_FOR_VLLM_LATEST] Quick fix for PR30684 (#742)
    
    1) Quick fix for upstream changes:
    [PR30684](https://github.com/vllm-project/vllm/pull/30684)
    2) Fix for upstream changes:
    https://github.com/vllm-project/vllm/pull/28891 (Port:
    [PR751](https://github.com/vllm-project/vllm-gaudi/pull/751))
    3) Fix for https://github.com/vllm-project/vllm/pull/31036
    issue: failed test case run_qwen3_compressed_tensor_dynamic_scaling_test
    ```(EngineCore_DP0 pid=5792)   File "/root/logs/vllm/vllm/model_executor/layers/fused_moe/layer.py", line 1487, in ensure_moe_quant_config_init
    (EngineCore_DP0 pid=5792)     self.quant_method.get_fused_moe_quant_config(self)
    (EngineCore_DP0 pid=5792)   File "/root/logs/vllm/vllm/model_executor/layers/quantization/fp8.py", line 1225, in get_fused_moe_quant_config
    (EngineCore_DP0 pid=5792)     w1_scale=layer.w13_weight_scale,
    (EngineCore_DP0 pid=5792)              ^^^^^^^^^^^^^^^^^^^^^^
    (EngineCore_DP0 pid=5792)   File "/usr/local/lib/python3.12/dist-packages/torch/nn/modules/module.py", line 1964, in __getattr__
    (EngineCore_DP0 pid=5792)     raise AttributeError(
    (EngineCore_DP0 pid=5792) AttributeError: 'FusedMoE' object has no attribute 'w13_weight_scale'. Did you mean: 'w13_weight_scale_inv'```
    
    This issue was already present, but it was not detected as marlin was disabled. After moe refactor in https://github.com/vllm-project/vllm/pull/31036, parameter self.use_marlin was replaced by self.fp8_backend. self.fp8_backend is disabled now
    
    ---------
    
    Signed-off-by: Iryna Boiko <iboiko@habana.ai>

[33mcommit b5a980dd168c1f815d7699487f586c790f54698c[m
Author: Konrad Zawora <kzawora@habana.ai>
Date:   Mon Dec 22 13:50:20 2025 +0100

    [Attention Metadata Overhaul 1/N] Extract metadata update to HPUAttentionMetadataProcessor  (#526)
    
    This PR is pretty simple - it takes all the metadata post-processing
    logic we do inside adapter, and yeets it from there into a separate
    class. This shouldn't introduce any functional changes other than a
    small refactor. In the next PR, I intend to remove metadata
    postprocessing from the adapter and do it beforehand, on CPU, but I
    didn't want to introduce too major changes here.
    
    I made this because I absolutely hated how
    https://github.com/vllm-project/vllm-gaudi/pull/475 ended up w.r.t.
    metadata postprocessing, so I'd like to gradually fix it before that PR
    lands.
    
    ---------
    
    Signed-off-by: Konrad Zawora <kzawora@habana.ai>
    Co-authored-by: Michał Kuligowski <michal.kuligowski@intel.com>

[33mcommit e09ea82c781f14b8551107462e28960a318712d5[m
Author: lkk <33276950+lkk12014402@users.noreply.github.com>
Date:   Sat Dec 20 04:54:51 2025 +0800

    Apply hw aligned scale (#734)
    
    refer this implementation:
    https://github.com/HabanaAI/vllm-fork/blob/habana_main/vllm/model_executor/layers/quantization/compressed_tensors/schemes/compressed_tensors_w8a8_fp8.py#L91
    
    ---------
    
    Signed-off-by: lvkaokao <kaokao.lv@intel.com>
    Co-authored-by: Chendi.Xue <chendi.xue@intel.com>

[33mcommit 3791cf027bed4ca463f9bb16e40bec35aac74316[m
Author: Patryk Wolsza <patryk.wolsza@intel.com>
Date:   Fri Dec 19 19:13:08 2025 +0100

    Update action to change CODEOWNERS for new release branch (#745)
    
    Signed-off-by: PatrykWo <patryk.wolsza@intel.com>

[33mcommit 163f28d54f25795a840506ee5094c9059a9537a1[m
Author: Konrad Zawora <kzawora@habana.ai>
Date:   Fri Dec 19 17:24:05 2025 +0100

    Implement Unified MLA backend (#721)
    
    This PR implements Multi-head Latent Attention (MLA) using unified
    attention algorithm for HPU.
    
    The forward pass handles two paths simultaneously:
    - **Causal path** for fresh tokens: Expands K from latent space and does
    full-dimensional attention. This is the "compute friendly" approach from
    the base MLA implementation.
    - **Cached path** for prefix-prefills[^1]/decode: Projects Q to latent
    space (via W_UK_T) and does attention in compressed space. This is the
    "data-movement friendly" approach.
    
    The nice thing about this approach is it handles mixed batches naturally
    - you can have both fresh tokens going through the causal path and
    cached tokens going through the cached path in the same forward pass,
    without the need to split this into separate `self._forward_prefill()`
    and `self._forward_decode()`, as you normally would when sticking to
    current MLA common code.
    
    Some other changes I had to make:
    - I've extended the existing unified attention cache helpers and some
    core functions to handle MLA. The main change is adding an `is_mla` flag
    to `CacheUtils` that controls whether we work with standard per-head
    caches or MLA's single latent cache. When `is_mla=True`, the code
    expects `value_cache=None` since everything's stored in one cache.
    - I also modified `partial_attn_causal()`, `partial_attn_shared()`, and
    `partial_attn_unique()` to accept an optional `w_uv` parameter. When
    provided, they know to handle MLA mode: fetch from latent cache, do
    attention, and apply the W_UV projection. The projection works
    surprisingly seamlessly with block-softmax, with no changes to scaling
    needed.
    - Added `unified_mla()` as the entry point that routes between causal
    and cached paths based on what's provided.
    
    I also added some GSM8K test coverage for DeepSeek-V2-Lite with unified
    MLA enabled. Accuracy is good. I didn't check performance.
    
    [^1]: I suspect cached path is not going to produce the best performance
    on big prefix-prefills. We'll need to check other approaches, perhaps
    some hybrid of the two, but this one is at least functional. It should
    do great on prefix-cached decodes though.
    
    ---------
    
    Signed-off-by: Konrad Zawora <kzawora@habana.ai>

[33mcommit b2ae5f4432a57bbc238646f1f38112f14be5373a[m
Author: lkk <33276950+lkk12014402@users.noreply.github.com>
Date:   Sat Dec 20 00:03:09 2025 +0800

    fix moe fp8 static model loading for compressored_tensors format. (#720)
    
    ## changes
    
    1. rewrite `HPUCompressedTensorsW8A8Fp8MoEMethod.__init__` to support
    fp8 static `weight per-channel, activation per-tensor`
    2. modify the `process_weights_after_loading` function to process moe
    fp8 static scales.
    
    
    <html xmlns:o="urn:schemas-microsoft-com:office:office"
    xmlns:dt="uuid:C2F41010-65B3-11d1-A29F-00AA00C14882"
    xmlns="http://www.w3.org/TR/REC-html40">
    
    <head>
    
    <meta name=ProgId content=OneNote.File>
    <meta name=Generator content="Microsoft OneNote 15">
    </head>
    
    <body lang=en-US style='font-family:Calibri;font-size:11.0pt'>
    <!--StartFragment-->
    
    <p style='margin:0in;font-family:Calibri;font-size:11.0pt'><span
    lang=en-US>use
    </span><a href="https://huggingface.co/Qwen/Qwen3-30B-A3B"><span
    lang=en-US>Qwen/Qwen3-30B-A3B
    · Hugging Face</span></a><span lang=en-US> and<span
    style='mso-spacerun:yes'> 
    </span>lm_eval (gsm8k, strict</span><span lang=zh-CN>-</span><span
    lang=en-US>match)
    for testing</span></p>
    
    <p style='margin:0in;font-family:Calibri;font-size:11.0pt'>&nbsp;</p>
    
    <div style='direction:ltr'>
    
    
    scheme | format | vllm gaudi | vllm project
    -- | -- | -- | --
    FP8 Dynamic (Qwen3-30B-A3B-FP8-DYNAMIC) | weight per-channel, activation
    per-token | 0.8939 | 0.8939
    FP8 Static (Qwen3-30B-A3B-FP8) | weight per-tensor, activation
    per-tensor | 0.8848 | 0.8923
    FP8 Static (Qwen3-30B-A3B-FP8-Static) | weight per-channel, activation
    per-tensor | 0.8923 | NA
    
    
    
    </div>
    
    <p style='margin:0in;font-family:"Microsoft YaHei";font-size:11.0pt'
    lang=zh-CN>&nbsp;</p>
    
    <p style='margin:0in;font-family:Calibri;font-size:11.0pt'><span
    lang=en-US>FP8
    Dynamic</span><span lang=zh-CN>&nbsp;</span><a
    
    href="https://huggingface.co/Intel/Qwen3-30B-A3B-FP8-DYNAMIC-Test-Only"><span
    lang=en-US>Intel/Qwen3-30B-A3B-FP8-DYNAMIC-Test-Only · Hugging
    Face</span></a></p>
    
    <p style='margin:0in;font-family:Calibri;font-size:11.0pt'><span
    lang=en-US>FP8
    Static</span><span lang=zh-CN>&nbsp;</span><a
    href="https://huggingface.co/Intel/Qwen3-30B-A3B-FP8-Test-Only"><span
    lang=en-US>Intel/Qwen3-30B-A3B-FP8-Test-Only · Hugging
    Face</span></a></p>
    
    <p style='margin:0in;font-family:Calibri;font-size:11.0pt'><span
    lang=en-US>FP8
    Static</span><span lang=zh-CN>&nbsp;</span><a
    
    href="https://huggingface.co/Intel/Qwen3-30B-A3B-FP8-Static-Test-Only"><span
    lang=en-US>Intel/Qwen3-30B-A3B-FP8-Static-Test-Only · Hugging
    Face</span></a></p>
    
    <p style='margin:0in;font-family:"Microsoft YaHei";font-size:11.0pt'
    lang=zh-CN>&nbsp;</p>
    
    <!--EndFragment-->
    </body>
    
    ---------
    
    Signed-off-by: lvkaokao <kaokao.lv@intel.com>
    Signed-off-by: Dudi Lester <dlester@habana.ai>
    Signed-off-by: git config -lDudi Lester <160421192+dudilester@users.noreply.github.com>
    Signed-off-by: mhelf-intel <monika.helfer@intel.com>
    Signed-off-by: Paweł Olejniczak <polejniczakx@habana.ai>
    Signed-off-by: Xinyu Chen <xinyu1.chen@intel.com>
    Signed-off-by: lkk <33276950+lkk12014402@users.noreply.github.com>
    Signed-off-by: Wuxun Zhang <wuxun.zhang@intel.com>
    Co-authored-by: git config -lDudi Lester <160421192+dudilester@users.noreply.github.com>
    Co-authored-by: Michał Kuligowski <michal.kuligowski@intel.com>
    Co-authored-by: Monika Helfer <monika.helfer@intel.com>
    Co-authored-by: Paweł Olejniczak <polejniczakx@habana.ai>
    Co-authored-by: Xinyu Chen <xinyu1.chen@intel.com>
    Co-authored-by: Copilot <175728472+Copilot@users.noreply.github.com>
    Co-authored-by: Wuxun Zhang <wuxun.zhang@intel.com>
    Co-authored-by: Chendi.Xue <chendi.xue@intel.com>

[33mcommit 6540110516812f7e99f00648d9517835e59d547e[m
Author: Sihan Chen <757407490@qq.com>
Date:   Fri Dec 19 23:59:26 2025 +0800

    Optimize KV block copy by merging k/v .to() calls (#729)
    
    merge 2 separate calls of key/value tensor `.to` function into 1 call
    can save up to 10% kv transfer time during per-layer block cp.
    
    ---------
    
    Signed-off-by: Spycsh <sihan.chen@intel.com>
    Co-authored-by: Chendi.Xue <chendi.xue@intel.com>

[33mcommit 026566723f2c9225d39d0badd2fcbf8284647955[m
Author: Patryk Wolsza <patryk.wolsza@intel.com>
Date:   Fri Dec 19 16:44:36 2025 +0100

    Revert "[GITHUB ACTION] Update BO process - add codeowner change and push to pre-releases" (#744)
    
    Reverts vllm-project/vllm-gaudi#663

[33mcommit 6db4fef6bf2b148ad406be14fab2e2ff31ef169d[m
Author: Kamil Kaczor <kamil.kaczor@intel.com>
Date:   Fri Dec 19 07:36:46 2025 +0100

    Add defrag unit tests (#738)
    
    During my analysis of the defragmenter mechanism I've created these unit
    tests so if they are useful we can merge them.
    
    Signed-off-by: Kamil Kaczor <kamil.kaczor@intel.com>

[33mcommit e9e0511987127048f1fc21c457a5792b5e504e9d[m
Author: Harish Subramony <harish.subramony@intel.com>
Date:   Thu Dec 18 11:32:56 2025 -0800

    enable lmcache  (#521)
    
    enable lmcache for hpu
    
    ---------
    
    Signed-off-by: Harish Subramony <hsubramony@habana.ai>
    Signed-off-by: Michał Kuligowski <michal.kuligowski@intel.com>
    Signed-off-by: mhelf-intel <monika.helfer@intel.com>
    Co-authored-by: Libin Tang <libin.tang@intel.com>
    Co-authored-by: Michał Kuligowski <michal.kuligowski@intel.com>
    Co-authored-by: mhelf-intel <monika.helfer@intel.com>

[33mcommit 18105cc511a3db222363797465ef2732a7306442[m
Author: Spurthi Lokeshappa <spurthi.lokeshappa@intel.com>
Date:   Thu Dec 18 21:08:14 2025 +0530

    Qwen2.5 vl no alignment (#698)
    
    Remove image restriction of 112 x 112 alignment by proper image padding
    and masking.
    
    ---------
    
    Signed-off-by: slokesha <slokeshappa@habana.ai>
    Signed-off-by: Spurthi Lokeshappa <slokeshappa@habana.ai>
    Signed-off-by: slokesha <spurthi.lokeshappa@intel.com>

[33mcommit 30bee3c4b2a938cc7717075cd567eb515172a922[m
Author: Patryk Wolsza <patryk.wolsza@intel.com>
Date:   Thu Dec 18 13:33:28 2025 +0100

    Pipeline parallelism updates (#737)
    
    Signed-off-by: PatrykWo <patryk.wolsza@intel.com>

[33mcommit 1e0409f1dd96d9c5366b156030818ee3d56f57f0[m
Author: Stanley <sckphoong@habana.ai>
Date:   Wed Dec 17 22:39:29 2025 -0800

    Fix missing unique attn image in the docs (#706)
    
    Not sure if this is the root cause, but the other images are loading
    properly on the readthedocs, only Unique Attn was not, and the format
    for unique attn was going to the root dir, so I think it might be this
    reason.
    
    This image shows the missing image in the readthedocs:
    
    <img width="802" height="494" alt="image"
    src="https://github.com/user-attachments/assets/b62a2670-c5f8-486b-8990-7d8ffa873310"
    />
    
    Source:
    https://docs.vllm.ai/projects/gaudi/en/latest/features/unified_attn.html#unique-attention

[33mcommit d6896de93041c532557849513fb82c713939d0d7[m
Author: Iryna Boiko <iboiko@habana.ai>
Date:   Thu Dec 18 07:07:49 2025 +0100

    [FIX_FOR_VLLM_LATEST] hourly fixes #647 and #732 (#735)
    
    Hourly fixes:
    CustomOp: grouped topk #647 - depends on
    https://github.com/vllm-project/vllm/pull/29575
    Fix HpuCommunicator.dispatch #732 - This is fix for upstream changes:
    https://github.com/vllm-project/vllm/pull/30014/files
    
    Signed-off-by: Iryna Boiko <iboiko@habana.ai>

[33mcommit bc203abd572dedd7a1b21423625295d523228560[m
Author: Xinyu Chen <xinyu1.chen@intel.com>
Date:   Thu Dec 18 07:27:11 2025 +0800

    mla: wrap sdpa with ModuleFusedSDPA (#730)
    
    for INC PatchedModule
    
    Signed-off-by: Xinyu Chen <xinyu1.chen@intel.com>

[33mcommit e26948506df2ec861fd5280f0c1eefdb97a5d956[m
Author: Libin Tang <litang@habana.ai>
Date:   Wed Dec 17 10:49:46 2025 -0800

    qwen3-vl enablement  (#700)
    
    1. Fix the crash issue below as inputs_embeds. is expected to be 2D but
    the current logic returns inputs_embeds.shape=torch.Size([1, 1024,
    5120]) when bs=1. bs>1 is not enabled yet for the flow
    
    
    "/root/litang/github/qwen3/vllm/vllm/model_executor/models/qwen3_vl.py",
    line 1563, in _compute_deepstack_embeds
    (EngineCore_DP0 pid=202) ERROR 12-08 15:59:28 [v1/engine/core.py:845]
    deepstack_input_embeds = deepstack_input_embeds.view(
    (EngineCore_DP0 pid=202) ERROR 12-08 15:59:28 [v1/engine/core.py:845]
    RuntimeError: shape '[1, 3, 5120]' is invalid for input of size 3072
    2. Enable multi-modal bucket warmup for qwen3-vl
    
    ---------
    
    Co-authored-by: Libin Tang <libin tang>

[33mcommit c81d0f7293126b5c4df1a82a8fdd572fc99becc2[m
Author: Jakub Sochacki <97886316+jakub-sochacki@users.noreply.github.com>
Date:   Wed Dec 17 10:03:20 2025 +0100

    Revert: Fix defragmenter compilation (8b131ae) (#719)
    
    This reverts the change from PR #334 commit
    8b131ae0bbc5467c8bf4a0dbc2db5f1c33057bb6 by compiling only the forward
    function instead of the entire module to restore previous behavior.
    
    Signed-off-by: Jacob-Intel <jakub.sochacki@intel.com>
    Co-authored-by: Kamil Kaczor <kamil.kaczor@intel.com>

[33mcommit b1f11cdd9d1f63f1bef8f43dd641ba8666ff7f70[m
Author: Xinyu Chen <xinyu1.chen@intel.com>
Date:   Wed Dec 17 08:36:00 2025 +0800

    DP: dispatch fp8 hidden_states in INC (#684)
    
    depends on #680
    
    ---------
    
    Signed-off-by: Xinyu Chen <xinyu1.chen@intel.com>

[33mcommit 955914b0b0fa33cd740c8f0aad1aed2254fd41d5[m
Author: Xinyu Chen <xinyu1.chen@intel.com>
Date:   Wed Dec 17 03:17:53 2025 +0800

    bucket: add query len 1 to prefill bucket (#645)
    
    Avoid the query length(1) of the prefix prefill on the decode side to be
    padded to the block size under PD+DP scenario.
    
    Use case:
    
    VLLM_EXPONENTIAL_BUCKETING=false/true VLLM_PROMPT_QUERY_BUCKET_MIN=1 on
    the decode side.
    
    ---------
    
    Signed-off-by: Xinyu Chen <xinyu1.chen@intel.com>
    Signed-off-by: Wuxun Zhang <wuxun.zhang@intel.com>
    Co-authored-by: Wuxun Zhang <wuxun.zhang@intel.com>
    Co-authored-by: Chendi.Xue <chendi.xue@intel.com>

[33mcommit f989d411c5533146bb0378c4caa658413b6ed77f[m
Author: Paweł Olejniczak <polejniczakx@habana.ai>
Date:   Tue Dec 16 15:58:01 2025 +0100

    [FIX_FOR_VLLM_LATEST] Add attn_selector_config and fix apply_rotary_pos_emb (#725)
    
    Fix for https://github.com/vllm-project/vllm/pull/30212 + cherry pick
    https://github.com/vllm-project/vllm-gaudi/pull/724/
    
    ---------
    
    Signed-off-by: Paweł Olejniczak <polejniczakx@habana.ai>

[33mcommit 15065b040f69705166953b9c9ea9bd37cf2de95c[m
Author: Xinyu Chen <xinyu1.chen@intel.com>
Date:   Tue Dec 16 00:26:54 2025 +0800

    pd: fix fp8_kv host transfer (#716)
    
    Signed-off-by: Xinyu Chen <xinyu1.chen@intel.com>

[33mcommit 262802a78c209886b76e966d5af1572976d6217f[m
Author: Paweł Olejniczak <polejniczakx@habana.ai>
Date:   Mon Dec 15 14:39:04 2025 +0100

    [FIX_FOR_VLLM_LATEST] Remove use_data_parallel from qwen2_5_vl (#718)
    
    Fix for upstream PR:
    https://github.com/vllm-project/vllm/pull/30125/
    
    Signed-off-by: Paweł Olejniczak <polejniczakx@habana.ai>

[33mcommit 454d97cfaf7748595adf8c312f45f04e0e43db15[m
Author: Monika Helfer <monika.helfer@intel.com>
Date:   Mon Dec 15 13:15:23 2025 +0100

    Add the quantization configuration to the public docs (#717)
    
    Signed-off-by: mhelf-intel <monika.helfer@intel.com>

[33mcommit ac2a881b3e16cbfcc4e201542da4c11ca038886f[m
Author: git config -lDudi Lester <160421192+dudilester@users.noreply.github.com>
Date:   Mon Dec 15 14:14:05 2025 +0200

    [GAUDISW-228042] Add support for dynamic vLLM kv-cache quantization (#538)
    
    Signed-off-by: Dudi Lester <dlester@habana.ai>
    Signed-off-by: git config -lDudi Lester <160421192+dudilester@users.noreply.github.com>
    Co-authored-by: Michał Kuligowski <michal.kuligowski@intel.com>

[33mcommit 5d13bcbcb60d9f690f05b91cf90e2d253f7b9a64[m
Author: Danny Semiat <dannysem@gmail.com>
Date:   Mon Dec 15 10:21:09 2025 +0200

    Block matmul and kv_cache in dynamic quantization (#673)
    
    Currently disabling matmul and kv_cache in dynamic quantization mode
    
    ---------
    
    Signed-off-by: Danny Semiat <dannysem@gmail.com>

[33mcommit b333e4bbbd184ccf2c1af0a9e83e40dbaa5207dc[m
Author: Soila Kavulya <soila.p.kavulya@intel.com>
Date:   Sun Dec 14 21:26:10 2025 -0800

    Optimize performance of static FP8 linear op (#715)
    
    Optimizes the performance of FP8 linear op similar to
    https://github.com/vllm-project/vllm-gaudi/blob/82937529b3535014e17b88aab7d8dc58e35176cc/vllm_gaudi/extension/ops.py#L687
    and
    https://github.com/vllm-project/vllm/blob/2f32a68d75324299d13025c75f9cb5427e5c445d/vllm/model_executor/layers/quantization/utils/w8a8_utils.py#L454
    
    Signed-off-by: Kavulya, Soila P <soila.p.kavulya@intel.com>

[33mcommit 82937529b3535014e17b88aab7d8dc58e35176cc[m
Author: Xinyu Chen <xichen@habana.ai>
Date:   Sat Dec 13 01:11:33 2025 +0800

    DP: dispatch tensor in FusedMoEMethod (#680)
    
    This PR is mainly to move the dispatch logic from vllm to vllm-gaudi so
    that we can do more ninja optimizations. E.g.,
    
    - we can dispatch the topk weights and ids instead of router_logits
    because the topk performance is not good when the sequence length is
    long.
    - we can dispatch the fp8 hidden_states after quantization for smaller
    message size. This will be addressed in #684
    
    Signed-off-by: Xinyu Chen <xinyu1.chen@intel.com>
    Signed-off-by: Chendi.Xue <chendi.xue@intel.com>
    Co-authored-by: Chendi.Xue <chendi.xue@intel.com>

[33mcommit db4162b154601f2f34d16e5a3cc8f8fc12c6246c[m
Author: Xinyu Chen <xichen@habana.ai>
Date:   Fri Dec 12 23:48:09 2025 +0800

    Optimize MoE via chunk settings (#658)
    
    add new feature `VLLM_MOE_CHUNK` and `VLLM_MOE_TOKEN_BOUNDARY`, with
    this, chunk_size and global_num_experts will be passed to
    torch.ops.hpu.mixture_of_experts for better performance.
    
    example, for ds_r1, we use the following setting for better performance.
    
      export VLLM_MOE_CHUNK="64, 128"
      export VLLM_MOE_TOKEN_BOUNDARY="2048, 4096"
    
    Signed-off-by: Xinyu Chen <xinyu1.chen@intel.com>

[33mcommit a7fb6b9b54481f217218073cd2a023af33f20175[m
Author: Agata Dobrzyniewicz <160237065+adobrzyn@users.noreply.github.com>
Date:   Fri Dec 12 14:30:51 2025 +0100

    [FIX_FOR_VLLM_LATEST] Maybe fix for 29066 (#709)
    
    Culprit commit: https://github.com/vllm-project/vllm/pull/29066
    
    ---------
    
    Signed-off-by: Dobrzyniewicz, Agata <agata.dobrzyniewicz@intel.com>
    Signed-off-by: Agata Dobrzyniewicz <160237065+adobrzyn@users.noreply.github.com>
    Signed-off-by: Iryna Boiko <iboiko@habana.ai>
    Co-authored-by: Copilot <175728472+Copilot@users.noreply.github.com>
    Co-authored-by: Iryna Boiko <iboiko@habana.ai>

[33mcommit 4fa4087e7c3efe4d63405d64f925789d0b109b88[m
Author: Adam Ghandoura <adam.ghandoura@intel.com>
Date:   Fri Dec 12 10:58:37 2025 +0100

    Add vLLM UBI Dockerfile for Gaudi with RHEL 9.6 (#686)
    
    GAUDISW-242243
    
    - Multi-stage build: gaudi-base → gaudi-pytorch → vllm-final
    
    Build arguments:
    - SYNAPSE_VERSION: Habana Synapse AI version (default: 1.22.1)
    - PT_VERSION: PyTorch version (default: 2.7.1)
    - VLLM_GAUDI_COMMIT: vllm-gaudi git commit/tag (default: main)
    - VLLM_PROJECT_COMMIT: vllm upstream commit (auto-detected if empty)
    - TORCH_TYPE: PyTorch type - 'upstream' or 'fork' (default: upstream)
    
    Usage:
      docker build --build-arg SYNAPSE_VERSION=1.23.0 -t vllm-gaudi:1.23.0 .
    
    ---------
    
    Co-authored-by: Patryk Wolsza <patryk.wolsza@intel.com>

[33mcommit c03ca8dd792fae26eb9a73bfcc4c5cdb6f8fb89a[m
Author: Xinyu Chen <xinyu1.chen@intel.com>
Date:   Fri Dec 12 08:14:12 2025 +0800

    Patch Grouped Topk (#708)
    
    Signed-off-by: Xinyu Chen <xinyu1.chen@intel.com>

[33mcommit e0b176c97ec3451b1a640035423d7a5715b8514a[m
Author: Mandy Li <mandy.j.li@intel.com>
Date:   Thu Dec 11 08:26:54 2025 -0800

    Enable inc dynamic quant for MoE models (#688)
    
    This PR enables INC dynamic quantization for MoE models by adding
    dequant channel-wise weight to MoE OP
    
    Signed-off-by: mandy-li <mandy.j.li@intel.com>

[33mcommit 37b56e5b9d57d41049ce7c4d20748e20504362e9[m
Author: Chendi.Xue <chendi.xue@intel.com>
Date:   Thu Dec 11 10:25:13 2025 -0600

    [ACTION] update PR dashboard refresh frequency (#710)
    
    Signed-off-by: Chendi Xue <chendi.xue@intel.com>

[33mcommit 897b0ad3ac38117b7bd054fb8ddb175143664d5e[m
Author: Spurthi Lokeshappa <spurthi.lokeshappa@intel.com>
Date:   Thu Dec 11 20:18:30 2025 +0530

    Handle "pooling_states" for Embed Task (#699)
    
    Signed-off-by: slokesha <slokeshappa@habana.ai>
    Signed-off-by: Spurthi Lokeshappa <spurthi.lokeshappa@intel.com>
    Co-authored-by: Paweł Olejniczak <pawelolejniczak92@gmail.com>

[33mcommit 0f9e1f3e26acfe114bf521c6b7d956ef7441730b[m
Author: Kamil Kaczor <kamil.kaczor@intel.com>
Date:   Thu Dec 11 12:57:48 2025 +0100

    Reduce defrag operations in non-apc runs (#685)
    
    Signed-off-by: Kamil Kaczor <kamil.kaczor@intel.com>

[33mcommit 547178039ebe914cc92fa51efffa4cab93d206d8[m
Author: Xinyu Chen <xinyu1.chen@intel.com>
Date:   Thu Dec 11 11:30:54 2025 +0800

    make mla weight contiguous (#646)
    
    Signed-off-by: Xinyu Chen <xinyu1.chen@intel.com>

[33mcommit 569f3419b72a25d28b4d12de8f4d2631c3514111[m
Author: Spurthi Lokeshappa <slokeshappa@habana.ai>
Date:   Tue Dec 9 23:00:04 2025 +0530

    Fixed Plugin Test (#70)
    
    Added main() to avoid python multiprocessing runtime error
    
    ---------
    
    Signed-off-by: slokesha <slokeshappa@habana.ai>

[33mcommit 46805f9eb21fdac53a65990c0971ed20c9c5b9d5[m
Author: Patryk Wolsza <patryk.wolsza@intel.com>
Date:   Tue Dec 9 10:20:02 2025 +0100

    Add command printout for vLLM server startup (#687)
    
    Generates the vllm serve command in the printout with autocalc
    parameters.
    
    ---------
    
    Signed-off-by: PatrykWo <patryk.wolsza@intel.com>

[33mcommit 4a653933ce886f42ee54085bf71e5838a19628ea[m
Author: Xinyu Chen <xinyu1.chen@intel.com>
Date:   Tue Dec 9 09:15:10 2025 +0800

    pd: support fp8 kvcache in insert_blocks_to_device (#693)
    
    Signed-off-by: Xinyu Chen <xinyu1.chen@intel.com>

[33mcommit d1f6d3569666c672c95ec85f16905f4f6e64ee2f[m
Author: Chendi.Xue <chendi.xue@intel.com>
Date:   Mon Dec 8 17:40:42 2025 -0600

    [ACTION] quick fix for PR dashboard action (#704)
    
    Signed-off-by: Chendi Xue <chendi.xue@intel.com>

[33mcommit e7539244a4be731c7cca34de8c8f38243b498fde[m
Author: Chendi.Xue <chendi.xue@intel.com>
Date:   Mon Dec 8 17:00:46 2025 -0600

    [ACTION] PR Dashboard (#703)
    
    Signed-off-by: Chendi Xue <chendi.xue@intel.com>

[33mcommit 784a07e752b22aa1be84066ca60f2fd9c166090d[m
Author: Chendi.Xue <chendi.xue@intel.com>
Date:   Mon Dec 8 16:15:24 2025 -0600

    [ACTION]enable workflow to create PR view (#702)
    
    This PR is to create a new workflow file which will update PR dashboard
    every 30 mins
    So we will have a clear view that if one PR has been assigned and how
    long PR was opened
    
    ---------
    
    Signed-off-by: Chendi Xue <chendi.xue@intel.com>

[33mcommit de92b87ffa1c56769d0b45fe15ea7a0ed2a4c135[m
Author: Agata Dobrzyniewicz <160237065+adobrzyn@users.noreply.github.com>
Date:   Mon Dec 8 13:23:55 2025 +0100

    [FIX_FOR_VLLM_LATEST] Fix for hourly (#697)
    
    Culprit: https://github.com/vllm-project/vllm/pull/29665 and
    https://github.com/vllm-project/vllm/pull/27938
    
    ---------
    
    Signed-off-by: Dobrzyniewicz, Agata <agata.dobrzyniewicz@intel.com>

[33mcommit 5273e99ccd4aa787318c9eba70c9256ed461ec27[m
Author: Monika Helfer <monika.helfer@intel.com>
Date:   Fri Dec 5 15:44:46 2025 +0100

    Update the compatibility matrix (#659)
    
    Signed-off-by: mhelf-intel <monika.helfer@intel.com>

[33mcommit b49ae19a47a4d2cfb8919ec0f3c87a08df3e829a[m
Author: Paweł Olejniczak <polejniczakx@habana.ai>
Date:   Fri Dec 5 15:24:39 2025 +0100

    [FIX_FOR_VLLM_LATEST] Remove MultiModalKwargs.as_kwargs usage (#696)
    
    Remove MultiModalKwargs.as_kwargs usage to align with upstream changes.
    Temporarily disable embedding tests.
    
    ---------
    
    Signed-off-by: Paweł Olejniczak <polejniczakx@habana.ai>

[33mcommit b03c6bd068db6afab174ba9fd97fa54dca616bdd[m
Author: Paweł Olejniczak <polejniczakx@habana.ai>
Date:   Fri Dec 5 15:01:44 2025 +0100

    Add get_device_total_memory method (#656)
    
    It is required to pass _test_defaults_with_usage_context_ tests.
    
    ---------
    
    Signed-off-by: Paweł Olejniczak <polejniczakx@habana.ai>
    Co-authored-by: Iryna Boiko <iboiko@habana.ai>

[33mcommit aba4e6c33d4589b916900a7129a24415db1699f8[m
Author: Konrad Zawora <kzawora@habana.ai>
Date:   Fri Dec 5 12:54:12 2025 +0100

    [CI] GSM8K: don't override lm_eval batch_size & don't disable prefix caching by default (#640)
    
    I'm not sure why we override the batch size - it makes the entire test
    so much slower, since the requests are not pipelined (continuous
    batching is effectively disabled), and I'm also not sure why we disable
    prefix caching by default. vLLM keeps it on and let's do it the same
    way.
    
    ---------
    
    Signed-off-by: Konrad Zawora <kzawora@habana.ai>
    Co-authored-by: Agata Dobrzyniewicz <160237065+adobrzyn@users.noreply.github.com>

[33mcommit 066908868335a79c03494f6ba81414dc03b5c9d4[m
Author: Chendi.Xue <chendi.xue@intel.com>
Date:   Thu Dec 4 14:43:18 2025 -0600

    [GAUDISW-241080] enable spec decode for Unified Attention (#619)
    
    SW-241080
    
    **Current status:**
       UA + spec_decode NGRAM => Done => accuracy verified
       UA + spec_decode eagle3 => Done => accuracy verified
    
    **design doc:**
    For non-UA, we will pad target_model input to fix token_ids shape to
    limit potential hpu graph possibility
    For UA, we can use actual draft token to avoid redundant padding =>
    follow very similar design as GPU does
    
    Which is to say, For UA spec decode:
    1. we skip spec decode for target model if non_draft_tokens generated
    from last run
    2. with draft token, do not pad input_token_ids, use
    target_token_indices and bonus_token_indices to indicate tokens for
    reject sampler to judge
    3. However, for inupt to draft model, we will reuse attn_metadata from
    target model(as initial impl) => update meta to remove rejected token
    will be next step.
    
    ```
    # Example:
          # scheduled_spec_decode_tokens={'0': [-1], '1': [-1], '2': [17689], '3': [-1]} => only 3rd request has draft token
          # token_ids = [[tok_0], [tok_1], [tok_2, draft_tok], [tok_4]]
          # draft_token_indices = [0, 0, 3, 0] => pos of token_ids for compare to target model output
          # target_token_indices = [-1, -1, 2, -1] => -1 is place holder, only verify pos==2 of target model output
          # bonus_token_indices = [0, 1, 3, 4] => new generated token from target model
    
          # current design for draft model fwd
          # say if target token gets verified by target model
          # => last token indices to select draft token from draft model is [0, 1, 3, 4]
          # say if target token gets rejected
          # => last token indices select draft token from from draft model is [0, 1, 2, 4]
    ```
    
    **workflow:**
    == start step ==
    input(contains prompt, no draft) => target_model => regular sampling =>
    update states => draft model => update draft token for next step
    
    == next step ==
    input(contains draft tokens) => target_model (use sharable attn with
    multiple tokens in one req - token + draft tokens) => rejection sampler
    (verify draft tokens to get final validated sampled tokens) => update
    states => draft model (to get new draft tokens, we need to skip any
    tokens rejected by target model) => update draft token for next step
    Example:
    input is
    input with draft is [[in your] [is] [nice tool] [name]]
    => output from target model is     [[your mind] [an] [way used] [name]]
    => after rejection sampler [[your mind] [an] [way] [name]] // only
    accept bonus token when draft accepeted
    => input to draft model [[mind] [an] [way] [name]] // notice, we need to
    create new attn_meta or reuse existing but calfully select output
    indices.
    
    
    **Changes introduced in this PR**
    1. add new arg for spec decode to `create_unified_batch`
    2. update `unified_execute_model` UPDATE REQUEST STATE part, so draft
    token can be picked by scheduler
    4. shift parameters `propose_draft_token_ids` so we can make several
    arguments with default values
    5. implement new `_prepare_spec_decode_inputs_for_ua` for Unified
    Attention preparation
    6. Add new propose_eagle_unified with new proposal file
    
    
    **Validation:**
    ```
    VLLM_UNIFIED_ATTN=True VLLM_SKIP_WARMUP=True PT_HPU_LAZY_MODE=1 python "${VLLM_GAUDI_PREFIX}/tests/full_tests/spec_decode.py" --task ngram --assert_acc_rate 0.25 --osl 512
    and
    VLLM_UNIFIED_ATTN=True VLLM_SKIP_WARMUP=True PT_HPU_LAZY_MODE=1 python "${VLLM_GAUDI_PREFIX}/tests/full_tests/spec_decode.py" --task eagle3 --assert_accept_rate 0.50 --osl 1024
    
    ================= spec_ngram =================
    latency: 46.99283313751221
    acc_counts: [1742, 0]
    acc_rate: 0.27142411966344654
    num_draft_tokens: 6418
    num_drafts: 6418
    ---
    Prompt: Hello, my name is
    Generated text:  Xiaoyu, and I'm a student at the University of Science and Technology of China. I'm currently studying in the Department of Physics. I'm in my second year, and I'm majoring in physics. I'm interested'...'
    ---
    Prompt: The president of the United States is
    Generated text:  the head of state and government of the United States. The president is the head of the executive branch of the U.S. government, and is the commander-in-chief of the United States Armed Forces. The p'...'
    ---
    Prompt: The capital of France is
    Generated text:  Paris. The capital of Germany is Berlin. The capital of Italy is Rome. The capital of Spain is Madrid. The capital of Portugal is Lisbon. The capital of Greece is Athens. The capital of Belgium is Br'...'
    ---
    Prompt: The future of AI is
    Generated text:  a topic that has been the subject of much speculation and debate. As the technology continues to evolve, it's clear that AI will play an increasingly important role in our lives. However, the questio'...'
    ---
    Prompt: San Francisco is know for its
    Generated text:  fog, but the fog is not the only thing that is fog-like. The city is also known for its fog, but the fog is not the only thing that is fog-like. The city is also known for its fog, but the fog is not'...'
    ---
    Prompt: Facebook was created in 2004 by
    Generated text:  Mark Zuckerberg, and it has become one of the most popular social media platforms. It is a social networking site that allows users to connect with friends and family, share photos and videos, and po'...'
    ---
    Prompt: Curious George is a
    Generated text:  2015 American 3D computer-animated comedy film directed by Tom McCamus and written by David W. Zucker, and starring the titular character, Curious George, voiced by the actor and comedian Will Ferrel'...'
    ---
    Prompt: Python 3.11 brings improvements to its
    Generated text:  standard library, including the `typing` module. One of the notable changes is the introduction of the `TypeAlias` feature, which allows for the creation of type aliases in a more readable and concis'...'
    =========================================
    ```
    
    ---------
    
    Signed-off-by: Chendi Xue <chendi.xue@intel.com>

[33mcommit 5685ee93420218c54b925ef895a8c6f8d74338ea[m
Author: Patryk Wolsza <patryk.wolsza@intel.com>
Date:   Thu Dec 4 17:44:23 2025 +0100

    Update nixl Dockerfile to use v1.22.2 and parameterize vllm-project (#674)
    
    Updates needed to make the dockerfile flexible for upcoming releases.
    
    ---------
    
    Signed-off-by: PatrykWo <patryk.wolsza@intel.com>
    Signed-off-by: Patryk Wolsza <patryk.wolsza@intel.com>
    Co-authored-by: Copilot <175728472+Copilot@users.noreply.github.com>

[33mcommit a955e1ebc3f474624a492bdab5881fb41ce0e18c[m
Author: Kacper Pietkun <kpietkun@habana.ai>
Date:   Thu Dec 4 16:08:36 2025 +0100

    Sleep mode support (#584)
    
    Sleep mode level 1 - based on
    https://github.com/HabanaAI/vllm-fork/pull/2055
    
    ---------
    
    Signed-off-by: Kacper Pietkun <kpietkun@habana.ai>

[33mcommit 3fc22d011e00e6099afd944cdf6ac41bf431a6eb[m
Author: Krzysztof Smusz <ksmusz@habana.ai>
Date:   Thu Dec 4 15:39:23 2025 +0100

    Implementing softmax_fa2 in partial_attn shared and causal (#566)
    
    Signed-off-by: Krzysztof Smusz <ksmusz@habana.ai>
    Co-authored-by: Konrad Zawora <kzawora@habana.ai>

[33mcommit b8515d5fb8d5966768ad03e71bbbe1ad6661d7df[m
Author: Haifeng Chen <haifeng.chen@intel.com>
Date:   Thu Dec 4 04:38:46 2025 +0800

    Spec decode warmup support (#624)
    
    GAUDISW-242931
    
    Because currently spec decode flatten the spec decode tokens into
    [batch_size * num_tokens, 1], we can warmup the decode shapes as it was.
    The thing changed is the maximum batch_size we should warmup in the
    configuration because the real batch size is batch_size * num_tokens
    which is num_tokens (1 + num_speculative_tokens) times of original batch
    size.
    
    The thing to care in the warmup is the draft token (and block) space for
    the proposing process in eagle. We need to leave out the
    num_speculative_tokens space to use by propose for eagle.
    
    Other care needs to be taken (already done in the PR of support
    num_speculative_tokens > 1) is warmup will be run in compile only mode
    without the real computation happening. So the operations for
    prepare_attn_metadata in the drafter which depends on the real position
    values must be done on CPU)
    
    Another issue of handling no spec decode tokens for decode phase has
    already been handled https://github.com/vllm-project/vllm-gaudi/pull/593
    
    ---------
    
    Signed-off-by: Chen Haifeng <haifeng.chen@intel.com>

[33mcommit 95bdd8faaa2994959a49c660ba7980552c28e165[m
Author: Iryna Boiko <iboiko@habana.ai>
Date:   Wed Dec 3 21:31:04 2025 +0100

    [FIX_FOR_VLLM_LATEST] CompressedTensorsWNA16MarlinMoEMethod fix for #28871 (#676)
    
    Signed-off-by: Iryna Boiko <iboiko@habana.ai>

[33mcommit b4233123441c14835a4d79eb8b62fd7be4d0de37[m
Author: Chendi.Xue <chendi.xue@intel.com>
Date:   Wed Dec 3 12:04:13 2025 -0600

    [GITHUB ACTION] Update BO process - add codeowner change and push to pre-releases (#663)
    
    Signed-off-by: Chendi Xue <chendi.xue@intel.com>

[33mcommit 36d92db13b80c3d767821d11e0eff936eebf59d1[m
Author: Jimin Ha <jimin.ha@intel.com>
Date:   Wed Dec 3 07:13:21 2025 -0800

    Fix compile error for Gemma3 multimodal inputs (#671)
    
    Due to the latest changes from upstream, gemma3 is failing to compile on
    HPU
    https://github.com/vllm-project/vllm/pull/27772/
    https://github.com/vllm-project/vllm/pull/28842
    
    -replace unfold to view/reshape
    -replace text embedding to avoid dynamic shape
    -remove merge_multimodal replacement since masked_scatter issue is fixed
    -enable back gemma3 model test
    
    ---------
    
    Signed-off-by: Jimin Ha <jimin.ha@intel.com>

[33mcommit 927dafa744f0accc1ccd08ac4953809a94befe24[m
Author: Tianmu Li <tianmu.li@intel.com>
Date:   Wed Dec 3 02:53:09 2025 -0800

    Resolve issue with async scheduling when decode and prompt tokens are mixed (#642)
    
    When decode tokens are not strictly before prompt tokens, tokens from
    the previous batch cannot be copied using :num_decodes when using async
    scheduling.
    
    ---------
    
    Signed-off-by: Tianmu Li <tianmu.li@intel.com>

[33mcommit 21a11362808120256b9de8de45002aa3b33198ec[m
Author: lkk <33276950+lkk12014402@users.noreply.github.com>
Date:   Wed Dec 3 08:07:56 2025 +0800

    fix loading fp8 static quantized model for compressored_tensors format. (#552)
    
    ## changes
    
    1. add "HPUCompressedTensorsLinearMethod" in the
    "WEIGHT_LOADER_V2_SUPPORTED", which can fix shape issue when load fp8
    quantization model
    2. fix "input_scale", use `input_scale.max()` for static per-tensor
    scale
    
    ---------
    
    Signed-off-by: lkk <33276950+lkk12014402@users.noreply.github.com>
    Signed-off-by: lvkaokao <kaokao.lv@intel.com>
    Co-authored-by: Chendi.Xue <chendi.xue@intel.com>

[33mcommit df3e30e51189d76d4f76b784ffe03f4196bdb5ac[m
Author: Mandy Li <mandy.j.li@intel.com>
Date:   Tue Dec 2 14:54:08 2025 -0800

    Enable dequant fp8 weights quantized per-channel with compressed-tensor method (#621)
    
    This PR enables dequant fp8 weights quantized with compressed-tensor
    method channel-wise
    
    Signed-off-by: mandy-li <mandy.j.li@intel.com>

[33mcommit 2c118170364a78dff282724de313d813f7efa820[m
Author: Iryna Boiko <iboiko@habana.ai>
Date:   Tue Dec 2 17:31:55 2025 +0100

    [FIX_FOR_VLLM_LATEST] test cases fix for PR29859 (#670)
    
    Signed-off-by: Iryna Boiko <iboiko@habana.ai>

[33mcommit a9712f4adc2af959e445a737bb06ce41879ec899[m
Author: Monika Helfer <monika.helfer@intel.com>
Date:   Tue Dec 2 15:08:06 2025 +0100

    Docs: broken links fixes (#668)
    
    Signed-off-by: mhelf-intel <monika.helfer@intel.com>

[33mcommit bb79c4dde0bf12571f4e0ad71b8b1bbed189d133[m
Author: Paweł Olejniczak <polejniczakx@habana.ai>
Date:   Tue Dec 2 13:36:09 2025 +0100

    [FIX_FOR_VLLM_LATEST] Fix the Attention imports (#652)
    
    This patch fixes 4 errors:
    
    ImportError: cannot import name 'Attention' from 'vllm.attention'
    (/usr/local/lib/python3.12/dist-packages/vllm/attention/__init__.py).
    Did you mean: 'attention'?
    
    TypeError: Qwen2_5_VisionBlock.__init__() got an unexpected keyword
    argument 'use_upstream_fa'
    
    TypeError: HPUCompressedTensorsWNA16MoEMethod.__init__() takes 3
    positional arguments but 4 were given
    
    AttributeError: 'LlamaForCausalLM' object has no attribute
    'embedding_padding_modules'
    
    ---------
    
    Signed-off-by: Paweł Olejniczak <polejniczakx@habana.ai>

[33mcommit e4f6147a5734180130d4a48a12770086ae5ecf55[m
Author: Monika Helfer <monika.helfer@intel.com>
Date:   Tue Dec 2 13:16:28 2025 +0100

    Docs: Missing content from Habana docs (#562)
    
    This PR adds the missing warm-up and quantization content from Habana
    docs. Additionally, the warm-up document has been split into shorter,
    separate documents and moved to the Configuration section, where it
    seems to fit better than Features.
    
    Please let me know if you have any comments.
    
    ---------
    
    Signed-off-by: mhelf-intel <monika.helfer@intel.com>

[33mcommit ddc256a15e113da532120ad88c01e7b4c23ed434[m
Author: Patryk Wolsza <patryk.wolsza@intel.com>
Date:   Tue Dec 2 13:00:10 2025 +0100

    1.22.2 post release edits to dockers (#660)
    
    Signed-off-by: PatrykWo <patryk.wolsza@intel.com>

[33mcommit 7528cef9216633f0bc090a2c19b94241a72c9601[m
Author: Chendi.Xue <chendi.xue@intel.com>
Date:   Tue Dec 2 05:11:24 2025 -0600

    [BUGFIX][GAUDISW-244351][SPEC DECODE] Fix failing introduced by #29223 (#665)
    
    GAUDISW-244351
    
    This PR should be merged after #664
    
     #29223 introduced a tuple return in rejection_sampler.parse_output
    Fixed in this PR
    
    ---------
    
    Signed-off-by: Chendi Xue <chendi.xue@intel.com>
    Signed-off-by: Iryna Boiko <iboiko@habana.ai>
    Co-authored-by: Iryna Boiko <iboiko@habana.ai>

[33mcommit 856f9804e0ae2be4f0aedd7a7913171ec181d07a[m
Author: Chendi.Xue <chendi.xue@intel.com>
Date:   Tue Dec 2 02:32:58 2025 -0600

    [SPEC_DECODE][BUG FIX] Fix incorrect output token with batch_size > 1 (#664)
    
    GAUDISW-244200
    
    Reported issue: accept rate is varying. Sometimes lower to 0.65 and
    sometimes over 0.7.
    
    Root Cause:
    Generated token is mapped to wrong req (shifting to next request)
    
    Error example:
    For req4, first token 'Paris' should be output of req3
    For req6, first token 'fog' should be output of req5
    ```
    ---
    Prompt: Hello, my name is
    Generated text:  Xiaoyu, and I'm a student at the University of Science and Technology of China...
    ---
    Prompt: The president of the United States is
    Generated text:  nevertheless a president of the United States, and the president of the United .'...'
    ---
    Prompt: The capital of France is
    Generated text:  the city of Paris. The city of Paris is located in the north of France..'...'
    ---
    Prompt: The future of AI is
    Generated text:  Paris, 2030. The city is a hub of innovation, with AI-driven solutions tra.'...'
    ---
    Prompt: San Francisco is know for its
    Generated text:  a lot of things, but one of the most famous is the Golden Gate Bridge..'...'
    ---
    Prompt: Facebook was created in 2004 by
    Generated text:  fog, Mark Zuckerberg, and his roommates.  The name Facebook came from the n.'...'
    ---
    Prompt: Curious George is a
    Generated text:  Markov chain. The states are the different animals in the book. The tra.'...'
    ---
    Prompt: Python 3.11 brings improvements to its
    Generated text:  3rd-party libraries, including the `typing` module. One of the notable changes .'...'
    ```
    
    Fix in this PR:
    Main reason is that, hpu_model_runner might pad token_ids to larger size
    to reduce hpu_graph possibility.
    And spec decode does not trim the padding tokens in output, which result
    in token generated by target_model from dummy was regarded as correct
    token
    
    After the fixing:
    ```
    Prompt: Hello, my name is
    Generated text:  Xiaoyu, and I'm a student at the University of Science and Technology of China. '...'
    ---
    Prompt: The president of the United States is
    Generated text:  the head of state and government of the United States. The president is the hea'...'
    ---
    Prompt: The capital of France is
    Generated text:  Paris. The capital of Germany is Berlin. The capital of Italy is Rome. The capital '...'
    ---
    Prompt: The future of AI is
    Generated text:  a topic that has captured the imagination of technologists, economists, an'...'
    ---
    Prompt: San Francisco is know for its
    Generated text:  fog, but the fog is not the only thing that is fog-like. The city is also known for i'...'
    ---
    Prompt: Facebook was created in 2004 by
    Generated text:  Mark Zuckerberg, and it has grown into a global social media platform w'...'
    ---
    Prompt: Curious George is a
    Generated text:  2015 American 3D computer-animated comedy film directed by Tom McCamus and'...'
    ---
    Prompt: Python 3.11 brings improvements to its
    Generated text:  standard library, including the `typing` module. One of the notable changes is'...'
    ```
    
    Add-on:
    To avoid similar issue, added accuracy test as well
    
    ---------
    
    Signed-off-by: Chendi Xue <chendi.xue@intel.com>

[33mcommit 1f1b07501e4fe6009f1eae77dec8fdc0e92fe3e0[m
Author: Yi Liu <yi4.liu@intel.com>
Date:   Tue Dec 2 06:45:43 2025 +0800

    Fix environment setup for FP8 (#623)
    
    In case the user provides only `QUANT_CONFIG` without specifying
    `quantization=inc`.
    
    Signed-off-by: yiliu30 <yi4.liu@intel.com>

[33mcommit aee4e1de7fca2e74158830f532273cb9c4b948de[m
Author: Patryk Wolsza <patryk.wolsza@intel.com>
Date:   Mon Dec 1 17:10:43 2025 +0100

    Updates of Gaudi SW version to CI&PT_HUD after 1.22.2 release  (#661)
    
    Signed-off-by: PatrykWo <patryk.wolsza@intel.com>

[33mcommit 321de9dd40560e1eedcacd6ec95c52ff75ea9ca7[m
Author: Patryk Wolsza <patryk.wolsza@intel.com>
Date:   Mon Dec 1 10:02:50 2025 +0100

    Removing external links from the main page (#638)
    
    Signed-off-by: PatrykWo <patryk.wolsza@intel.com>
    Co-authored-by: Michał Kuligowski <michal.kuligowski@intel.com>

[33mcommit dce7d129f1eef55b3c751164424f7e93715f2284[m
Author: Patryk Wolsza <patryk.wolsza@intel.com>
Date:   Mon Dec 1 10:02:21 2025 +0100

    Fix for links in docker md (#653)
    
    Signed-off-by: PatrykWo <patryk.wolsza@intel.com>
    Co-authored-by: Michał Kuligowski <michal.kuligowski@intel.com>

[33mcommit 49593fa70289b2fc9931376d0201002e54e68206[m
Author: Patryk Wolsza <patryk.wolsza@intel.com>
Date:   Fri Nov 28 14:45:46 2025 +0100

    Enabling qwen3-0.6b server/benchmark on a docker (#654)
    
    Signed-off-by: PatrykWo <patryk.wolsza@intel.com>

[33mcommit eb1f4c4423fbe2d25885e1718bcb3179504a0b8a[m
Author: Vivek Goel <vivek.goel@intel.com>
Date:   Fri Nov 28 09:47:47 2025 +0530

    Fix LoRA tests (#630)
    
    Fix [vllm plugin] hourly lora basic tests are failed due to
    https://github.com/vllm-project/vllm/pull/28545
    
    ---------
    
    Signed-off-by: Vivek Goel <vivek.goel@intel.com>
    Co-authored-by: Michał Kuligowski <michal.kuligowski@intel.com>

[33mcommit 35fbcaea98ad155241ea0d17cbf3acf1084a9c57[m
Author: Thomas Atta-Fosu <thomas.atta-fosu@intel.com>
Date:   Thu Nov 27 07:16:09 2025 -0800

    Dev/attafosu/port v0 optimizations for qwen2.5 vl (#643)
    
    Signed-off-by: attafosu <thomas.atta-fosu@intel.com>

[33mcommit af983a1286f52196f24549a3d95c304f4e10fb8f[m
Author: Monika Helfer <monika.helfer@intel.com>
Date:   Thu Nov 27 14:17:30 2025 +0100

    Docs: Update to Gaudi 1.22.2 and vLLM 0.11.2 (#631)
    
    I updated the versions and hid the unified attention feature as it will
    be supported in 1.24.0.
    
    Signed-off-by: mhelf-intel <monika.helfer@intel.com>
    Co-authored-by: Michał Kuligowski <michal.kuligowski@intel.com>

[33mcommit a18dee4a965387c8ec92a9d68f82b3e668285104[m
Author: Patryk Wolsza <patryk.wolsza@intel.com>
Date:   Thu Nov 27 09:41:01 2025 +0100

    Implement changes to add logging for Docker compose benchmark (#641)
    
    Signed-off-by: PatrykWo <patryk.wolsza@intel.com>
    Signed-off-by: Patryk Wolsza <patryk.wolsza@intel.com>
    Co-authored-by: Copilot <175728472+Copilot@users.noreply.github.com>

[33mcommit 180e798c2018eda0960a067304d5df81fdbf2634[m
Author: Artur Fierka <artur.fierka@intel.com>
Date:   Thu Nov 27 08:56:54 2025 +0100

    Add missing quantization files (#639)
    
    Ported missing:
    - `quantization_config` subdirectory
    - `convert.py` script
    from https://github.com/HabanaAI/vllm-hpu-extension project to
    calibration subdirectory in plugin.
    
    ---------
    
    Signed-off-by: Artur Fierka <artur.fierka@intel.com>
    Co-authored-by: Copilot <175728472+Copilot@users.noreply.github.com>

[33mcommit e38c8e9e28d0ac99b0d5364445b803d53e16bbb8[m
Author: Konrad Zawora <kzawora@habana.ai>
Date:   Wed Nov 26 16:01:44 2025 +0100

    Prepare Unified Attention biases on HPU + add NumPy memory pooling (#550)
    
    This PR moves bias preparation logic from CPU to HPU - moving causal
    bias creation alone can reduce batch preparation time >2x (~20ms down to
    ~8ms), with more to be gained on top of that with shared biases. Causal
    biases give the most benefit, although we can shave couple of
    milliseconds by also doing shared biases on HPU - although given their
    dynamicity (depending on number of shared tokens), it's hard to make
    them work under static shapes, so I've implemented some heuristics that
    cause in HPU execution when it's deemed beneficial, and fall back to CPU
    otherwise.
    
    Also, this seems to affect torch.compile much more than lazy - I've
    observed much bigger performance gains there and it now seems to perform
    close-ish to lazy (and it used to be >2x slower)
    
    This also includes a fancy memory pooling optimization for persistent
    numpy buffers, used for padding (in `hpu_tensor` method) - we don't do
    online padding anymore, we pre-allocate padded ndarray filled with
    certain value and store it in LRU cache for later use. If someone
    requests a placeholder larger than whatever is in the array, we extend
    the memory pool to accommodate that - then, when someone requests a
    smaller placeholder, we just reuse the previously allocated larger one,
    and just trim it to size.
    
    In the end, I've managed to get the host overheads of batch preparation
    down to ~3ms (GSM8k scenario), compared to ~20 ms with current numpy
    implementation, or ~120ms compared to previous PyTorch implementation.
    Pretty cool, I guess.
    
    If you want, you can disable all the stuff I added by setting
    `hpu_bias_acceleration` to `False` and `hpu_tensor_online_padding` to
    `True`. I used these for A/B testing and left them as a fallback in case
    something's broken and I haven't caught it.
    
    ---------
    
    Signed-off-by: Konrad Zawora <kzawora@habana.ai>
    Co-authored-by: Agata Dobrzyniewicz <160237065+adobrzyn@users.noreply.github.com>

[33mcommit f96e7cd2fd428182caff87245c1f418a3d511149[m
Author: Luca Calabria <luca.calabria@intel.com>
Date:   Wed Nov 26 12:21:42 2025 +0100

    disabled interleaved sliding window llama4 (#616)
    
    The fix has been done because vllm-upstream changed the the way to check
    if **interleaved_sliding_window** must be enabled or not. The new check
    doens't fit LLama4 Maverick model config.
    
    vllm-upstream v0.11.0 returns is_interleaved=False:
    
    ```
    def is_interleaved(config: PretrainedConfig) -> bool:
        """
        Detect if the model with this config is used with interleaved attention.
        """
        text_config = config.get_text_config()
        if layer_types := getattr(text_config, "layer_types", None):
            interleaved_types = {"full_attention", "sliding_attention"}
            return interleaved_types.issubset(layer_types)
        return False
    ```
    
    vllm-upstream latest returns (wrongly/unexpected) is_interleaved=True:
    
    ```
    def is_interleaved(config: PretrainedConfig) -> bool:
        """
        Detect if the model with this config is used with interleaved attention.
        """
        text_config = config.get_text_config()
        if layer_types := getattr(text_config, "layer_types", None):
            return len(set(layer_types)) > 1
        return False
    ```
    
    Adding the extra check self.sliding_window==None on plugin solve the
    issue. Runtime performance look still good
    
    ---------
    
    Signed-off-by: Luca Calabria <luca.calabria@intel.com>
    Co-authored-by: Copilot <175728472+Copilot@users.noreply.github.com>

[33mcommit 54fab50c309d32ee4e18d9720ea2aff5c127fc36[m
Author: Luca Calabria <luca.calabria@intel.com>
Date:   Wed Nov 26 12:17:20 2025 +0100

    fix assert failure on hpu_paged_attn (#598)
    
    Vllm repo added an assert to check builder subclass
    ```
     ERROR 11-19 09:59:12 [multiproc_executor.py:743]     attn_backend = create_chunked_local_attention_backend(
     ERROR 11-19 09:59:12 [multiproc_executor.py:743]   File "/root/vllm124/vllm/vllm/attention/layers/chunked_local_attention.py", line 39, in create_chunked_local_attention_backend
     ERROR 11-19 09:59:12 [multiproc_executor.py:743]     assert issubclass(underlying_builder, AttentionMetadataBuilder)
    ```
    We need to update vllm-gaudi side to pass the assert adding inherit from
    AttentionMetadataBuilder
    
    ---------
    
    Signed-off-by: Luca Calabria <luca.calabria@intel.com>
    Signed-off-by: PatrykWo <patryk.wolsza@intel.com>
    Signed-off-by: Patryk Wolsza <patryk.wolsza@intel.com>
    Signed-off-by: Tianmu Li <tianmu.li@intel.com>
    Signed-off-by: Chendi Xue <chendi.xue@intel.com>
    Signed-off-by: Chendi.Xue <chendi.xue@intel.com>
    Signed-off-by: Agata Dobrzyniewicz <adobrzyniewicz@habana.ai>
    Signed-off-by: Chen Haifeng <haifeng.chen@intel.com>
    Signed-off-by: Artur Fierka <artur.fierka@intel.com>
    Signed-off-by: Iryna Boiko <iboiko@habana.ai>
    Signed-off-by: slokesha <slokeshappa@habana.ai>
    Signed-off-by: attafosu <thomas.atta-fosu@intel.com>
    Signed-off-by: Chendi Xue <Chendi.Xue@intel.com>
    Signed-off-by: Xinyu Chen <xinyu1.chen@intel.com>
    Signed-off-by: Wuxun Zhang <wuxun.zhang@intel.com>
    Signed-off-by: Kamil Kaczor <kamil.kaczor@intel.com>
    Signed-off-by: Michal Adamczyk <michal.adamczyk@intel.com>
    Signed-off-by: Kavulya, Soila P <soila.p.kavulya@intel.com>
    Signed-off-by: Konrad Zawora <kzawora@habana.ai>
    Signed-off-by: Jimin Ha <jimin.ha@intel.com>
    Signed-off-by: Mohit Deopujari <mdeopujari@habana.ai>
    Signed-off-by: Harish Subramony <hsubramony@habana.ai>
    Signed-off-by: Karol Damaszke <kdamaszke@habana.ai>
    Signed-off-by: Vivek Goel <vivek.goel@intel.com>
    Signed-off-by: Spurthi Lokeshappa <slokeshappa@habana.ai>
    Signed-off-by: Agata Dobrzyniewicz <160237065+adobrzyn@users.noreply.github.com>
    Co-authored-by: Patryk Wolsza <patryk.wolsza@intel.com>
    Co-authored-by: Copilot <175728472+Copilot@users.noreply.github.com>
    Co-authored-by: Tianmu Li <tianmu.li@intel.com>
    Co-authored-by: Chendi.Xue <chendi.xue@intel.com>
    Co-authored-by: Agata Dobrzyniewicz <160237065+adobrzyn@users.noreply.github.com>
    Co-authored-by: Haifeng Chen <jerrychenhf@yahoo.com>
    Co-authored-by: Michał Kuligowski <michal.kuligowski@intel.com>
    Co-authored-by: Artur Fierka <artur.fierka@intel.com>
    Co-authored-by: Iryna Boiko <iboiko@habana.ai>
    Co-authored-by: Spurthi Lokeshappa <slokeshappa@habana.ai>
    Co-authored-by: Konrad Zawora <kzawora@habana.ai>
    Co-authored-by: Thomas Atta-Fosu <thomas.atta-fosu@intel.com>
    Co-authored-by: Xinyu Chen <xinyu1.chen@intel.com>
    Co-authored-by: Wuxun Zhang <wuxun.zhang@intel.com>
    Co-authored-by: Kamil Kaczor <kamil.kaczor@intel.com>
    Co-authored-by: Michal Adamczyk <madamczyk@habana.ai>
    Co-authored-by: Soila Kavulya <soila.p.kavulya@intel.com>
    Co-authored-by: Jimin Ha <jimin.ha@intel.com>
    Co-authored-by: Mohit Deopujari <mdeopujari@habana.ai>
    Co-authored-by: Harish Subramony <hsubramony@habana.ai>
    Co-authored-by: Karol Damaszke <kdamaszke@habana.ai>
    Co-authored-by: Vivek Goel <vivek.goel@intel.com>
    Co-authored-by: Haifeng Chen <haifeng.chen@intel.com>

[33mcommit d621578a571521526a692a3e90790d307bdaa6b1[m
Author: Tianmu Li <tianmu.li@intel.com>
Date:   Wed Nov 26 02:07:09 2025 -0800

    Handle incorrect appended logits indices during unified attention + async scheduling warmup (#632)
    
    …warmup)
    
    Signed-off-by: Tianmu Li <tianmu.li@intel.com>

[33mcommit f6830aead256c02efe913b3af0d94554227b98b1[m
Author: Patryk Wolsza <patryk.wolsza@intel.com>
Date:   Wed Nov 26 10:26:50 2025 +0100

    1.22.2 Updates to matrix (#626)
    
    Signed-off-by: PatrykWo <patryk.wolsza@intel.com>

[33mcommit b44f489f9ad725024ef5e304f6413726c3c26ce5[m
Author: Monika Helfer <monika.helfer@intel.com>
Date:   Wed Nov 26 10:25:42 2025 +0100

    Add version variable (#628)
    
    Signed-off-by: mhelf-intel <monika.helfer@intel.com>

[33mcommit 8d9286466ff01037e4bac8032156ca53a37d8a50[m
Author: Thomas Atta-Fosu <thomas.atta-fosu@intel.com>
Date:   Wed Nov 26 00:29:52 2025 -0800

    multimodal support for unified attn (#423)
    
    - Enables Multimodal support for unified attention
    
    ---------
    
    Signed-off-by: attafosu <thomas.atta-fosu@intel.com>
    Signed-off-by: Thomas Atta-Fosu <thomas.atta-fosu@intel.com>
    Co-authored-by: Copilot <175728472+Copilot@users.noreply.github.com>

[33mcommit 8ef73e609e636e80cc013709753a854f3cb6b1cb[m
Author: Haifeng Chen <haifeng.chen@intel.com>
Date:   Wed Nov 26 15:16:48 2025 +0800

    Spec decode: fix MTP function broken by using unified MTP method names (#633)
    
    vLLM PR (https://github.com/vllm-project/vllm/pull/25232) has
    consolidated speculative decode method name for MTP. We need to use
    "mtp" in the code instead of other names.
    
    Signed-off-by: Chen Haifeng <haifeng.chen@intel.com>

[33mcommit 5c09ab712aaf835b33e39404ad828e06d5a88ad2[m
Author: Haifeng Chen <haifeng.chen@intel.com>
Date:   Tue Nov 25 23:32:02 2025 +0800

    Spec decode: support of more than one num speculative tokens  (#609)
    
    The main logic of support num_speculative_tokens resides in the propose
    method of the refactored HpuEagleProposer class.
    The generation of the first draft token is special because it can reuse
    the existing attention metadata of the main model. If
    num_speculative_tokens > 1, we have to do a token by token generation of
    the remaining draft tokens. For each remaining draft token, we:
    1. Prepare the attn metadata
    2. Prepare the inputs (input token ids, positions, hidden states)
    3. Call the draft model to generate a new draft token
    
    The draft tokens generated in each iteration (including the first) stack
    to get the output of the propose in the shape of [batch_size,
    num_speculative_tokens].
    
    To prepare the attn metadata, we need to pass the original block_table.
    While the original block_table includes all the decodes and prefills,
    when running, the decode and prefills are batched to different forwards
    and we need to correctly address the block_table based on batching
    information.
    
    ---------
    
    Signed-off-by: Chen Haifeng <haifeng.chen@intel.com>

[33mcommit 642ae02a58b75122c2c00444fc9c02985c0de978[m
Author: Paweł Olejniczak <polejniczakx@habana.ai>
Date:   Tue Nov 25 14:53:06 2025 +0100

    [FIX_FOR_VLLM_LATEST] Fix crash after the select_experts became a non-static method (#629)
    
    Signed-off-by: Paweł Olejniczak <polejniczakx@habana.ai>

[33mcommit d8c64e6010646ec2391ea34c71163ea200489b8a[m
Author: Monika Helfer <monika.helfer@intel.com>
Date:   Tue Nov 25 11:28:35 2025 +0100

    Add the missing requirement (#627)
    
    Signed-off-by: mhelf-intel <monika.helfer@intel.com>

[33mcommit 2d4378ddc6b9adea404b28b441137ff36b400b89[m
Author: Patryk Wolsza <patryk.wolsza@intel.com>
Date:   Tue Nov 25 11:05:08 2025 +0100

    Updates to validated models list (#614)
    
    Signed-off-by: PatrykWo <patryk.wolsza@intel.com>

[33mcommit 7acb528967c467acac007a27fbd9770a6e99f62c[m
Author: Michał Kuligowski <michal.kuligowski@intel.com>
Date:   Tue Nov 25 10:07:56 2025 +0100

    Call shutdown_inc to mitiagate driver worker teardown order (#511)
    
    Signed-off-by: Michał Kuligowski <michal.kuligowski@intel.com>

[33mcommit 770cc9c95553a5a446611dc52d6b4209e21c2970[m
Author: Monika Helfer <monika.helfer@intel.com>
Date:   Tue Nov 25 07:42:35 2025 +0100

    Add a plugin for variable support in Markdown (#554)
    
    [This plugin](https://mkdocs-macros-plugin.readthedocs.io/en/latest/)
    supports using variables in Markdown, which is useful for defining some
    values or versions (such as PT_VERSION) in one place. I tested it
    locally and it's working correctly:
    <img width="1075" height="221" alt="image"
    src="https://github.com/user-attachments/assets/4a823378-780b-4e37-9efd-e3bf0c3e3a6c"
    />
    
    Signed-off-by: mhelf-intel <monika.helfer@intel.com>
    Co-authored-by: Patryk Wolsza <patryk.wolsza@intel.com>

[33mcommit 5b805480d7fd4bc7d45864d047c1bf2deb0fdd3f[m
Author: Monika Helfer <monika.helfer@intel.com>
Date:   Tue Nov 25 07:34:29 2025 +0100

    Add the missing step to the Quick Start guide (#599)
    
    I added the missing Pinning CPU Cores for Memory Access Coherence
    section to the Quick Start guide and replaced the unreviewed Quick Start
    content in the README with links to the official guides. This avoids
    duplicating content across multiple locations and prevents potential
    inconsistencies.
    
    Signed-off-by: mhelf-intel <monika.helfer@intel.com>

[33mcommit f71c4ca532a6e9a4ee46f1b6ca87db482537624f[m
Author: Agata Dobrzyniewicz <160237065+adobrzyn@users.noreply.github.com>
Date:   Mon Nov 24 16:15:00 2025 +0100

    New codeowners (#622)
    
    Signed-off-by: Agata Dobrzyniewicz <adobrzyniewicz@habana.ai>

[33mcommit bc135721faddbad2ce78807639e019808e400b89[m
Author: Neelesh Gokhale <neelesh.gokhale@intel.com>
Date:   Mon Nov 24 18:04:09 2025 +0530

    Cherry-pick release docker cmdline fixes, WA and long context support (#576)
    
    Signed-off-by: Neelesh Gokhale <neelesh.gokhale@intel.com>
    Co-authored-by: Michal Gawarkiewicz <michal.gawarkiewicz@intel.com>
    Co-authored-by: Michał Kuligowski <michal.kuligowski@intel.com>

[33mcommit e18a075722853d40decf75ca69348e0f7733452a[m
Author: Iryna Boiko <iboiko@habana.ai>
Date:   Fri Nov 21 16:56:17 2025 +0100

    [FIX_FOR_VLLM_LATEST] Fix for PR29121, revert of PR575 (#615)
    
    Signed-off-by: Iryna Boiko <iboiko@habana.ai>

[33mcommit b530e99d7ccb3ad332841fbd2eaaae766af306fa[m
Author: Michal Muszynski <141021743+mmuszynskihabana@users.noreply.github.com>
Date:   Fri Nov 21 16:52:23 2025 +0100

    Allow building vllm-plugin docker for ubuntu with upstream torch (#607)
    
    To create vllm-plugin docker for Ubuntu with torch package taken from
    upstream we need to modify 'FROM' directive - image should be based on
    pytorch-upstream-installer.
    
    'TORCH_TYPE_SUFFIX' arg will have one of two values:
    - empty string (default)
    - 'upstream-'
    
    It's the same change as
    https://github.com/vllm-project/vllm-gaudi/pull/155 but this time for
    Ubuntu instead of RHEL.
    
    Signed-off-by: Michal Muszynski <mmuszynski@habana.ai>

[33mcommit 0b1c374d164d3b64d0a1fc0464d38c14a8620039[m
Author: Artur Fierka <artur.fierka@intel.com>
Date:   Fri Nov 21 10:26:04 2025 +0100

    Update CODEOWNERS (#592)
    
    Remove Marcin
    
    Signed-off-by: Artur Fierka <artur.fierka@intel.com>

[33mcommit a066667a53acadbb99d6a80f3081fc51ebdcdba1[m
Author: Agata Dobrzyniewicz <160237065+adobrzyn@users.noreply.github.com>
Date:   Fri Nov 21 09:09:23 2025 +0100

    Check min package version for use_output_tensor_in_matmulqk (#606)
    
    Signed-off-by: Agata Dobrzyniewicz <adobrzyniewicz@habana.ai>
    Signed-off-by: Agata Dobrzyniewicz <160237065+adobrzyn@users.noreply.github.com>

[33mcommit 0e087987357e81310c0f2eede2acd7ac3c9a9537[m
Author: Tianmu Li <tianmu.li@intel.com>
Date:   Thu Nov 20 23:57:58 2025 -0800

    Add async scheduling for unified attention (#414)
    
    #134 + #184 + #360 for unified attention
    
    ---------
    
    Signed-off-by: Tianmu Li <tianmu.li@intel.com>

[33mcommit c25258a845b19d0fa4aa4815b28b6895f2435789[m
Author: Chendi.Xue <chendi.xue@intel.com>
Date:   Thu Nov 20 12:42:31 2025 -0600

    [NIXL] add PD gaudi direct test (#591)
    
    need t wait until #590 merged
    
    ---------
    
    Signed-off-by: Chendi Xue <chendi.xue@intel.com>
    Signed-off-by: Chendi.Xue <chendi.xue@intel.com>
    Co-authored-by: Copilot <175728472+Copilot@users.noreply.github.com>
    Co-authored-by: Michał Kuligowski <michal.kuligowski@intel.com>

[33mcommit a59011aaf44058e06f4a00a1d1134e9e0cba7f73[m
Author: Haifeng Chen <haifeng.chen@intel.com>
Date:   Fri Nov 21 01:08:04 2025 +0800

    Refactor part of spec decode structure identical to vLLM (#544)
    
    Currently a lot of spec decode related code were implemented directly in
    hpu model runner and model runner will gets piled up into complexity
    when more spec decode improvements and features are added.
    
    We need to follow the same structure as vLLM implementation to put as
    much spec decode related code to the corresponding proposer.
    
    This refactor will make a basic refactor following this structure idea
    but try touch as less as possible the detailed implementation logic so
    that the functionality is not impacted.
    
    ---------
    
    Signed-off-by: Chen Haifeng <haifeng.chen@intel.com>

[33mcommit 143531f3f8171da202cf57b626c798ae838b749c[m
Author: Spurthi Lokeshappa <slokeshappa@habana.ai>
Date:   Thu Nov 20 22:11:19 2025 +0530

    WarmUp for Pooling - Embed Task (#170)
    
    Enabled Warmup for Pooling
    
    ---------
    
    Signed-off-by: slokesha <slokeshappa@habana.ai>
    Signed-off-by: Agata Dobrzyniewicz <adobrzyniewicz@habana.ai>
    Signed-off-by: attafosu <thomas.atta-fosu@intel.com>
    Signed-off-by: Chendi Xue <Chendi.Xue@intel.com>
    Signed-off-by: Tianmu Li <tianmu.li@intel.com>
    Signed-off-by: Xinyu Chen <xinyu1.chen@intel.com>
    Signed-off-by: PatrykWo <patryk.wolsza@intel.com>
    Signed-off-by: Wuxun Zhang <wuxun.zhang@intel.com>
    Signed-off-by: Kamil Kaczor <kamil.kaczor@intel.com>
    Signed-off-by: Michal Adamczyk <michal.adamczyk@intel.com>
    Signed-off-by: Kavulya, Soila P <soila.p.kavulya@intel.com>
    Signed-off-by: Konrad Zawora <kzawora@habana.ai>
    Signed-off-by: Jimin Ha <jimin.ha@intel.com>
    Signed-off-by: Mohit Deopujari <mdeopujari@habana.ai>
    Signed-off-by: Harish Subramony <hsubramony@habana.ai>
    Signed-off-by: Karol Damaszke <kdamaszke@habana.ai>
    Signed-off-by: Vivek Goel <vivek.goel@intel.com>
    Signed-off-by: Iryna Boiko <iboiko@habana.ai>
    Signed-off-by: Spurthi Lokeshappa <slokeshappa@habana.ai>
    Co-authored-by: Agata Dobrzyniewicz <160237065+adobrzyn@users.noreply.github.com>
    Co-authored-by: Chendi.Xue <chendi.xue@intel.com>
    Co-authored-by: Konrad Zawora <kzawora@habana.ai>
    Co-authored-by: Thomas Atta-Fosu <thomas.atta-fosu@intel.com>
    Co-authored-by: Tianmu Li <tianmu.li@intel.com>
    Co-authored-by: Xinyu Chen <xinyu1.chen@intel.com>
    Co-authored-by: PatW <patryk.wolsza@intel.com>
    Co-authored-by: Wuxun Zhang <wuxun.zhang@intel.com>
    Co-authored-by: Kamil Kaczor <kamil.kaczor@intel.com>
    Co-authored-by: Michal Adamczyk <madamczyk@habana.ai>
    Co-authored-by: Soila Kavulya <soila.p.kavulya@intel.com>
    Co-authored-by: Jimin Ha <jimin.ha@intel.com>
    Co-authored-by: Mohit Deopujari <mdeopujari@habana.ai>
    Co-authored-by: Harish Subramony <hsubramony@habana.ai>
    Co-authored-by: Karol Damaszke <kdamaszke@habana.ai>
    Co-authored-by: Vivek Goel <vivek.goel@intel.com>
    Co-authored-by: Iryna Boiko <iboiko@habana.ai>

[33mcommit e2b4bd3aacf9e17744362d4508e462908595ebf4[m
Author: Chendi.Xue <chendi.xue@intel.com>
Date:   Thu Nov 20 10:29:58 2025 -0600

    [GITHUB ACTION] enable --privileged for docker run (#605)
    
    With coming work for integrating nixl running on RDMA, we will need to
    provide 'privileged' permission otherwise,
    docker won't be able to get network_fd
    
    Signed-off-by: Chendi Xue <chendi.xue@intel.com>

[33mcommit 6516bfc0f0e038ae85d1cea31cd17dd2b27df3f2[m
Author: Iryna Boiko <iboiko@habana.ai>
Date:   Thu Nov 20 13:55:22 2025 +0100

    hourly tests: run_spec_decode_ngram_test enabled, run_gemma3_test disabled (#596)
    
    Signed-off-by: Iryna Boiko <iboiko@habana.ai>

[33mcommit 57824823db523b70e9e920e864e05e970df3ae0a[m
Author: Artur Fierka <artur.fierka@intel.com>
Date:   Thu Nov 20 12:46:08 2025 +0100

    Fix reverse inull security issue (#588) (#594)
    
    Cherry-pick of #588 from `0.11.1`
    
    Signed-off-by: Artur Fierka <artur.fierka@intel.com>

[33mcommit fb990e326f17686d1bb8e7f13124e4f7942e091d[m
Author: Haifeng Chen <jerrychenhf@yahoo.com>
Date:   Thu Nov 20 18:41:40 2025 +0800

    Fix the spec decode the latest functionality broken (#593)
    
    The latest vllm code will not schedule more spec decode tokens if the
    latest output token is already hit the output length limit.
    So there will be cases that the num draft token is 0 for some sequence
    in the decoding process. This cases are not properly handled in the
    current spec decode code and needs to be addressed:
    1. There is a bug in our rejection_sampler for handling the case that
    one or more sequence doesn't have draft tokens
    2. The decode assert on spec_decode_metadata for not None but this will
    happen at the end of decoding phase that all sequence reach their limit
    and no draft tokens are needed.
    
    ---------
    
    Signed-off-by: Chen Haifeng <haifeng.chen@intel.com>
    Co-authored-by: Michał Kuligowski <michal.kuligowski@intel.com>

[33mcommit 65ed5c26bcc86f904eb772180b6f6c6039f8503b[m
Author: Agata Dobrzyniewicz <160237065+adobrzyn@users.noreply.github.com>
Date:   Thu Nov 20 09:37:52 2025 +0100

    Port: Fix prefix caching automatic off with conti pa (#583) (#586)
    
    Signed-off-by: Agata Dobrzyniewicz <adobrzyniewicz@habana.ai>

[33mcommit 767a4e988667f5a2af90a5ef3b61b764dab5fc8d[m
Author: Chendi.Xue <chendi.xue@intel.com>
Date:   Thu Nov 20 02:34:05 2025 -0600

    [PD][NIXL]Fix bug after upstream adding virtual block_size support (#590)
    
    PD test is commented out in GAUDISW-243609
    
    This PR is to bring it back.
    
    Root Cause of the error:
    after recent works in nixl_connector, GPU now support virtual block size
    which can be 2x to physical kernel size; And nixl_connector intended to
    always assume kernel shape layout is NHD, so the previous way we used in
    HPU to use HND shape does not work.
    
    Solution:
    This is PR is to now follow exact same shape as GPU
    
    ---------
    
    Signed-off-by: Chendi Xue <chendi.xue@intel.com>
    Signed-off-by: Chendi.Xue <chendi.xue@intel.com>
    Co-authored-by: Copilot <175728472+Copilot@users.noreply.github.com>

[33mcommit cff73437ac442939226133664c50d5a76fd871c1[m
Author: Tianmu Li <tianmu.li@intel.com>
Date:   Wed Nov 19 23:19:26 2025 -0800

    Fix async scheduling + request preemption (#589)
    
    Port #26385 fix in upstream gpu_model_runner.py
    
    Signed-off-by: Tianmu Li <tianmu.li@intel.com>

[33mcommit 87b6fe069c42eae223c1ecb67fe277bed9ef33f6[m
Author: Patryk Wolsza <patryk.wolsza@intel.com>
Date:   Wed Nov 19 17:03:30 2025 +0100

    Edit docker file to resolve conflicts issue243959 (#587)
    
    Changing yum parameters to resolve conflicts.
    
    ---------
    
    Signed-off-by: PatrykWo <patryk.wolsza@intel.com>
    Signed-off-by: Patryk Wolsza <patryk.wolsza@intel.com>
    Co-authored-by: Copilot <175728472+Copilot@users.noreply.github.com>

[33mcommit 0b2e5bb7290d1f5157e1801d06c2377137a26894[m
Author: Iryna Boiko <iboiko@habana.ai>
Date:   Wed Nov 19 11:08:19 2025 +0100

    Fix for PR24248 (#578)
    
    ACC is back from 0 to  0.91796875
    
    ---------
    
    Signed-off-by: Iryna Boiko <iboiko@habana.ai>

[33mcommit 8109325e36e2fa3a354e49fa78588926bb244dde[m
Author: Artur Fierka <artur.fierka@intel.com>
Date:   Tue Nov 18 12:33:46 2025 +0100

    Update hpu_model_runner.py (#582)
    
    Fix warmup_mode for unified_execute_model function
    
    Signed-off-by: Artur Fierka <artur.fierka@intel.com>

[33mcommit a8211ff1eb975badb6a6b584a71682ded006611c[m
Author: Agata Dobrzyniewicz <160237065+adobrzyn@users.noreply.github.com>
Date:   Tue Nov 18 07:42:29 2025 +0100

    Specify output tensor in matmul_qk - with version difference (#571)
    
    From: https://github.com/vllm-project/vllm-gaudi/pull/188
    
    ---------
    
    Signed-off-by: Agata Dobrzyniewicz <adobrzyniewicz@habana.ai>
    Signed-off-by: Agata Dobrzyniewicz <160237065+adobrzyn@users.noreply.github.com>
    Co-authored-by: Copilot <175728472+Copilot@users.noreply.github.com>
    Co-authored-by: Michał Kuligowski <michal.kuligowski@intel.com>

[33mcommit 78a48c60fc4c0064170cce5ba424d6281b8e0571[m
Author: Patryk Wolsza <patryk.wolsza@intel.com>
Date:   Mon Nov 17 17:03:05 2025 +0100

    Nixl deployment fixes (#573)
    
    The change is to fix NIXL deployment procedure.
    
    ---------
    
    Signed-off-by: PatrykWo <patryk.wolsza@intel.com>
    Signed-off-by: Patryk Wolsza <patryk.wolsza@intel.com>
    Co-authored-by: Copilot <175728472+Copilot@users.noreply.github.com>
    Co-authored-by: Iryna Boiko <iboiko@habana.ai>

[33mcommit 9de420b64621c797ecb1bfaa5d4f90efbda2f474[m
Author: Paweł Olejniczak <polejniczakx@habana.ai>
Date:   Mon Nov 17 15:25:39 2025 +0100

    [FIX_FOR_VLLM_LATEST] Fix crash after the sampled_token_ids type change (#575)
    
    sampled_token_ids was changed from list[list[int]] to list[list[int]]:
    https://github.com/vllm-project/vllm/pull/26368
    
    Signed-off-by: Paweł Olejniczak <polejniczakx@habana.ai>

[33mcommit 05c2470641928cd44827a38d6342a99b26bae907[m
Author: Monika Helfer <monika.helfer@intel.com>
Date:   Fri Nov 14 16:12:39 2025 +0100

    Readme updates and release notes for 0.10.2 (#565)
    
    This PR contains:
    - Basic Readme updates following the documentation review
    - New release notes for the upcoming release
    - A few minor cleanup fixes
    
    ---------
    
    Signed-off-by: mhelf-intel <monika.helfer@intel.com>
    Signed-off-by: Monika Helfer <monika.helfer@intel.com>
    Co-authored-by: Copilot <175728472+Copilot@users.noreply.github.com>

[33mcommit ecc67e1338d4a0e829151f1d514926ab033b7946[m
Author: Monika Helfer <monika.helfer@intel.com>
Date:   Fri Nov 14 14:09:37 2025 +0100

    Final documentation improvements and broken link fixes (#558)
    
    Signed-off-by: mhelf-intel <monika.helfer@intel.com>
    Co-authored-by: Michał Kuligowski <michal.kuligowski@intel.com>

[33mcommit 316349893bb708cefeb8e117f236c628cf1db416[m
Author: Agata Dobrzyniewicz <160237065+adobrzyn@users.noreply.github.com>
Date:   Fri Nov 14 10:49:53 2025 +0100

    UX fix: hide warmup logs (#539)
    
    If VLLM_ENABLE_EXPERIMENTAL_FLAGS is set to 0 or not set warmup will
    stay hidden with only progress bar.
    Enabling this flag will bring back old logs
    
    Additionally remove VLLM_USE_V1 flag
    
    Additionally all user flags are no longer experimental
    
    ---------
    
    Signed-off-by: Agata Dobrzyniewicz <adobrzyniewicz@habana.ai>
    Co-authored-by: Michał Kuligowski <michal.kuligowski@intel.com>

[33mcommit e3510f44cf4dfc46b199e6f03b20b79f6d7648fd[m
Author: Iryna Boiko <iboiko@habana.ai>
Date:   Fri Nov 14 09:52:53 2025 +0100

    Fix for PR546, adding float32 and float16 (#569)
    
    Signed-off-by: Iryna Boiko <iboiko@habana.ai>

[33mcommit 9abbdc05d6e4cd9cd670293c84cdddab996c19b2[m
Author: Iryna Boiko <iboiko@habana.ai>
Date:   Thu Nov 13 23:47:33 2025 +0100

    [FIX_FOR_VLLM_LATEST] fix pr28534 (#568)
    
    Signed-off-by: Iryna Boiko <iboiko@habana.ai>
    Co-authored-by: Michał Kuligowski <michal.kuligowski@intel.com>

[33mcommit 46bc9759a1953a17e99e4d2cb58cf4b4645770ea[m
Author: Michał Kuligowski <michal.kuligowski@intel.com>
Date:   Thu Nov 13 14:28:32 2025 +0100

    v0 cleanup (#563)
    
    Signed-off-by: Michał Kuligowski <michal.kuligowski@intel.com>

[33mcommit 271ce850684f9e74477101a1f017f2f768ebd349[m
Author: Daniel Socek <daniel.socek@intel.com>
Date:   Thu Nov 13 08:07:37 2025 -0500

    Automatically adjust VLLM_DECODE_BLOCK_BUCKET_MIN if it exceeds max_blocks (#432)
    
    # What does this PR do?
    
    During engine warmup, the max decode block bucket size
    `VLLM_DECODE_BLOCK_BUCKET_MAX` is capped based on the available
    `max_blocks`. However, the minimum bucket size
    `VLLM_DECODE_BLOCK_BUCKET_MIN` was not similarly constrained, which
    could lead to a configuration where VLLM_DECODE_BLOCK_BUCKET_MIN >
    VLLM_DECODE_BLOCK_BUCKET_MAX (or even > `max_blocks`). This invalid
    state causes runtime error.
    
    This PR ensures that `VLLM_DECODE_BLOCK_BUCKET_MIN` is automatically
    clamped to `max_blocks` (and not greater than
    `VLLM_DECODE_BLOCK_BUCKET_MAX`) during initialization, preventing
    invalid bucket size configurations.
    
    Signed-off-by: Daniel Socek <daniel.socek@intel.com>
    Co-authored-by: Michał Kuligowski <michal.kuligowski@intel.com>

[33mcommit f3f66f6a7cc323a82501ef77ebe4a26b51ff8914[m
Author: Monika Helfer <monika.helfer@intel.com>
Date:   Thu Nov 13 13:17:13 2025 +0100

    Replace the deprecated logo (#564)
    
    The old Habana logo is deprecated and should no longer be used. I
    replaced it with the current logo. For the favicon, I made a slight
    modification since the original logo is too long to display properly in
    a browser tab, which requires a square format. As a result, the logo has
    been slightly compressed, but only for this specific purpose. It looks
    like this:
    <img width="768" height="212" alt="image"
    src="https://github.com/user-attachments/assets/68bddac8-8aef-41c0-acd1-f30d934f8de3"
    />
    
    Signed-off-by: mhelf-intel <monika.helfer@intel.com>

[33mcommit 6cd337715d837d259114295b041f6712478b9057[m
Author: Paweł Olejniczak <polejniczakx@habana.ai>
Date:   Thu Nov 13 12:46:25 2025 +0100

    [FIX_FOR_VLLM_LATEST] Rename get_input_embeddings and get_multimodal_embeddings (#561)
    
    New names: embed_input_ids and embed_multimodal
    The change is enforced by the upstream.
    
    Signed-off-by: Paweł Olejniczak <polejniczakx@habana.ai>

[33mcommit c23b5e227563a3da7b3192ac7e2a8094182eb58b[m
Author: zhao, zhenhui <zhenzhao@habana.ai>
Date:   Thu Nov 13 16:55:24 2025 +0800

    Skip HPUGraph exceed max_cudagraph_capture_size (#551)
    
    Port max_cudagraph_capture_size from v0
    
    https://github.com/HabanaAI/vllm-fork/blob/deepseek_r1/vllm/worker/hpu_model_runner.py#L1027C10-L1028
    
    Signed-off-by: zhenzhao <zhenzhao@habana.ai>

[33mcommit 495e4c613ecc82fb8acf973f13a2dc12d3391b28[m
Author: Kacper Pietkun <kpietkun@habana.ai>
Date:   Thu Nov 13 08:15:08 2025 +0100

    Fix for compiled_methods (#559)
    
    Fix for a `_compile_methods` function - now it won't throw an exception
    if a method is not a member of a model
    
    Signed-off-by: Kacper Pietkun <kpietkun@habana.ai>

[33mcommit 52ecddcda79a33acea0ddc46aaf3e14f46068153[m
Author: Iryna Boiko <iboiko@habana.ai>
Date:   Wed Nov 12 23:59:15 2025 +0100

    [FIX_FOR_VLLM_LATEST] Fix upstream execute_model crash (#546)
    
    Signed-off-by: Iryna Boiko <iboiko@habana.ai>

[33mcommit e3c20ec22d556af414fabbff12c20dc8070ec14f[m
Author: Agata Dobrzyniewicz <160237065+adobrzyn@users.noreply.github.com>
Date:   Wed Nov 12 09:46:26 2025 +0100

    [Docs] Readme for bucketing from file + env var added (#545)
    
    Signed-off-by: Agata Dobrzyniewicz <adobrzyniewicz@habana.ai>
    Signed-off-by: Agata Dobrzyniewicz <160237065+adobrzyn@users.noreply.github.com>
    Signed-off-by: PatrykWo <patryk.wolsza@intel.com>
    Signed-off-by: mhelf-intel <monika.helfer@intel.com>
    Co-authored-by: Copilot <175728472+Copilot@users.noreply.github.com>
    Co-authored-by: PatrykWo <patryk.wolsza@intel.com>
    Co-authored-by: Michał Kuligowski <michal.kuligowski@intel.com>
    Co-authored-by: Monika Helfer <monika.helfer@intel.com>

[33mcommit 15dddd14a2ddab270846d351df360614ff437125[m
Author: Miroslav Goncharenko <miroslav.goncharenko@intel.com>
Date:   Wed Nov 12 07:42:50 2025 +0100

    Fix typo in bucketing_file.txt (#553)
    
    Signed-off-by: Miroslav Goncharenko <miroslav.goncharenko@intel.com>

[33mcommit 37726a41345cfd11dcb4f8128db8fd059eec4d2c[m
Author: Soila Kavulya <soila.p.kavulya@intel.com>
Date:   Mon Nov 10 14:47:13 2025 -0800

    [SW-242523] Support per-tensor FP8 scaling (#483)
    
    Support FP8 inference for models with per-tensor scales
    
    ---------
    
    Signed-off-by: Kavulya, Soila P <soila.p.kavulya@intel.com>

[33mcommit 0a6113b7520d271c2de0ade2950418bd926ec44e[m
Author: Kacper Pietkun <kacper.pietkun00@gmail.com>
Date:   Fri Nov 7 19:39:05 2025 +0100

    Add tests for custom operator implementation correctness (#457)
    
    I added tests for custom ops defined in `vllm_gaudi/ops`:
    - For the tests of ops that are not using cuda kernels - native ops and
    hpu ops are triggered for the same input and their outputs are compared
    - For others tests that are using cuda kernels (so cannot be called with
    vllm-gaudi plugin) I created separate directory to store some predefined
    small tensors - weights, inputs and outputs. These tensors are too big
    to hardcode them in tests, however their sizes were adjusted, so all of
    them weight less than 3MB in total. Tensors are stored in a .safetensors
    format. Such tests run hpu ops with loaded inputs and weights and
    compare their outputs with the loaded outputs.
    
    ---------
    
    Signed-off-by: Kacper Pietkun <kpietkun@habana.ai>
    Signed-off-by: Kacper Pietkun <kacper.pietkun00@gmail.com>
    Co-authored-by: Michał Kuligowski <michal.kuligowski@intel.com>
    Co-authored-by: Copilot <175728472+Copilot@users.noreply.github.com>

[33mcommit 16683615979aa1a40f56cefaaefd231a3731fc5a[m
Author: Kamil Kaczor <kamil.kaczor@intel.com>
Date:   Fri Nov 7 19:33:38 2025 +0100

    Fix unified preemption no attr found (#528)
    
    Running:
    `VLLM_SKIP_WARMUP=false \
    VLLM_UNIFIED_ATTN=1 \
    PT_HPU_LAZY_MODE=1 \
    vllm serve granite-3b-code-instruct-128k/ \
        --tensor-parallel-size 1 \
        --max-num-seqs 42 \
        --block-size 128 \
        --disable-log-stats \
        --dtype bfloat16 \
        --max-model-len 2048 \
        --async_scheduling`
    and user:
    `python -m vllm.entrypoints.cli.main bench serve \
        --backend vllm \
        --model granite-3b-code-instruct-128k/ \
        --tokenizer granite-3b-code-instruct-128k/ \
        --percentile-metrics ttft,tpot,itl,e2el \
        --metric-percentiles 50,90,99 \
        --ignore-eos \
        --num-prompts 42 \
        --port 8000 \
        --dataset-name random \
        --random-input-len 1024 \
        --random-output-len 1024 \
        --max-concurrency 42`
    we run into resumed_req_ids has no attribute "resumed_req_ids". This is
    covering this case.
    
    ---------
    
    Signed-off-by: Kamil Kaczor <kamil.kaczor@intel.com>
    Signed-off-by: Kamil Kaczor <kkaczor@habana.ai>
    Co-authored-by: Copilot <175728472+Copilot@users.noreply.github.com>

[33mcommit 93f6fb52ffd717c901b0fc3b62d0b0c8db840d56[m
Author: Artur Fierka <artur.fierka@intel.com>
Date:   Fri Nov 7 19:33:02 2025 +0100

    Enable FP8 with unified attention (#516)
    
    Signed-off-by: Artur Fierka <artur.fierka@intel.com>

[33mcommit 2bb03f34c9fa6b214613d435070b91756e669246[m
Author: Agata Dobrzyniewicz <160237065+adobrzyn@users.noreply.github.com>
Date:   Fri Nov 7 19:32:27 2025 +0100

    [Docs] Unified attn style update (#533)
    
    Signed-off-by: Agata Dobrzyniewicz <adobrzyniewicz@habana.ai>
    Signed-off-by: Agata Dobrzyniewicz <160237065+adobrzyn@users.noreply.github.com>
    Signed-off-by: PatrykWo <patryk.wolsza@intel.com>
    Co-authored-by: Copilot <175728472+Copilot@users.noreply.github.com>
    Co-authored-by: PatrykWo <patryk.wolsza@intel.com>
    Co-authored-by: Michał Kuligowski <michal.kuligowski@intel.com>

[33mcommit 509b7818e43ab8a2860e682d28309c25e3193742[m
Author: Kamil Kaczor <kamil.kaczor@intel.com>
Date:   Fri Nov 7 19:30:14 2025 +0100

    Fix missing non-causal buckets (#540)
    
    WIthout this for small BS values we are skipping warming-up non-causal
    buckets.
    
    Signed-off-by: Kamil Kaczor <kamil.kaczor@intel.com>
    Co-authored-by: Michał Kuligowski <michal.kuligowski@intel.com>

[33mcommit 2878f6245e73a0becd38ab44b4dab36ffa4d4ab9[m
Author: Louie Tsai <louie.tsai@intel.com>
Date:   Fri Nov 7 08:24:32 2025 -0800

    [New Feature] Add cpu core pinning to vllm-server to improve performance. (#502)
    
    ## Purpose
    
    - Identify a performance issue on GNR.
    - Fix the performance gap by pinning the right number of CPUs for
    different models, and maintain the model and #cpu mapping in CSV files
    as lookup tables.
    - Add some python scripts to generate right CPU id list and pinning CPU
    for vllm-server with a docker-compose.override.yaml file.
    - We also apply same workflows on EMR.
    - It not only help on Gaudi performance and also release other idle CPU
    for other CPU workloads.
    
    docker-compose.override.yaml  example
    
    _services:
        vllm-server:
    cpuset: "21,22,23,45,46,47,69,70,71,93,94,95,117,118,119,141,42,143"
           cpus: "18_"
    
    
    ## Test Plan
    manually tested.
    ## Test Result
    
     ### GNR
    
    By pinning different number of CPUs, we could see different throughput,
    TTFT and TPOT on different models.
    
    **Llama3.1 405B**
    For Llama3.1 405B, 18 CPU cores gave the best performance, so we map
    Llama3.1 405B with number of CPU "18"
    
    <img width="633" height="289" alt="image"
    src="https://github.com/user-attachments/assets/0a0bc518-d74d-4b85-907c-19b55d8ebdd4"
    />
    
    <img width="590" height="286" alt="image"
    src="https://github.com/user-attachments/assets/0a2ad257-273d-4174-89aa-4f2ee84bbb3e"
    />
    
    <img width="568" height="292" alt="image"
    src="https://github.com/user-attachments/assets/47ddf263-879d-477d-a13c-fb40a3162eb4"
    />
    
    **Llama3.1 70B**
    For Llama3.1 70B, 12 CPU cores gave the best performance, so we map
    Llama3.1 70B with number of CPU "12"
    
    <img width="687" height="283" alt="image"
    src="https://github.com/user-attachments/assets/f53a17f7-cb25-4fd2-a637-d4975d0b2089"
    />
    
    <img width="585" height="289" alt="image"
    src="https://github.com/user-attachments/assets/6465f684-5877-462b-8303-4dc526069614"
    />
    
    <img width="594" height="285" alt="image"
    src="https://github.com/user-attachments/assets/c752b313-13f6-4f42-a20b-65d2aa54b095"
    />
    
    
    
    
    **Why performance drop when we use more CPUs?**
    
    Here are perfspect results for #CPU=18 and #CPU=24 cases.
    
     **#CPU=18**
    CPU Frequency is around 2300 Hz.
    <img width="1041" height="559" alt="image"
    src="https://github.com/user-attachments/assets/0d681b47-ec30-45be-ae85-69150dcca65d"
    />
    
    Gaudi utilization is around 40%.
    <img width="1029" height="565" alt="image"
    src="https://github.com/user-attachments/assets/ef5c5c2e-28af-46d5-8e1f-3be44178333f"
    />
    
    **#CPU = 24**
    CPU frequency dropped to ~1800 Hz
    <img width="1035" height="549" alt="image"
    src="https://github.com/user-attachments/assets/c834d1f2-15d9-443b-b615-45012fecdecb"
    />
    
    Gaudi utilization dropped to 30%.
    <img width="1022" height="564" alt="image"
    src="https://github.com/user-attachments/assets/ead9220c-7afd-44e9-b3b9-64528c96040d"
    />
    
    
    Therefore, more CPU cores than needed might drop the CPU frequency and
    it also drop the Gaudi utilization due to low performance on CPU.
    
    ---------
    
    Signed-off-by: louie-tsai <louie.tsai@intel.com>
    Signed-off-by: Tsai, Louie <louie.tsai@intel.com>
    Co-authored-by: romir desai <romir.desai@intel.com>

[33mcommit c6eead09280792495915d5f286e387a6e4c68974[m
Author: Monika Helfer <monika.helfer@intel.com>
Date:   Fri Nov 7 16:06:09 2025 +0100

    Documentation: Troubleshooting and FAQ updates and the updated documentation structure (#548)
    
    This PR includes the following updates:
    - Reviewed and updated the Troubleshooting and FAQ documents.
    - Reorganized the documentation structure by replacing the top
    navigation with a left-side navigation bar. This layout is more common
    in technical documentation and makes it easier to browse, view available
    documents, and switch between them.
    - Adjusted the document locations in the navigation bar to better fit
    their categories under the new navigation structure.
    - Added custom styling to make category headers in the sidebar more
    prominent and easier to distinguish from individual documents.
    - Removed unnecessary index pages (e.g., for Configuration, User Guides,
    and Developer Guides), which were previously empty and nor really
    needed.
    - Reorganized doc files into appropriate folders (to match the updated
    website structure) and updated links to these documents.
    
    ---------
    
    Signed-off-by: mhelf-intel <monika.helfer@intel.com>

[33mcommit eee0bbe3cde0031f61586a788e390b895f82e4a0[m
Author: Monika Helfer <monika.helfer@intel.com>
Date:   Fri Nov 7 15:45:57 2025 +0100

    FP8 documentation review (#518)
    
    As discussed with Patryk, I’ve added instructions from GitHub and some
    missing details from the Habana documentation. The FP8 feature is now
    listed under Features and mentions calibration, quantization, and
    inference, with links to the relevant configuration guides. The
    Calibration and Quantization/Inference sections have been moved under
    Configuration.
    
    Please review the updates and let me know if I’ve missed or got anything
    wrong.
    
    ---------
    
    Signed-off-by: mhelf-intel <monika.helfer@intel.com>

[33mcommit 1dba21ea8687bc3fc5a9e54678e24f95c23732fc[m
Author: Monika Helfer <monika.helfer@intel.com>
Date:   Fri Nov 7 14:27:04 2025 +0100

    Doc updates: introduction and developer guides (#529)
    
    Updated the homepage and developer guides.
    
    ---------
    
    Signed-off-by: mhelf-intel <monika.helfer@intel.com>
    Signed-off-by: PatrykWo <patryk.wolsza@intel.com>
    Co-authored-by: PatrykWo <patryk.wolsza@intel.com>

[33mcommit 18d767d6505313e08dbbfdeb76b1aea304c479e3[m
Author: Chendi.Xue <chendi.xue@intel.com>
Date:   Thu Nov 6 22:25:53 2025 -0800

    [GITHUB ACTION][NIXL]update install_nixl.py script (#543)
    
    Follow upstream fix to fix our install_nixl script to work with nixl
    0.7.0
    
    Signed-off-by: Chendi Xue <chendi.xue@intel.com>

[33mcommit 3c3a7a907dfebb32303a2784b1e989a0b50de9f2[m
Author: Wuxun Zhang <wuxun.zhang@intel.com>
Date:   Fri Nov 7 13:18:14 2025 +0800

    Update finished KV transfer state after every step (#532)
    
    In P/D disaggregation scenario, most of time are decoding forward runs
    in decode instances, we need update finished KV transfer states after
    decode forward as well (not only prefill forward). Otherwise, even KV
    transfer is already finished in prefill instance, while decode instance
    cannot get finished state in time (switching state from
    `WAITING_FOR_REMOTE_KVS` to `WAITING`) which will increase TTFT.
    
    Signed-off-by: Wuxun Zhang <wuxun.zhang@intel.com>

[33mcommit 257dadabb7c68129a0f23a43d2a277d89604a1e6[m
Author: Konrad Zawora <kzawora@habana.ai>
Date:   Thu Nov 6 15:12:10 2025 +0100

    Add graph compilation tracking to high level profiler (#50)
    
    Signed-off-by: Konrad Zawora <kzawora@habana.ai>

[33mcommit a2c77c64bb6cd2277804ce05bdb0da553aaea203[m
Author: zhao, zhenhui <zhenzhao@habana.ai>
Date:   Thu Nov 6 19:29:46 2025 +0800

    Port: add VLLM_DISABLE_MARK_SCALES_AS_CONST  (#522)
    
    add VLLM_DISABLE_MARK_SCALES_AS_CONST  to avoid too much graph compile.
    V0 code base
    :https://github.com/HabanaAI/vllm-fork/blob/habana_main/vllm/worker/hpu_model_runner.py#L1228-L1230
    
    Signed-off-by: zhenzhao <zhenzhao@habana.ai>
    Co-authored-by: Michał Kuligowski <michal.kuligowski@intel.com>

[33mcommit f7050a9ea2ba6a2087ea017fd786078ca463db63[m
Author: Agata Dobrzyniewicz <160237065+adobrzyn@users.noreply.github.com>
Date:   Thu Nov 6 09:36:29 2025 +0100

    [Bucketing] Prompt with 0 min and max context blocks (#534)
    
    Signed-off-by: Agata Dobrzyniewicz <adobrzyniewicz@habana.ai>
    Signed-off-by: Agata Dobrzyniewicz <160237065+adobrzyn@users.noreply.github.com>
    Co-authored-by: Copilot <175728472+Copilot@users.noreply.github.com>

[33mcommit f42b34b0c69cdea083feaf52009e7dca3cfd8596[m
Author: Patryk Wolsza <patryk.wolsza@intel.com>
Date:   Wed Nov 5 17:09:50 2025 +0100

    Removing leftovers fork from plugin (#525)
    
    Signed-off-by: PatrykWo <patryk.wolsza@intel.com>

[33mcommit 82085eb443127843959b8b9c9036950702d677ae[m
Author: Konrad Zawora <kzawora@habana.ai>
Date:   Wed Nov 5 15:03:51 2025 +0100

    Fix preemption handling (#524)
    
    This PR fixes multitude of bugs we had in preemption handling:
    - Fixed output token update of `CachedRequestState` - was updated twice
    per iteration, resulting in doubled tokens - this broke preemption when
    request was being re-added to input batch
    - Batch preparation now uses input+output tokens in prefill for
    preempted sequences (both non-unified and unified attention)
    - Preempted sequences now get correctly recognized as prefills after
    they exceed their original prefill length (e.g. prompt was 3 tokens,
    generated 1024 before preemption - the sequence would get treated as
    decode after first 3 tokens)
    - Removed some incorrect assumptions about prefills (can have no
    pre-existing output tokens)
    
    Scenarios with preemptions yield proper accuracy, as can be tested with
    very low `gpu_memory_utilization` and relatively high `max_num_seqs`:
    ```
     PT_HPU_LAZY_MODE=1 VLLM_SKIP_WARMUP=true lm_eval --model vllm --model_args pretrained=/mnt/weka/data/pytorch/llama3.1/Meta-Llama-3.1-8B-Instruct/,enforce_eager=False,dtype=bfloat16,max_num_seqs=128,gpu_memory_utilization=0.05,max_model_len=4096,enable_prefix_caching=True,add_bos_token=false,tensor_parallel_size=1,max_gen_toks=2048 --tasks gsm8k_cot_llama --batch_size auto --trust_remote_code --apply_chat_template --fewshot_as_multiturn --num_fewshot 8
    
    |     Tasks     |Version|     Filter     |n-shot|  Metric   |   |Value |   |Stderr|
    |---------------|------:|----------------|-----:|-----------|---|-----:|---|-----:|
    |gsm8k_cot_llama|      3|flexible-extract|     8|exact_match|↑  |0.8408|±  |0.0101|
    |               |       |strict-match    |     8|exact_match|↑  |0.8415|±  |0.0101|
    ```
    
    ---------
    
    Signed-off-by: Konrad Zawora <kzawora@habana.ai>

[33mcommit f4aeae81b02abff0b8bc0a327a812569374af8f6[m[33m ([m[1;31mskaul/main[m[33m)[m
Author: Monika Helfer <monika.helfer@intel.com>
Date:   Tue Nov 4 14:38:54 2025 +0100

    Documentation updates - part 1 (#493)
    
    This is the first batch of documentation updates. I divided my work into
    several parts to minimize merge conflicts, as multiple people are
    contributing to these files.
    
    The main focus of my changes is to apply good writing practices that
    enhance readability, flow, and overall professionalism. I also split the
    Quick Start guide into three separate documents, as it’s an important
    procedure that should remain easy to follow. It was starting to feel too
    long and complex in its previous form.
    
    Please review these updates and let me know if any corrections are
    needed, particularly regarding the environment variables section. It
    wasn’t clear where VLLM_PROMPT_SEQ_BUCKET_MAX,
    VLLM_HANDLE_TOPK_DUPLICATES, and VLLM_CONFIG_HIDDEN_LAYERS belong, since
    they were only mentioned in a tip. I’ve placed them in an additional
    table for now, but please let me know if that’s not accurate.
    
    ---------
    
    Signed-off-by: mhelf-intel <monika.helfer@intel.com>
    Co-authored-by: Patryk Wolsza <patryk.wolsza@intel.com>

[33mcommit 6d64695fd78132680197cb2a085eef72bd701591[m
Author: Patryk Wolsza <patryk.wolsza@intel.com>
Date:   Tue Nov 4 13:19:04 2025 +0100

    vllm matrix table (#517)
    
    plus fix for PR #275
    
    ---------
    
    Signed-off-by: PatrykWo <patryk.wolsza@intel.com>

[33mcommit be9503d40c7de2eb7082ee49216749af1afa13e8[m
Author: Konrad Zawora <kzawora@habana.ai>
Date:   Tue Nov 4 12:29:31 2025 +0100

    Unified Attention - batch preparation rewrite (#400)
    
    This PR introduces a persistent batch optimization for unified
    attention, as well as full rewrite from PyTorch to NumPy. Effectively,
    it reduces the batch preparation time by 6-15x across the board.
    
    ---------
    
    Signed-off-by: Konrad Zawora <kzawora@habana.ai>
    Co-authored-by: Artur Fierka <artur.fierka@intel.com>

[33mcommit 724f8c1bc35233ba713a06a4a8a08e0bb6bd1b14[m
Author: Michal Adamczyk <michal.adamczyk@intel.com>
Date:   Tue Nov 4 10:41:31 2025 +0100

    Add Unified Attention docs (#275)
    
    Signed-off-by: Michal Adamczyk <michal.adamczyk@intel.com>
    Co-authored-by: Agata Dobrzyniewicz <160237065+adobrzyn@users.noreply.github.com>

[33mcommit c173900787e015140d64322636d98a4c0a482892[m
Author: Patryk Wolsza <patryk.wolsza@intel.com>
Date:   Tue Nov 4 09:31:40 2025 +0100

    Removing data from a deleted column (#514)
    
    Fix for the PR #433 which removed only the header of the
    delayed_sampling column without removing data.
    
    Signed-off-by: PatrykWo <patryk.wolsza@intel.com>

[33mcommit c79e132f639051d3916d9ee563eaafac90a46853[m
Author: Xiaochang Wu <xiaochang.wu@intel.com>
Date:   Mon Nov 3 22:52:29 2025 +0800

    Remove VLLM_DELAYED_SAMPLING  (#433)
    
    VLLM_DELAYED_SAMPLING is for v0 and not used here.
    
    Signed-off-by: Xiaochang Wu <xiaochang.wu@intel.com>

[33mcommit 823bd56d4af6f9a5fd94b5dfa868553b766d29ae[m
Author: Paweł Olejniczak <polejniczakx@habana.ai>
Date:   Mon Nov 3 14:46:26 2025 +0100

    Simplify requirements (#458)
    
    Remove unused and duplicate dependencies, update version constraints
    to unblock security patches while maintaining compatibility.
    
    Changes:
    - Remove: numexpr, numpy, tabulate, setuptools, setuptools-scm
    - Update: ray (<2.49.0 -> >=2.48.0), pandas (-> >=2.2.3), numba (->
    >=0.58.0), numpy (-> >=1.26.0)
    - Keep: transformers >=4.1,<4.56.0 (neural_compressor limitation)
    
    Result: 9 packages reduced to 5, maintains all functionality.
    
    Signed-off-by: Paweł Olejniczak <polejniczakx@habana.ai>

[33mcommit 6a16e9738ea3991ab9f5cd70f1032fa12883be1c[m
Author: Xinyu Chen <xinyu1.chen@intel.com>
Date:   Mon Nov 3 17:00:32 2025 +0800

    DP: allreduce on the host (#498)
    
    let `get_dp_padding` happen on the host since the input and the output
    will only be used on the host and won't block the async scheduling.
    before:
    <img width="737" height="540" alt="Screenshot 2025-10-29 at 14 33 47"
    src="https://github.com/user-attachments/assets/184be344-1e4b-48c8-a646-bb985148ddf2"
    />
    after:
    <img width="884" height="554" alt="Screenshot 2025-10-29 at 14 33 42"
    src="https://github.com/user-attachments/assets/926397ae-2fa6-4078-a527-cf161b19f544"
    />
    
    Signed-off-by: Xinyu Chen <xinyu1.chen@intel.com>
    Co-authored-by: Michał Kuligowski <michal.kuligowski@intel.com>

[33mcommit 55f044008078d077d61ee64b2ca308404922d364[m
Author: Xinyu Chen <xinyu1.chen@intel.com>
Date:   Fri Oct 31 23:22:42 2025 +0800

    MLA: reshape non-contiguous tensor (#505)
    
    verified with non-pd deepseek-r1 correctness with config:
    `--max-num-batched-tokens 131072 --no-enable-prefix-caching`
    
    Signed-off-by: Xinyu Chen <xinyu1.chen@intel.com>
    Co-authored-by: Xiaochang Wu <xiaochang.wu@intel.com>

[33mcommit 90a31997b81b26e9d91b0f1f45ea24ca175c2040[m
Author: Jakub Byczkowski <jakub.byczkowski@intel.com>
Date:   Fri Oct 31 12:49:37 2025 +0100

    Update TESTOWNERS (#494)

[33mcommit 4d8e1b58d9636f979a0626c3b69a55d7541cf562[m
Author: Agata Dobrzyniewicz <160237065+adobrzyn@users.noreply.github.com>
Date:   Fri Oct 31 11:06:59 2025 +0100

    [FIX_FOR_VLLM_LATEST] Hourly fix after: [BugFix] Handle unscheduled requests properly when async scheduling #27756 (#507)
    
    Culprit commit: https://github.com/vllm-project/vllm/pull/27756
    
    ---------
    
    Signed-off-by: Agata Dobrzyniewicz <adobrzyniewicz@habana.ai>
    Signed-off-by: Michał Kuligowski <michal.kuligowski@intel.com>
    Signed-off-by: Agata Dobrzyniewicz <160237065+adobrzyn@users.noreply.github.com>
    Co-authored-by: Michał Kuligowski <michal.kuligowski@intel.com>

[33mcommit 4891c2466f3286c3c0a7455a481809f3f8ebed48[m
Author: Michał Kuligowski <michal.kuligowski@intel.com>
Date:   Fri Oct 31 08:29:24 2025 +0100

    Update troubleshooting.md (#416)
    
    Co-authored-by: Agata Dobrzyniewicz <160237065+adobrzyn@users.noreply.github.com>

[33mcommit e3c6d6a66ed2071512eb1c880dfaad816009f787[m
Author: Jakub Sochacki <97886316+jakub-sochacki@users.noreply.github.com>
Date:   Fri Oct 31 00:31:31 2025 +0100

    Add HABANA_VISIBLE_DEVICES env to Dockerfile.hpu used for PyTorch CI HUD (#506)
    
    Signed-off-by: jakub-sochacki <jakub.sochacki@intel.com>

[33mcommit 173d1cfdd88f519c4b51014dc5c2321dcce333ea[m
Author: Jakub Byczkowski <jakub.byczkowski@intel.com>
Date:   Thu Oct 30 17:06:42 2025 +0100

    [SW-243111] Add correctors for decode buckets (#504)
    
    Decode buckets will not be setup properly if maximum context value is
    missing from context range. Fillters will only remove certain buckets,
    but will not introduce new values in the defined ranges. Instead, for
    decode buckets, use a corrector function that will truncate context
    lenght values to the maximum.
    
    ---------
    
    Co-authored-by: Agata Dobrzyniewicz <160237065+adobrzyn@users.noreply.github.com>

[33mcommit c88562c22cf8ce58a333d11859485661ae1ed480[m
Author: Agata Dobrzyniewicz <160237065+adobrzyn@users.noreply.github.com>
Date:   Thu Oct 30 17:06:04 2025 +0100

    Use query in linear flags - seq as fallback option (#396)
    
    No flag -> default
    Query -> query value
    Seq -> seq value + "will be depricated"warning
    Query & seq -> query value
    
    ---------
    
    Signed-off-by: Agata Dobrzyniewicz <adobrzyniewicz@habana.ai>
    Signed-off-by: Agata Dobrzyniewicz <160237065+adobrzyn@users.noreply.github.com>
    Co-authored-by: Michał Kuligowski <michal.kuligowski@intel.com>

[33mcommit e095e6f8acb4f1578b10b05f2494b7656fae559b[m
Author: Konrad Zawora <kzawora@habana.ai>
Date:   Thu Oct 30 12:20:00 2025 +0100

    Unified Attention - High Level Profiler Integration (#399)
    
    requires the two other big unified attn prs
    
    ---------
    
    Signed-off-by: Agata Dobrzyniewicz <adobrzyniewicz@habana.ai>
    Signed-off-by: Konrad Zawora <kzawora@habana.ai>
    Co-authored-by: Agata Dobrzyniewicz <adobrzyniewicz@habana.ai>
    Co-authored-by: Michał Kuligowski <michal.kuligowski@intel.com>
    Co-authored-by: Agata Dobrzyniewicz <160237065+adobrzyn@users.noreply.github.com>

[33mcommit d15b8211809dc2e507fd1b295d67865fcc9a00bf[m
Author: Konrad Zawora <kzawora@habana.ai>
Date:   Thu Oct 30 09:25:15 2025 +0100

    Add unified attention Granite-8b test (#277)
    
    Co-authored-by: Chendi.Xue <chendi.xue@intel.com>
    Co-authored-by: Michał Kuligowski <michal.kuligowski@intel.com>
    Co-authored-by: Agata Dobrzyniewicz <160237065+adobrzyn@users.noreply.github.com>

[33mcommit 16021e3291b95e9b8e27570baf5d6b2577e5fd1e[m
Author: Jakub Sochacki <97886316+jakub-sochacki@users.noreply.github.com>
Date:   Wed Oct 29 19:49:13 2025 +0100

    HPU Dockerfile for PyTorch CI HUD (#501)
    
    Signed-off-by: jakub-sochacki <jakub.sochacki@intel.com>

[33mcommit 5d7da6f9e548bd74edeffdaa3702811376b386f3[m
Author: Paweł Olejniczak <polejniczakx@habana.ai>
Date:   Wed Oct 29 14:26:14 2025 +0100

    Add docs: Plugin System (#446)
    
    Signed-off-by: Paweł Olejniczak <polejniczakx@habana.ai>
    Signed-off-by: PatrykWo <patryk.wolsza@intel.com>
    Co-authored-by: PatrykWo <patryk.wolsza@intel.com>
    Co-authored-by: Chendi.Xue <chendi.xue@intel.com>
    Co-authored-by: Michał Kuligowski <michal.kuligowski@intel.com>

[33mcommit 47697f6677a6cf069c72b487047797c6619399dd[m
Author: Kamil Kaczor <kamil.kaczor@intel.com>
Date:   Wed Oct 29 12:58:03 2025 +0100

    Fix profiler using wrong bucket (#497)
    
    Signed-off-by: Kamil Kaczor <kamil.kaczor@intel.com>

[33mcommit c54e37001455c98500882a88282514c7861d8c5c[m
Author: Patryk Wolsza <patryk.wolsza@intel.com>
Date:   Wed Oct 29 12:57:37 2025 +0100

    rhel docker fix to main (#489)
    
    Signed-off-by: PatrykWo <patryk.wolsza@intel.com>

[33mcommit b254d5f28c747dcbdf4774788ad3d44e6c5c2457[m
Author: Iryna Boiko <iboiko@habana.ai>
Date:   Wed Oct 29 10:53:20 2025 +0100

    [FIX_FOR_VLLM_LATEST] Fix for Make LayerBlockType a Literal instead of Enum #27658 (#499)
    
    Signed-off-by: Iryna Boiko <iboiko@habana.ai>

[33mcommit ba86c6cc84c3ce844141782364338af09b07536c[m
Author: Artur Fierka <artur.fierka@intel.com>
Date:   Wed Oct 29 10:23:31 2025 +0100

    [Security] Remove unused triton script with null-like value issue (#447) (#492)
    
    Port from releases/v0.11.0 branch
    
    Signed-off-by: Artur Fierka <artur.fierka@intel.com>

[33mcommit 68b3f8823b8f4e428c4e77c7e4a7bb196915d8f1[m
Author: Artur Fierka <artur.fierka@intel.com>
Date:   Wed Oct 29 10:23:16 2025 +0100

    [Security] Fix/remove logically dead code (#448) (#491)
    
    Port from releases/v0.11.0 branch
    
    Signed-off-by: Artur Fierka <artur.fierka@intel.com>

[33mcommit b18734aae7756890f53559b7bb06efdf19d5f17f[m
Author: Artur Fierka <artur.fierka@intel.com>
Date:   Wed Oct 29 10:22:37 2025 +0100

    [Security] Remove structurally dead code (#444) (#490)
    
    Port from releases/v0.11.0 branch
    
    Signed-off-by: Artur Fierka <artur.fierka@intel.com>

[33mcommit bf9e9999476cdebefca7dd48681beaa56ab838a1[m
Author: Yi Liu <yi4.liu@intel.com>
Date:   Wed Oct 29 15:43:28 2025 +0800

    Update the duplicate module list for deepseek r1 (#478)
    
    Signed-off-by: Yi Liu <yiliu4@habana.ai>
    
    ---------
    
    Signed-off-by: Yi Liu <yiliu4@habana.ai>
    Co-authored-by: Yi Liu <yiliu4@habana.ai>

[33mcommit 0924722152b4b81b347141ade187e47c03e31abf[m
Author: Artur Fierka <artur.fierka@intel.com>
Date:   Tue Oct 28 14:00:56 2025 +0100

    Update requirements.txt (#487)
    
    Set ray to latest

[33mcommit 32dd177999d2db22b2e7a5159ec94cbf0a14b646[m
Author: Jacek Czaja <jacek.czaja@intel.com>
Date:   Tue Oct 28 07:40:33 2025 +0100

    Added info if H2d (runtime scale patching) is set (#480)
    
    Added an info about RUNTIME_SCALE_PATCHING when this feature is set .
    exemplary output:
    
    <img width="1782" height="910" alt="image"
    src="https://github.com/user-attachments/assets/02a95685-678a-4659-9fe8-cbc861a96831"
    />
    
    ---------
    
    Signed-off-by: Jacek Czaja <jacek.czaja@intel.com>

[33mcommit 2f287c32b62b7b60257193d9b940e5df465f59dd[m
Author: Iryna Boiko <iboiko@habana.ai>
Date:   Mon Oct 27 20:31:41 2025 +0100

    [FIX_FOR_VLLM_LATEST] Fix for Clean up utils #27552 (#481)
    
    Signed-off-by: Iryna Boiko <iboiko@habana.ai>

[33mcommit c834e0a24b17cb604b4f121215c1dcc173ec056d[m
Author: Uri Livne <ulivne@habana.ai>
Date:   Mon Oct 27 17:20:26 2025 +0200

    Add granite calibration test to all tests function (#453)
    
    Signed-off-by: Uri Livne <ulivne@habana.ai>

[33mcommit 2613b14b0ca3491427bdb3f436e72b000eefa688[m
Author: Wuxun Zhang <wuxun.zhang@intel.com>
Date:   Mon Oct 27 21:42:26 2025 +0800

    fix dummy run config for P/D prefiller instance (#467)
    
    When P/D disaggregation + DP used, prefiller instance should run dummy
    prefill run instead of decode run when certain DP rank has no work to
    do.
    
    Signed-off-by: Wuxun Zhang <wuxun.zhang@intel.com>

[33mcommit 5cae03ba0f83ed8d2e361a66863197fcd3b9fd92[m
Author: Patryk Wolsza <patryk.wolsza@intel.com>
Date:   Mon Oct 27 14:39:20 2025 +0100

    New docs part3 updates (#456)
    
    Part 3 of changes and updates to mkdocs documentation.
    
    ---------
    
    Signed-off-by: PatrykWo <patryk.wolsza@intel.com>

[33mcommit e98e0fc4fbc2f497f087e3d03207a61dc25825ec[m
Author: Kamil Kaczor <kamil.kaczor@intel.com>
Date:   Mon Oct 27 13:22:04 2025 +0100

    Fix prompt/decode profiler (#472)
    
    Signed-off-by: Kamil Kaczor <kamil.kaczor@intel.com>

[33mcommit 611f4155ec3e79d4682d58683a841ec88d56522d[m
Author: Jimin Ha <jimin.ha@intel.com>
Date:   Fri Oct 24 18:30:43 2025 -0700

    Gemma3 Multimodal optimization (#404)
    
    This is to optimize gemma3 multimodal memory/performance.
    - bucket vision tower based on batch bucket to reduce recompile overhead
    - modify merge_multimodal to use torch.where instead of masked_scatter
    for performance issue
     - add warmup multimodal bucket to precompile vision tower
    - port PT_HPU_SDPA_QKV_SLICE_MODE_FWD feature from vllm-fork v0 : this
    is necessary to reduce the memory for the longer sequence length.
    - add 01,02 prefix for the general plugin so the order of initialization
    of plugins is guaranteed(ops followed by model)
    
    ---------
    
    Signed-off-by: Jimin Ha <jimin.ha@intel.com>
    Signed-off-by: Mohit Deopujari <mdeopujari@habana.ai>
    Co-authored-by: Mohit Deopujari <mdeopujari@habana.ai>

[33mcommit 6666216b2d2c5df7347ad05206a84e09a7f00da6[m
Author: Iryna Boiko <iboiko@habana.ai>
Date:   Sat Oct 25 02:06:08 2025 +0200

    Applying of [V1][spec decode] return logprobs for spec decoding #26060 (#476)
    
    Signed-off-by: Iryna Boiko <iboiko@habana.ai>

[33mcommit dbf9e48d7cc839679d740cfd29c50d7021d10b3f[m
Author: Wuxun Zhang <wuxun.zhang@intel.com>
Date:   Fri Oct 24 22:13:25 2025 +0800

    Update KVConnectorOutpout for P/D when async scheduling turned on (#468)
    
    When async scheduling enabled, need also update `KVConnectorOutput`
    inside `ModelRunnerOutput`, otherwise in next schedule step,
    `_update_waiting_for_remote_kv` cannot check which req is finsihed
    transferring and put into `running` queue.
    
    
    https://github.com/vllm-project/vllm/blob/85fee74b337522f7e0807fc100b9e00682ff45e1/vllm/v1/core/sched/scheduler.py#L1342-L1344
    
    Signed-off-by: Wuxun Zhang <wuxun.zhang@intel.com>

[33mcommit 692fdaa4ba9d4ee43a0a10c300ea49401b2cfd60[m
Author: Iryna Boiko <iboiko@habana.ai>
Date:   Fri Oct 24 15:37:52 2025 +0200

    [FIX_FOR_VLLM_LATEST] Fix for is_pin_memory_available import and skip of run_spec_decode_ngram_test due to #26060 (#471)
    
    Signed-off-by: Iryna Boiko <iboiko@habana.ai>

[33mcommit 7981a1de36f6bdc4a6832839288d22e6417f1d09[m
Author: Konrad Zawora <kzawora@habana.ai>
Date:   Fri Oct 24 14:01:51 2025 +0200

    Fix defragmentation for MLA-based models (#470)
    
    MLA-based models don't have key-value pairs in cache and utilize a
    single latent cache, so trying to swap anything in value cache will
    fail, as the cache is None. GSM8k seems to work fine now with contiguous
    PA + defrag + APC disabled on deepseek-ai/DeepSeek-V2-Lite-Chat:
    ```
    VLLM_SKIP_WARMUP=true VLLM_CONTIGUOUS_PA=true VLLM_DEFRAG=true PT_HPU_LAZY_MODE=1 lm_eval --model vllm --model_args pretrained=deepseek-ai/DeepSeek-V2-Lite-Chat,,enforce_eager=False,dtype=bfloat16,max_num_seqs=128,gpu_memory_utilization=0.8,max_model_len=4096,enable_prefix_caching=False,add_bos_token=false,tensor_parallel_size=1,max_gen_toks=2048 --tasks gsm8k_cot_llama --batch_size auto --trust_remote_code --apply_chat_template --fewshot_as_multiturn --num_fewshot 8
    
    |     Tasks     |Version|     Filter     |n-shot|  Metric   |   |Value |   |Stderr|
    |---------------|------:|----------------|-----:|-----------|---|-----:|---|-----:|
    |gsm8k_cot_llama|      3|flexible-extract|     8|exact_match|↑  |0.7074|±  |0.0125|
    |               |       |strict-match    |     8|exact_match|↑  |0.6952|±  |0.0127|
    ```

[33mcommit c43ec9c495c7fedc65353644c705522b77fd6632[m
Author: Jakub Sochacki <97886316+jakub-sochacki@users.noreply.github.com>
Date:   Thu Oct 23 12:52:35 2025 +0200

    Fix requirements filtering in HPU Dockerfiles (#419)
    
    This line of code incorrectly filtered requirements to be installed:
    `sed '/^[torch]/d' requirements/build.txt`
    - filters out all packages which names start with t / o / r / c / h
    
    So if requirements were:
    `
    cmake>=3.26.1
    ninja
    packaging>=24.2
    setuptools>=77.0.3,<80.0.0
    setuptools-scm>=8
    torch==2.8.0
    wheel
    jinja2>=3.1.6
    regex
    build
    `
    we were skipping cmake, torch and regex.
    
    `sed '/^torch/d' requirements/build.txt`
    this would skip only all torch packages.
    
    Signed-off-by: jakub-sochacki <jakub.sochacki@intel.com>

[33mcommit 5dfcb10ef610513f2848354905fbe857e2008272[m
Author: Soila Kavulya <soila.p.kavulya@intel.com>
Date:   Wed Oct 22 23:25:41 2025 -0700

    Fix math log2 exponential bucket error if max_model_len <= block_size (#451)
    
    INC calibration fails with math domain error when math.log2 op has
    undefined result during INC calibration. The INC calibration scripts
    uses max_model_len=128 which results in max_ctx=0. log2 of zero is
    undefined.
    
    `PT_HPU_LAZY_MODE=1 ./calibrate_model.sh -m meta-llama/Llama-3.1-70B -d
    NeelNanda/pile-10k -o ./inc2 -b 1 -t 2 -l 5`
    
    
    ESC[1;36m(Worker_TP1 pid=65456)ESC[0;0m INFO 10-22 22:53:05
    [hpu_worker.py:242] Initializing cache engine took 74.67 GiB of device
    memory (109 GiB/126.5 GiB used) and -1.967 GiB of host memory (93.07
    GiB/1007 GiB used)
    ESC[1;36m(Worker_TP1 pid=65456)ESC[0;0m ERROR 10-22 22:53:05
    [multiproc_executor.py:703] WorkerProc hit an exception.
    ESC[1;36m(Worker_TP1 pid=65456)ESC[0;0m ERROR 10-22 22:53:05
    [multiproc_executor.py:703] Traceback (most recent call last):
    ESC[1;36m(Worker_TP1 pid=65456)ESC[0;0m ERROR 10-22 22:53:05
    [multiproc_executor.py:703] File
    "/tmp/vllm/vllm/v1/executor/multiproc_executor.py", line 698, in
    worker_busy_loop
    ESC[1;36m(Worker_TP1 pid=65456)ESC[0;0m ERROR 10-22 22:53:05
    [multiproc_executor.py:703] output = func(*args, **kwargs)
    ESC[1;36m(Worker_TP1 pid=65456)ESC[0;0m ERROR 10-22 22:53:05
    [multiproc_executor.py:703] File
    "/tmp/vllm/vllm/v1/worker/worker_base.py", line 305, in
    initialize_from_config
    ESC[1;36m(Worker_TP1 pid=65456)ESC[0;0m ERROR 10-22 22:53:05
    [multiproc_executor.py:703]
    self.worker.initialize_from_config(kv_cache_config) # type: ignore
    ESC[1;36m(Worker_TP1 pid=65456)ESC[0;0m ERROR 10-22 22:53:05
    [multiproc_executor.py:703] File
    "/tmp/vllm-gaudi/vllm_gaudi/v1/worker/hpu_worker.py", line 243, in
    initialize_from_config
    ESC[1;36m(Worker_TP1 pid=65456)ESC[0;0m ERROR 10-22 22:53:05
    [multiproc_executor.py:703] self.compile_or_warm_up_model()
    ESC[1;36m(Worker_TP1 pid=65456)ESC[0;0m ERROR 10-22 22:53:05
    [multiproc_executor.py:703] File
    "/tmp/vllm-gaudi/vllm_gaudi/v1/worker/hpu_worker.py", line 249, in
    compile_or_warm_up_model
    ESC[1;36m(Worker_TP1 pid=65456)ESC[0;0m ERROR 10-22 22:53:05
    [multiproc_executor.py:703] self.model_runner.warmup_model()
    ESC[1;36m(Worker_TP1 pid=65456)ESC[0;0m ERROR 10-22 22:53:05
    [multiproc_executor.py:703] File
    "/usr/local/lib/python3.10/dist-packages/torch/utils/_contextlib.py",
    line 116, in decorate_context
    ESC[1;36m(Worker_TP1 pid=65456)ESC[0;0m ERROR 10-22 22:53:05
    [multiproc_executor.py:703] return func(*args, **kwargs)
    ESC[1;36m(Worker_TP1 pid=65456)ESC[0;0m ERROR 10-22 22:53:05
    [multiproc_executor.py:703] File
    "/tmp/vllm-gaudi/vllm_gaudi/v1/worker/hpu_model_runner.py", line 4028,
    in warmup_model
    ESC[1;36m(Worker_TP1 pid=65456)ESC[0;0m ERROR 10-22 22:53:05
    [multiproc_executor.py:703]
    self.bucketing_manager.generate_prompt_buckets()
    ESC[1;36m(Worker_TP1 pid=65456)ESC[0;0m ERROR 10-22 22:53:05
    [multiproc_executor.py:703] File
    "/tmp/vllm-gaudi/vllm_gaudi/extension/bucketing/common.py", line 110, in
    generate_prompt_buckets
    ESC[1;36m(Worker_TP1 pid=65456)ESC[0;0m ERROR 10-22 22:53:05
    [multiproc_executor.py:703] bs_cfg, query_cfg, ctx_cfg =
    strategy.get_prompt_cfgs(max_num_prefill_seqs=self.max_num_prefill_seqs,
    ESC[1;36m(Worker_TP1 pid=65456)ESC[0;0m ERROR 10-22 22:53:05
    [multiproc_executor.py:703] File
    "/tmp/vllm-gaudi/vllm_gaudi/extension/bucketing/exponential.py", line
    42, in get_prompt_cfgs
    ESC[1;36m(Worker_TP1 pid=65456)ESC[0;0m ERROR 10-22 22:53:05
    [multiproc_executor.py:703] max_prompt_ctx_limit = 2 if max_ctx == 1
    else math.ceil(math.log2(max_ctx)) + 1
    ESC[1;36m(Worker_TP1 pid=65456)ESC[0;0m ERROR 10-22 22:53:05
    [multiproc_executor.py:703] ValueError: math domain error
    
    Signed-off-by: Kavulya, Soila P <soila.p.kavulya@intel.com>

[33mcommit e2060291c3c952ba0b70c0ee648b13d207e35f84[m
Author: Agata Dobrzyniewicz <160237065+adobrzyn@users.noreply.github.com>
Date:   Thu Oct 23 08:21:03 2025 +0200

    Buckets from file - alpha version (#375)
    
    Add bucketing from file - experimental use for now!
    
    - [X] Prompt / decode read from file
    - [X] Example file with example buckets
    - Unified attention from file -> to be done in differen PR
    - [X] Filter recieved buckets?
    - [X] Ranges
    - [ ] README
    
    ---------
    
    Signed-off-by: Agata Dobrzyniewicz <adobrzyniewicz@habana.ai>
    Co-authored-by: Michał Kuligowski <michal.kuligowski@intel.com>

[33mcommit 3d6721bfe39e2453e9a6a531a746794ec25c606b[m
Author: Agata Dobrzyniewicz <160237065+adobrzyn@users.noreply.github.com>
Date:   Thu Oct 23 08:18:27 2025 +0200

    [Linear warmup] Default values optimization (#426)
    
    Signed-off-by: Agata Dobrzyniewicz <adobrzyniewicz@habana.ai>

[33mcommit 54fd7c2c524dcfbbd993d767573df50774675861[m
Author: Wuxun Zhang <wuxun.zhang@intel.com>
Date:   Thu Oct 23 06:20:29 2025 +0800

    Support DP for unified attention (#242)
    
    ~Depends on https://github.com/vllm-project/vllm-gaudi/pull/226.~
    See last commit added for this PR.
    
    ---------
    
    Signed-off-by: Wuxun Zhang <wuxun.zhang@intel.com>
    Co-authored-by: Chendi.Xue <chendi.xue@intel.com>
    Co-authored-by: Agata Dobrzyniewicz <160237065+adobrzyn@users.noreply.github.com>

[33mcommit 0e1adc074e9c2e218cbec91161eaa06dc87ac658[m
Author: Wuxun Zhang <wuxun.zhang@intel.com>
Date:   Thu Oct 23 04:59:58 2025 +0800

    reuse DP allgather tensor across layers (#415)
    
    This is to reduce cached memory size for DP dispatch/combine when HPU
    graph enabled.
    
    ---------
    
    Signed-off-by: Wuxun Zhang <wuxun.zhang@intel.com>
    Co-authored-by: Chendi.Xue <chendi.xue@intel.com>

[33mcommit 6906e234abff9076cbfa10af0f5132e5df2b40a1[m
Author: Chendi.Xue <chendi.xue@intel.com>
Date:   Wed Oct 22 15:33:11 2025 -0500

    [GITHUB ACTION] Always run same job to same node (#450)
    
    Signed-off-by: Chendi Xue <chendi.xue@intel.com>

[33mcommit eb979d7ff4039517ff152e698730838a40660e82[m
Author: Artur Fierka <artur.fierka@intel.com>
Date:   Wed Oct 22 21:09:56 2025 +0200

    [Security] Update requirements.txt (#443) (#445)
    
    Set numpy to latest
    
    Cherry-pick from releases/v0.11.0:
    https://github.com/vllm-project/vllm-gaudi/pull/443
    
    Signed-off-by: Artur Fierka <artur.fierka@intel.com>

[33mcommit 8fffc18f4e23ee7523b7e134827e12e84a0d6044[m
Author: Paweł Olejniczak <polejniczakx@habana.ai>
Date:   Wed Oct 22 18:20:54 2025 +0200

    Update docs: Quickstart - Executing inference (#410)
    
    Completed _Executing inference_ section of _Quickstart_ doc.
    
    Signed-off-by: Paweł Olejniczak <polejniczakx@habana.ai>
    Co-authored-by: Patryk Wolsza <patryk.wolsza@intel.com>

[33mcommit 3d134d394afa8adae7bf2fbfa7cf064d49b856f3[m
Author: Konrad Zawora <kzawora@habana.ai>
Date:   Wed Oct 22 14:54:34 2025 +0200

    [main] Defragmenter warmup accuracy workaround (#436)
    
    the title of this pr sounds like random set of words put together to
    sound wise (or something from passphrase generator like "rose listen
    donkey wild function") but i swear it is not
    
    Signed-off-by: Konrad Zawora <kzawora@habana.ai>
    Co-authored-by: Patryk Wolsza <patryk.wolsza@intel.com>

[33mcommit 98238588331839e1f8ecc1381c01470575be7ed7[m
Author: Iryna Boiko <iboiko@habana.ai>
Date:   Wed Oct 22 12:37:28 2025 +0200

    [FIX_FOR_VLLM_LATEST] Fix for #26440 (#442)
    
    Fix for "[Performance] Dual stream execution of "shared_experts" and
    "selected_experts" inside FusedMoE #26440"
    Similar to [Bugfix][CPU] Disable dual stream execution for experts on
    CPU #27320
    
    Signed-off-by: Iryna Boiko <iboiko@habana.ai>

[33mcommit 86162c4445e6ff7f13544d0f119df9aaa2335785[m
Author: Patryk Wolsza <patryk.wolsza@intel.com>
Date:   Tue Oct 21 11:14:43 2025 +0200

    Docs update post v0.11 (#428)
    
    General formatting, fixes, some updates to docs
    
    ---------
    
    Signed-off-by: PatrykWo <patryk.wolsza@intel.com>

[33mcommit 05580e0676a8657d589a9ea6460dbaab0aedcdf9[m
Author: Soila Kavulya <soila.p.kavulya@intel.com>
Date:   Tue Oct 21 02:09:53 2025 -0700

    [SW-242466] Update not_over_max_model_len filter to fix warmup perf regression (#424)
    
    Fix performance regression caused by missing warmup buckets associated
    with https://github.com/vllm-project/vllm-gaudi/pull/331
    
    Signed-off-by: Kavulya, Soila P <soila.p.kavulya@intel.com>

[33mcommit 3b629a82146ddd06263b093b047ee433d0015a9a[m
Author: Yaser Afshar <yaser.afshar@intel.com>
Date:   Mon Oct 20 13:05:42 2025 -0700

    Fix typo in installation.md: correct script name to install_nixl.py (#385)
    
    ### Summary
    
    This PR fixes a minor typo in the `installation.md` file where the
    installation command incorrectly referenced `install_nixl.sh`. The
    correct script name is `install_nixl.py`.
    
    ### Changes Made
    
    - Updated `python install_nixl.sh` to `python install_nixl.py` in
    installation.md.
    
    ### Why This Is Needed
    
    The incorrect script name could lead to confusion or installation errors
    for users following the documentation. This change ensures clarity and
    accuracy in the setup instructions.
    
    Co-authored-by: Michał Kuligowski <michal.kuligowski@intel.com>

[33mcommit 9992a7ebad28ab17c2479dfc7f06b1acf73c83d0[m
Author: Chendi.Xue <chendi.xue@intel.com>
Date:   Mon Oct 20 14:51:27 2025 -0500

    [NIXL]Enable prefill TP < Decode TP with host_buffer (#421)
    
    SW-242362
    
    Change DecodeTP=2, PrefillTP=1 for test
    
    Issue:
    - HPU attn is using NHD, and vllm upstream only support prefill / decode
    heterogeneous TP with HND.
    
    Solution:
     - init:
        hpu_attn with NHD -> host_buffer with HND
    
     - copy device to host:
        permute kv for req -> copy to host buffer
    
     - nixl_connector transfer host KV with HND + TP_ratio support
    
     - copy host to device
        permute kv for req -> copy to device
    
    =====
    
    Validated, accuracy is good
    
    ---
    FYI, no need change for MLA (Deepseek)
    
    ---------
    
    Signed-off-by: Chendi Xue <chendi.xue@intel.com>

[33mcommit d8f142ac331e3dd27fab4d84e286268b238246e0[m
Author: Iryna Boiko <iboiko@habana.ai>
Date:   Mon Oct 20 18:39:12 2025 +0200

    [FIX_FOR_VLLM_LATEST] Fixes for upstream #26908 and #27143 and #27169 (#427)
    
    Signed-off-by: Iryna Boiko <iboiko@habana.ai>

[33mcommit 7e2d9c9cdc7efb180cbf9d5c097d46018bc83fa8[m
Author: Michal Gawarkiewicz <michal.gawarkiewicz@intel.com>
Date:   Mon Oct 20 15:39:48 2025 +0200

    Update supported_features.md (#180)
    
    Signed-off-by: PatrykWo <patryk.wolsza@intel.com>
    Co-authored-by: Agata Dobrzyniewicz <160237065+adobrzyn@users.noreply.github.com>
    Co-authored-by: Patryk Wolsza <patryk.wolsza@intel.com>

[33mcommit 87b8456fdebb68b16c1ef5e5ecfb7216299b51ef[m
Author: Chendi.Xue <chendi.xue@intel.com>
Date:   Sat Oct 18 01:25:44 2025 -0500

    [CI]unified attn is too easy to fail, add small RTOL (#422)
    
    Signed-off-by: Chendi Xue <chendi.xue@intel.com>

[33mcommit b8ded81efeb170c6c232d6058a9ecfdf6628b50f[m
Author: Agata Dobrzyniewicz <160237065+adobrzyn@users.noreply.github.com>
Date:   Sat Oct 18 04:20:56 2025 +0200

    [FIX_FOR_VLLM_LATEST]  Fix for #27022 (#418)
    
    Culprit commit: https://github.com/vllm-project/vllm/pull/27022
    
    ---------
    
    Signed-off-by: Agata Dobrzyniewicz <adobrzyniewicz@habana.ai>

[33mcommit 7d24df2958a7f00ce690a0f3bdcf6abadaa4a9a5[m
Author: Artur Fierka <artur.fierka@intel.com>
Date:   Fri Oct 17 08:12:13 2025 +0200

    Add fp8 calibration procedure (#309)
    
    Porting the FP8 calibration procedure from vllm-hpu-extension:
    https://github.com/HabanaAI/vllm-hpu-extension/tree/main/calibration
    
    ---------
    
    Signed-off-by: Artur Fierka <artur.fierka@intel.com>

[33mcommit ef6b8b357b7932e2a8bac14828d5ebb1ccf2a6e1[m
Author: Iryna Boiko <iboiko@habana.ai>
Date:   Thu Oct 16 17:01:54 2025 +0200

    [FIX_FOR_VLLM_LATEST] Fix for Separate out vllm.utils.collections #26990 (#413)
    
    Signed-off-by: Iryna Boiko <iboiko@habana.ai>
    Co-authored-by: Artur Fierka <artur.fierka@intel.com>

[33mcommit ae1eaacfca5d2d42b2d27b002590bc683897bac1[m
Author: Mohit Deopujari <mdeopujari@habana.ai>
Date:   Thu Oct 16 04:52:15 2025 -0700

    Multi-image generation CI tests (#377)
    
    ### Test with multimodal-support for multiple images
    
    - Current CI test for gemma3 only runs single image per prompt and its
    input seq_len is less than current sliding_window length (1024).
    - This new test is designed such that the total input length exceeds the
    default sliding_window length (1024) to help validate the sliding_window
    mechanism is actually working or not.
    
    ---------
    
    Signed-off-by: Mohit Deopujari <mdeopujari@habana.ai>
    Co-authored-by: Michał Kuligowski <michal.kuligowski@intel.com>

[33mcommit 3626290fa5b61cd5a72f65811b6d196dffad242e[m
Author: Chendi.Xue <chendi.xue@intel.com>
Date:   Thu Oct 16 03:52:56 2025 -0500

    [NIXL][BUGFIX][Gaudi2Gaudi accuracy] use 4d kv_cache for nixl_connector KV register and update host_buffer accordingly (#411)
    
    Implement for SW-242433
    
    ** Root cause for accuracy issue **
    
    nixl_connector register_kv_cache assume any input kv_caches are with 4D
    or 5D.
    `[2, num_blocks, block_size, num_kv_heads, head_size]` or `[num_blocks,
    block_size, num_kv_heads, head_size]`
    
    However, HPU KV cache is with 3D tuple: `Tuple([num_blocks*block_size,
    num_kv_heads, head_size], ...)`
    
    => Different KV layout leads to incorrect num_blocks and data_ptr
    calculation in nixl_connector => leads to wrong data copied.
    
    ** Solution **
    
    1. create a new KV_caches_4D dict for nixl_conenctor. This 4D KV_caches
    is a reference to original KV_cache with 4D view. (Same memory address)
    2. Fix `habana_frameworks.torch.utils.experimental._data_ptr`
    incapability of fetching address on view-tensor by using global map to
    map virtual to physical addr.
    3. add a new TupleTensor Class which treat tuple as Tensor to return
    shape, device, dtype
    
    
    ** validation **
    
    Tested with both Gaudi2Gaudi and Gaudi2CPU2Gaudi on "Qwen/Qwen3-0.6B",
    "deepseek-ai/DeepSeek-V2-Lite-Chat"
    
    All 4 cases get expected accuracy.
    
    ---------
    
    Signed-off-by: Chendi Xue <Chendi.Xue@intel.com>
    Signed-off-by: Chendi Xue <chendi.xue@intel.com>

[33mcommit a7b2a4166cc1f9d9532f643047230496fd1c8ab3[m
Author: Agata Dobrzyniewicz <160237065+adobrzyn@users.noreply.github.com>
Date:   Thu Oct 16 07:25:10 2025 +0200

    Unified attention improvemets (#363)
    
    - [x] warmup funcioning
    - [x] no recompiles
    
    ---------
    
    Signed-off-by: Agata Dobrzyniewicz <adobrzyniewicz@habana.ai>
    Co-authored-by: Michał Kuligowski <michal.kuligowski@intel.com>
    Co-authored-by: Konrad Zawora <kzawora@habana.ai>

[33mcommit b0bd04b944411cd01de21b229748324c38e87c45[m
Author: Iryna Boiko <iboiko@habana.ai>
Date:   Thu Oct 16 07:24:49 2025 +0200

    Add missing prompt bucket to warmup, when max_ctx is 0 (#352)
    
    Signed-off-by: Iryna Boiko <iboiko@habana.ai>
    Co-authored-by: Michał Kuligowski <michal.kuligowski@intel.com>

[33mcommit fb59059d25d3499162cae5963a72e710be7e5ad3[m
Author: Izabela Irzynska <izabela.irzynska@intel.com>
Date:   Thu Oct 16 07:24:09 2025 +0200

    Unit test for prefix caching in Gaudi plugin (#349)
    
    Unit test for Automatic Prefix Caching for plugin (ticket:
    https://jira.habana-labs.com/browse/SW-236107) .
    
    ---------
    
    Signed-off-by: Izabela Irzynska <iirzynska@habana.ai>

[33mcommit f97f2b54234f04849109e63f53886ef876c8b720[m
Author: Neelesh Gokhale <neelesh.gokhale@intel.com>
Date:   Thu Oct 16 10:50:05 2025 +0530

    Cherrypick cd docker fixes/commits from v0.10.2 to main v0.11.0 (#341)
    
    Signed-off-by: Neelesh Gokhale <neelesh.gokhale@intel.com>
    Signed-off-by: PatrykWo <patryk.wolsza@intel.com>
    Co-authored-by: Agata Dobrzyniewicz <160237065+adobrzyn@users.noreply.github.com>
    Co-authored-by: PatrykWo <patryk.wolsza@intel.com>
    Co-authored-by: Michał Kuligowski <michal.kuligowski@intel.com>

[33mcommit 5eaed66471acbf29b8615dbea4aa36bc8d2482d9[m
Author: Iryna Boiko <iboiko@habana.ai>
Date:   Wed Oct 15 17:28:47 2025 +0200

    [FIX_FOR_VLLM_LATEST] Upstream vllm fixes for #26355 and #26737 (#407)
    
    Fix for name change compilation level to compilation mode, deprecation
    compilation level (#26355)
    
    ---------
    
    Signed-off-by: Iryna Boiko <iboiko@habana.ai>
    Co-authored-by: Michał Kuligowski <michal.kuligowski@intel.com>

[33mcommit b6768b9175ef700c2ed675d5c417609215e04682[m
Author: Agata Dobrzyniewicz <160237065+adobrzyn@users.noreply.github.com>
Date:   Wed Oct 15 11:43:11 2025 +0200

    Enviroment logs - disable prefix caching with conti pa + add vllm brnach+commit value to logs (#402)
    
    Signed-off-by: Agata Dobrzyniewicz <adobrzyniewicz@habana.ai>

[33mcommit 5716c5d1d36d9025306d82e43f607a2320373262[m
Author: Kamil Kaczor <kamil.kaczor@intel.com>
Date:   Wed Oct 15 10:28:58 2025 +0200

    Fix linear assert (#401)

[33mcommit f755642e59457b3a786ebeb6bdc241c4da73378f[m
Author: Michał Kuligowski <michal.kuligowski@intel.com>
Date:   Wed Oct 15 10:18:15 2025 +0200

    Minor optimizationm for bucketing calc (#395)

[33mcommit 09e4a685475ef5e4e957c1365c17e8bbc765bd94[m
Author: Konrad Zawora <kzawora@habana.ai>
Date:   Wed Oct 15 10:15:38 2025 +0200

    Unified Attention Accuracy Bugfixes (#393)
    
    I've noticed two accuracy issues in unified attention:
    1. We weren't updating the persistent request states and batch in the
    `unified_execute_model` method.
    2. We were overextending non-aligned prefix_prefill context lengths by
    one token .
    
    The first one had major impact - I suspect we were malforming batches as
    the generation process went on, since the self.input_batch.num_tokens &
    req_state.output_token_ids were not updated correctly - in Granite GSM8K
    fixing that yielded +10 percentage points improvement
    The second one had a negligible impact - I didn't notice any acc
    improvement in any tests I've run - but we should be masking anything
    above context length regardless.
    
    I've added GSM8k accuracy test to CI with this PR that should now pass
    as well.
    
    ---------
    
    Signed-off-by: Konrad Zawora <kzawora@habana.ai>
    Co-authored-by: Michał Kuligowski <michal.kuligowski@intel.com>

[33mcommit dbb941c77e1c55d8981ce962b87933366eb80498[m
Author: Xinyu Chen <xinyu1.chen@intel.com>
Date:   Wed Oct 15 03:30:19 2025 +0800

    nixl: support mla kvcache transfer (#403)
    
    Signed-off-by: Xinyu Chen <xinyu1.chen@intel.com>

[33mcommit 8c08770863245217a9dffdfca00cf4c7db25e814[m
Author: Tianmu Li <tianmu.li@intel.com>
Date:   Tue Oct 14 04:58:19 2025 -0700

    Fix issue with async_scheduling when dealing with chunked input (#360)
    
    Cherry-pick of #359
    
    ---------
    
    Signed-off-by: Tianmu Li <tianmu.li@intel.com>
    Co-authored-by: Michał Kuligowski <michal.kuligowski@intel.com>

[33mcommit cdf63c661495baeb67092d7004e959ec39d3ac97[m
Author: Agata Dobrzyniewicz <160237065+adobrzyn@users.noreply.github.com>
Date:   Tue Oct 14 10:28:46 2025 +0200

    Port: [Docs] CI failures chapter (#276) (#389)
    
    Signed-off-by: Agata Dobrzyniewicz <adobrzyniewicz@habana.ai>
    Co-authored-by: Michał Kuligowski <michal.kuligowski@intel.com>

[33mcommit 484feb6ba626436333a1b7a2b64210848b2efa49[m
Author: Agata Dobrzyniewicz <160237065+adobrzyn@users.noreply.github.com>
Date:   Tue Oct 14 09:55:43 2025 +0200

    Experimental - fatal errro from 0.12 release (#398)
    
    Signed-off-by: Agata Dobrzyniewicz <adobrzyniewicz@habana.ai>
    Co-authored-by: Michał Kuligowski <michal.kuligowski@intel.com>

[33mcommit 987db220f1a91b8974a0f999f7051e70825b9524[m
Author: Soila Kavulya <soila.p.kavulya@intel.com>
Date:   Tue Oct 14 00:44:23 2025 -0700

    [SW-241908] Omit all prompt buckets that exceed max_num_batched_tokens (#331)
    
    [SW-241908] Fixes regression in tests due to invalid buckets generated
    if VLLM_PROMPT_BS_BUCKET_MAX is set, and the number of tokens in prefill
    batch exceeds the max_num_batched_tokens. The regression is associated
    with https://github.com/vllm-project/vllm-gaudi/pull/224
    
    This fix checks that the number of tokens in both the current and next
    token bucket does not exceed max_num_batched tokens, and resolves
    "ValueError: operands could not be broadcast together with shape"
    exception
    
    `VLLM_CONTIGUOUS_PA=false VLLM_DECODE_BLOCK_BUCKET_MAX=512 VLLM_USE_V1=1
    VLLM_PROMPT_BS_BUCKET_MAX=16 vllm serve meta-llama/Llama-3.1-8B-Instruct
    --dtype bfloat16 --tensor-parallel-size 1 --swap-space 16`
    
    (EngineCore_DP0 pid=352574) File
    "/vllm-gaudi/vllm_gaudi/v1/worker/hpu_model_runner.py", line 4051, in
    warmup_model
    (EngineCore_DP0 pid=352574)     self.warmup_graphs(
    (EngineCore_DP0 pid=352574) File
    "/vllm-gaudi/vllm_gaudi/v1/worker/hpu_model_runner.py", line 3659, in
    warmup_graphs
    (EngineCore_DP0 pid=352574) self._prepare_dummy_scenario(prompt_cfg,
    decode_cfg)
    (EngineCore_DP0 pid=352574) File
    "/vllm-gaudi/vllm_gaudi/v1/worker/hpu_model_runner.py", line 3897, in
    _prepare_dummy_scenario
    (EngineCore_DP0 pid=352574) self._execute_dummy_scenario(requests,
    scheduled_tokens)
    (EngineCore_DP0 pid=352574) File
    "/vllm-gaudi/vllm_gaudi/v1/worker/hpu_model_runner.py", line 3928, in
    _execute_dummy_scenario
    (EngineCore_DP0 pid=352574) self.execute_model(sched_output,
    warmup_mode=True)
    (EngineCore_DP0 pid=352574) File
    "/usr/local/lib/python3.10/dist-packages/torch/utils/_contextlib.py",
    line 120, in decorate_context
    (EngineCore_DP0 pid=352574)     return func(*args, **kwargs)
    (EngineCore_DP0 pid=352574) File
    "/vllm-gaudi/vllm_gaudi/v1/worker/hpu_model_runner.py", line 2914, in
    execute_model
    (EngineCore_DP0 pid=352574) prefill_input_data, decode_input_data =
    self._prepare_inputs(scheduler_output, num_prefills, num_decodes,
    (EngineCore_DP0 pid=352574) File
    "/vllm-gaudi/vllm_gaudi/v1/worker/hpu_model_runner.py", line 2304, in
    _prepare_inputs
    (EngineCore_DP0 pid=352574)
    np.add(self.input_batch.num_computed_tokens_cpu[req_indices], arange,
    out=positions_np)
    (EngineCore_DP0 pid=352574) ValueError: operands could not be broadcast
    together with shapes (6144,) (6144,) (2048,)
    
    ---------
    
    Signed-off-by: Kavulya, Soila P <soila.p.kavulya@intel.com>
    Co-authored-by: Michał Kuligowski <michal.kuligowski@intel.com>

[33mcommit b211d0e1f876fc3bcf31292179f9a045e0ba0d88[m
Author: Agata Dobrzyniewicz <160237065+adobrzyn@users.noreply.github.com>
Date:   Mon Oct 13 21:04:22 2025 +0200

    [Bugfix] Fix min linear decode value (#391)
    
    Currently out of the box linear doesn't work, this should help
    
    ---------
    
    Signed-off-by: Agata Dobrzyniewicz <adobrzyniewicz@habana.ai>
    Co-authored-by: Michał Kuligowski <michal.kuligowski@intel.com>
    Co-authored-by: Chendi.Xue <chendi.xue@intel.com>

[33mcommit 640c1bb50b480f64fd4d68701436964f49594026[m
Author: Iryna Boiko <iboiko@habana.ai>
Date:   Mon Oct 13 19:38:26 2025 +0200

    [FIX_FOR_VLLM_LATEST] Fix #24172, [Refactor]: Use M-RoPE interface directly while defining model class instead of maintaining model specific M-RoPE implementation in mrope.py (#388)
    
    Signed-off-by: Iryna Boiko <iboiko@habana.ai>
    Co-authored-by: Michał Kuligowski <michal.kuligowski@intel.com>
    Co-authored-by: Chendi.Xue <chendi.xue@intel.com>

[33mcommit a57e364e433cbc8ccafaf503d2fcc14c19e73ff5[m
Author: Xinyu Chen <xinyu1.chen@intel.com>
Date:   Mon Oct 13 23:47:42 2025 +0800

    ray: pin ray to <2.49.0 (#386)
    
    port from SW-240222. the latest ray will lost HPU device.
    
    Signed-off-by: Xinyu Chen <xinyu1.chen@intel.com>

[33mcommit 069d44abece500824ef2d66bcd29f2ab75d193ec[m
Author: Xinyu Chen <xinyu1.chen@intel.com>
Date:   Mon Oct 13 21:34:53 2025 +0800

    Correct htexp._data_ptr utility (#387)
    
    Signed-off-by: Xinyu Chen <xinyu1.chen@intel.com>

[33mcommit ef76936ef19fd355f213aa5024ecf9302be04edf[m
Author: Patryk Wolsza <patryk.wolsza@intel.com>
Date:   Mon Oct 13 14:31:36 2025 +0200

    Docs installation, quick start and build fixes (#384)
    
    Signed-off-by: PatrykWo <patryk.wolsza@intel.com>

[33mcommit 17aa25743a17afdb0690808782a41c7482cc2a73[m
Author: Tadeusz Lipinski <147429754+tlipinski1337@users.noreply.github.com>
Date:   Mon Oct 13 12:27:21 2025 +0200

    [SW-239226] Adjust junit xml filenames for retry mechanism (#382)
    
    Signed-off-by: Tadeusz Lipinski <tlipinski@habana.ai>

[33mcommit 3af0d64bd7f843735c182f2021dcd09e65f1f0a2[m
Author: Jan Wieczorek <jwieczorek@habana.ai>
Date:   Mon Oct 13 09:51:16 2025 +0200

    Enable Parallel Compilation feature for compile mode by default (#370)
    
    Co-authored-by: Chendi.Xue <chendi.xue@intel.com>
    Co-authored-by: Michał Kuligowski <michal.kuligowski@intel.com>

[33mcommit 9e1949a578438729f7faf90d2e80e332195378b8[m
Author: Iryna Boiko <iboiko@habana.ai>
Date:   Sat Oct 11 01:13:24 2025 +0200

    [FIX_FOR_VLLM_LATEST] Fix upstream crash introduced by #24486 + #24926 + #25103 + #25807 (#366)
    
    Signed-off-by: Iryna Boiko <iboiko@habana.ai>
    Signed-off-by: Chendi Xue <Chendi.Xue@intel.com>
    Co-authored-by: Chendi.Xue <chendi.xue@intel.com>

[33mcommit de045cc1623374bfb590758ee30a4e99b4e55cb8[m
Author: Patryk Wolsza <patryk.wolsza@intel.com>
Date:   Fri Oct 10 16:00:42 2025 +0200

    Change to starting page and installation (#371)
    
    Signed-off-by: PatrykWo <patryk.wolsza@intel.com>
    Co-authored-by: Chendi.Xue <chendi.xue@intel.com>
    Co-authored-by: Michał Kuligowski <michal.kuligowski@intel.com>

[33mcommit a9282bb4fb10f3774dec218212b640013ddef563[m
Author: Konrad Zawora <kzawora@habana.ai>
Date:   Fri Oct 10 11:38:05 2025 +0200

    Create LICENSE (#379)

[33mcommit 6025986015c8fffc36f61af6e25148891470febe[m
Author: Wuxun Zhang <wuxun.zhang@intel.com>
Date:   Fri Oct 10 02:25:27 2025 +0800

    Fix dp padding after upstream change #25768 (#362)
    
    `coordinate_batch_across_dp` does more work than what we need for dp
    padding here, so just implement the logic in plugin.
    
    Signed-off-by: Wuxun Zhang <wuxun.zhang@intel.com>
    Co-authored-by: Chendi.Xue <chendi.xue@intel.com>

[33mcommit d00b8311f816d44e8cb894fdd810eaf1c2a8f4a5[m
Author: Konrad Zawora <kzawora@habana.ai>
Date:   Thu Oct 9 20:24:19 2025 +0200

    [CI] Set seeds for e2e tests (#368)
    
    Current CI tests don't utilize fixed seeds, resulting in some minor
    accuracy fluctuations that can sometimes fall just under the tolerance
    threshold (likely due to random sampling). A better way would be to fix
    the seeds and always expect the same results.
    
    Signed-off-by: Konrad Zawora <kzawora@habana.ai>

[33mcommit a3b9ef1a3b2430e7c3ab7fd5fe1839dc70d724bc[m
Author: Chendi.Xue <chendi.xue@intel.com>
Date:   Thu Oct 9 10:40:04 2025 -0500

    [GITHUB ACTION] Remove commits comparison so we can rerun (#373)
    
    Signed-off-by: Chendi Xue <Chendi.Xue@intel.com>

[33mcommit 6e119cd6783a95242c7d507e8632be49947cd435[m
Author: Konrad Zawora <kzawora@habana.ai>
Date:   Thu Oct 9 16:11:49 2025 +0200

    [Bugfix] Fix bucketing UT  (#367)
    
    UT reference fix after bucketing changes in
    https://github.com/vllm-project/vllm-gaudi/pull/355 and
    https://github.com/vllm-project/vllm-gaudi/pull/350
    
    ---------
    
    Signed-off-by: Konrad Zawora <kzawora@habana.ai>

[33mcommit 9777b2729f84f23523f301ce7fb685ce77083c7e[m
Author: Konrad Zawora <kzawora@habana.ai>
Date:   Wed Oct 8 16:57:45 2025 +0200

    [Bugfix] Fix decode bucket validity condition (#355)
    
    Signed-off-by: Konrad Zawora <kzawora@habana.ai>

[33mcommit f53e8ced69762bbfdf5f9bb843fa80133a3a8f14[m
Author: Krzysztof Smusz <ksmusz@habana.ai>
Date:   Wed Oct 8 16:17:59 2025 +0200

    [Docs] README update - bucketing, warmup, defragmenter and sampler warmup (#353)
    
    Porting https://github.com/vllm-project/vllm-gaudi/pull/231 and
    https://github.com/vllm-project/vllm-gaudi/pull/278
    
    ---------
    
    Signed-off-by: Krzysztof Smusz <ksmusz@habana.ai>
    Co-authored-by: Agata Dobrzyniewicz <160237065+adobrzyn@users.noreply.github.com>
    Co-authored-by: Yaser Afshar <yaser.afshar@intel.com>

[33mcommit dbe5ec74c6c61b901a0de72ceb374cac6f51b67a[m
Author: Konrad Zawora <kzawora@habana.ai>
Date:   Wed Oct 8 14:58:27 2025 +0200

    Fix bucketing of query + num_blocks neighbor expansion (#350)
    
    Signed-off-by: Konrad Zawora <kzawora@habana.ai>

[33mcommit a77dddb10b293091afe80dada8911127ade40e83[m
Author: Konrad Zawora <kzawora@habana.ai>
Date:   Wed Oct 8 14:45:58 2025 +0200

    Remove changed-files CI step (#351)

[33mcommit f5ea1fba7ef9c4cbce92566680a69e0100fb71e2[m
Author: Artur Fierka <artur.fierka@intel.com>
Date:   Wed Oct 8 10:31:07 2025 +0200

    Fix long-context scenarios - torch.cat error (#346)
    
    Case when only prefill happens, no decodes in batch
    
    Signed-off-by: Artur Fierka <artur.fierka@intel.com>

[33mcommit 5f40761fca940a54f90a4abd4c093119f20aa3a6[m
Author: Iryna Boiko <iboiko@habana.ai>
Date:   Wed Oct 8 09:49:46 2025 +0200

    Update long context README (#256)
    
    Signed-off-by: Iryna Boiko <iboiko@habana.ai>
    Co-authored-by: Chendi.Xue <chendi.xue@intel.com>

[33mcommit b96ccb8ca42f389b55b3d1cc06d762918a472f13[m
Author: Chendi.Xue <chendi.xue@intel.com>
Date:   Tue Oct 7 11:36:43 2025 -0500

    [SKIP CI][DP] disable DP test due hourly fail (#339)
    
    Signed-off-by: Chendi Xue <Chendi.Xue@intel.com>

[33mcommit 1d7062ebd0bf10e3e0969d3b88f58b4b20d427c6[m
Author: Neelesh Gokhale <neelesh.gokhale@intel.com>
Date:   Tue Oct 7 19:01:03 2025 +0530

    Add Plugin V1 specific recipe changes (#187)
    
    Calculate prompt graphs and decode graphs as per latest 3d bucketing.
    Add max_num_batched_tokens column and set as per v1 default
    Set max_num_prefill_seqs as per v1 default
    Increased 70B mem per graph setting.
    Allow gpu_mem_utilization to be changed by user.
    
    Tested on default model len and tp of validated models.
    
    Known issues:
    New ctx_cfg (1,1,max_model_len//block_size) causes huge number of
    buckets at long context.
    New 3d prefill graphs are bigger in size than 2d, so mem per graph need
    adjustment.
    Increase in graphs due ctx_cfg and increase in graph size limits long
    contexts supported due to memory used.
    
    Signed-off-by: Neelesh Gokhale <neelesh.gokhale@intel.com>
    Co-authored-by: Michał Kuligowski <michal.kuligowski@intel.com>

[33mcommit cf2300cd979f2055f7079678db202f0e70189fa2[m
Author: Konrad Zawora <kzawora@habana.ai>
Date:   Tue Oct 7 14:21:04 2025 +0200

    Fix defragmenter compilation (#334)
    
    From testing, it seems like the forward pass of defragmenter is running
    eagerly, even though it's being compiled. This fix compiles the full
    module and significantly speeds up the forward pass in eager/compile
    mode. VLLM_DEFRAG_WITH_GRAPHS=1 is still needed for compiling
    defragmenter.
    
    Signed-off-by: Konrad Zawora <kzawora@habana.ai>

[33mcommit 1a98d502903f4c059d857d44c1631da1b78a0fb0[m
Author: Marcin Swiniarski <marcin.swiniarski@intel.com>
Date:   Tue Oct 7 09:42:57 2025 +0200

    Fix calculating used blocks (#318)
    
    Rebased original PR: https://github.com/vllm-project/vllm-gaudi/pull/232
    
    Signed-off-by: Michal Adamczyk <michal.adamczyk@intel.com>
    Co-authored-by: Michal Adamczyk <michal.adamczyk@intel.com>

[33mcommit 1e018aa342874b30cc40d212842ea5cf70fa407b[m
Author: Jacek Czaja <jacek.czaja@intel.com>
Date:   Tue Oct 7 08:59:17 2025 +0200

    RUNTIME SCALE PATCHGIN info (#317)
    
    info on runtime scale patching observable problems and how to overcome
    them
    
    ---------
    
    Signed-off-by: Jacek Czaja <jacek.czaja@intel.com>

[33mcommit ae8459efdbd71cb7a2032056971160f7c7afd250[m
Author: Krzysztof Smusz <ksmusz@habana.ai>
Date:   Tue Oct 7 08:45:39 2025 +0200

    Fix for missing graphed_buckets attr while bucketing is off (#321)
    
    Signed-off-by: Krzysztof Smusz <ksmusz@habana.ai>

[33mcommit 4780ae3c26403148b7ede17f9c9ab2c79cd934ee[m
Author: Chendi.Xue <chendi.xue@intel.com>
Date:   Mon Oct 6 14:12:49 2025 -0500

    [GITHUB ACTION] Quick fix on pre-merge enabling files change compare on fork repo (#328)
    
    Signed-off-by: Chendi Xue <Chendi.Xue@intel.com>

[33mcommit 4cdea417db08b473d44fed8a541835b58ed78749[m
Author: Chendi.Xue <chendi.xue@intel.com>
Date:   Mon Oct 6 11:33:19 2025 -0500

    [GITHUB ACTION]only trigger tests for certain folder and add skip-gaudi-tests (#325)
    
    Signed-off-by: Chendi Xue <Chendi.Xue@intel.com>

[33mcommit 8e24e7f2afaf03f53681b2b3b9182ff08e86788e[m
Author: Chendi.Xue <chendi.xue@intel.com>
Date:   Fri Oct 3 20:16:09 2025 -0500

    [GITHUB ACTION][BO] update create_branch_action (#315)
    
    1. push to pre_releases instead of releases => lack of push permission
    2. fix document
    
    This PR is verified here:
    https://github.com/vllm-project/vllm-gaudi/actions/runs/18235919834
    
    ---------
    
    Signed-off-by: Chendi Xue <Chendi.Xue@intel.com>

[33mcommit 9a94d7c0e5fd19b896e137cb60dedd0c897a8ac1[m
Author: Chendi.Xue <chendi.xue@intel.com>
Date:   Fri Oct 3 17:35:38 2025 -0500

    {GITHUB ACTION}[BO_ACTION] New action for release branch out (#312)
    
    create a new BO release after the upstream delivers the latest vllm
    release.
    
    1. Create a BO, for example, releases/v0.11.0.
    2. Generate a file with the SHA values from the upstream releases
    corresponding to the upstream release tag.
    3. Run CI check to get the BO results and save the CI output.
    
    ---------
    
    Signed-off-by: Chendi Xue <Chendi.Xue@intel.com>

[33mcommit 723b6fc42bc05131095fac0cc3a5e7ddace3d924[m
Author: Chendi.Xue <chendi.xue@intel.com>
Date:   Fri Oct 3 15:56:57 2025 -0500

    [FIX_FOR_VLLM_LATEST] update hpu_model_runner according to #25676 (#311)
    
    Failing is captured when https://github.com/vllm-project/vllm/pull/26117
    merged
    And the actual update should be done according to
    https://github.com/vllm-project/vllm/pull/25676
    
    Signed-off-by: Chendi Xue <Chendi.Xue@intel.com>

[33mcommit d73f2602b30ad4a6a24784445f094c0250b7ec6d[m
Author: Chendi.Xue <chendi.xue@intel.com>
Date:   Fri Oct 3 11:54:06 2025 -0500

    [FIX_FOR_VLLM_LATEST] fix issue brought by upstream PR #25893 (#310)
    
    https://github.com/vllm-project/vllm/pull/25893
    
    Signed-off-by: Chendi Xue <Chendi.Xue@intel.com>

[33mcommit 13c0d71aebad56009065f1454bb8e708ffee26f3[m
Author: Chendi.Xue <chendi.xue@intel.com>
Date:   Fri Oct 3 10:17:53 2025 -0500

    [README]Add NIXL installation guide in README (#308)
    
    Signed-off-by: Chendi Xue <Chendi.Xue@intel.com>
    Co-authored-by: Artur Fierka <artur.fierka@intel.com>

[33mcommit f929720e0d5da518ae31c4166a4d3a895529cdbb[m
Author: Iryna Boiko <iboiko@habana.ai>
Date:   Fri Oct 3 08:40:09 2025 +0200

    Add restriction of usage VLLM_DECODE_BLOCK_BUCKET_MAX>max_blocks (#302)
    
    Signed-off-by: Iryna Boiko <iboiko@habana.ai>

[33mcommit 6cf95e72d8a86ce0fe10dcedf525405ac2d9778f[m
Author: Iryna Boiko <iboiko@habana.ai>
Date:   Fri Oct 3 07:42:01 2025 +0200

    Update CODEOWNERS (#303)
    
    Add iboiko-habana
    
    Signed-off-by: Iryna Boiko <iboiko@habana.ai>

[33mcommit 1e34f50dd9f8bd3c9d72c945a2b8390b1c7f52a4[m
Author: Soila Kavulya <soila.p.kavulya@intel.com>
Date:   Thu Oct 2 17:33:46 2025 -0700

    [BugFix][Deepseek][INC] fix duplicate submodules for deepseek INC quantization (#305)
    
    Fixes "no measures were supplied" errors from duplicate submodules
    during INC quantization
    
    Signed-off-by: Kavulya, Soila P <soila.p.kavulya@intel.com>

[33mcommit 89903ea0d55e08768840bc69115a38b36f176d56[m
Author: Chendi.Xue <chendi.xue@intel.com>
Date:   Thu Oct 2 18:19:19 2025 -0500

    [GLM-4.5] [BugFix] make GLM-4.5 working by adding model to flatten_input list (#306)
    
    Before：
    <img width="817" height="170" alt="image"
    src="https://github.com/user-attachments/assets/35b13b53-89d7-4adb-b09e-bad3f22eb043"
    />
    
    
    After PR:
    <img width="1607" height="445" alt="image"
    src="https://github.com/user-attachments/assets/a3799784-9192-45e1-afb2-9aeabce35a72"
    />
    
    Signed-off-by: Chendi Xue <Chendi.Xue@intel.com>

[33mcommit d742d68debf52a44d85092810d189c493ac23677[m
Author: Chendi.Xue <chendi.xue@intel.com>
Date:   Thu Oct 2 17:48:04 2025 -0500

    [NIXL][Dockerfile] add docker file for latest vllm_gaudi + nixl for llmd (#307)
    
    Signed-off-by: Chendi Xue <Chendi.Xue@intel.com>

[33mcommit afaa6288dd39112890f7f7a62913317f75f221da[m
Author: Chendi.Xue <chendi.xue@intel.com>
Date:   Thu Oct 2 15:01:38 2025 -0500

    [CI][NIXL]cache/reuse pre-build wheel to skip always re-build for nixl (#304)
    
    Since we can't use nixl pypi package due to libcuda hard-dependencies,
    we build from source
    But everytime build from source is toooooo slow.
    
    This PR is to cache the pre-build wheel to host dir and if there is a
    pre-build, we can skip the build process and simply store from it
    
    ---------
    
    Signed-off-by: Chendi Xue <Chendi.Xue@intel.com>

[33mcommit 6d48f0f516b2305662add07e69c3abfd5639fddc[m
Author: Krzysztof Smusz <ksmusz@habana.ai>
Date:   Thu Oct 2 13:17:47 2025 +0200

    Fixing padded iterators in _align_and_pad (#300)
    
    The iterators we are padding currently to match the buckets are taken
    from the same object of an iterator `itertools.islice(padding_gen,
    target_len)`. This is done via `itertools.repeat(padding)` being passed
    to `pad_list` method, where `padding` is mentioned `islice(...)` object.
    
    This creates an issue, where we can call `next()` on such object only
    `target_len` number of times. If we are padding only one dimension e.g.
    from 3 to 4, then it is sufficient for our case. But if we have to pad
    more than one dimension, then all the concurrent calls to `next` on this
    generator's objects do not return any value. For example, when padding
    from 12 to 16 we'll only cover the 13th dimension and any `next()` calls
    on 14th-16th dimensions will have already emptied iterators.
    
    This change returns independent iterators objects for each padded
    dimensions, instead of using the same iterator everywhere.
    
    ---------
    
    Signed-off-by: Krzysztof Smusz <ksmusz@habana.ai>

[33mcommit 919435fb432fe4f79454e3a540c50dfa4b415610[m
Author: Michal Adamczyk <michal.adamczyk@intel.com>
Date:   Wed Oct 1 15:04:43 2025 +0200

    Use type strings to be compatible with python 3.10 (#214)
    
    Signed-off-by: Michal Adamczyk <michal.adamczyk@intel.com>
    Co-authored-by: Michał Kuligowski <michal.kuligowski@intel.com>

[33mcommit c71670e2143e550e0069b0a5cb305946de88976c[m
Author: Michał Kuligowski <michal.kuligowski@intel.com>
Date:   Wed Oct 1 14:51:19 2025 +0200

    Update CODEOWNERS (#297)

[33mcommit 410882ba1ff20c5c82b63ccdcd8586940295082c[m
Author: Iryna Boiko <iboiko@habana.ai>
Date:   Wed Oct 1 12:21:00 2025 +0200

    Add assert for empty buckets (#236)
    
    Add assert for empty buckets
    
    ---------
    
    Signed-off-by: Iryna Boiko <iboiko@habana.ai>
    Co-authored-by: Michal Gawarkiewicz <michal.gawarkiewicz@intel.com>
    Co-authored-by: Agata Dobrzyniewicz <160237065+adobrzyn@users.noreply.github.com>

[33mcommit 18ead2d5cc2fd171ee59e6c8b24ab4d681c08b92[m
Author: Agata Dobrzyniewicz <160237065+adobrzyn@users.noreply.github.com>
Date:   Wed Oct 1 05:03:27 2025 +0200

    Fix after #16229, mm (#286)
    
    upstream PR: https://github.com/vllm-project/vllm/pull/16229
    Fix is still in progress, don't merge yet
    
    ---------
    
    Signed-off-by: Agata Dobrzyniewicz <adobrzyniewicz@habana.ai>
    Signed-off-by: Chendi Xue <Chendi.Xue@intel.com>
    Co-authored-by: Chendi Xue <Chendi.Xue@intel.com>

[33mcommit a65a1bcb5577010d41baa45be6f4cd0d35188c20[m
Author: Spurthi Lokeshappa <slokeshappa@habana.ai>
Date:   Tue Sep 30 18:29:02 2025 -0700

    Fix Embeding hang (#291)
    
    Signed-off-by: slokesha <slokeshappa@habana.ai>
    Co-authored-by: Chendi.Xue <chendi.xue@intel.com>

[33mcommit 24be2abf803ea5719f9c397e08fc9d9eef0499a9[m
Author: Chendi.Xue <chendi.xue@intel.com>
Date:   Tue Sep 30 20:06:45 2025 -0500

    [FIX_FOR_VLLM_LATEST] Fix for crash introduced by upstream PR 19330 (#295)
    
    Signed-off-by: Chendi Xue <Chendi.Xue@intel.com>

[33mcommit 08d8fe43b06bff6c8a41e7067a898c85e98e2920[m
Author: Chendi.Xue <chendi.xue@intel.com>
Date:   Tue Sep 30 19:22:30 2025 -0500

    [MLA][Deepseek] Bring back deepseek after change from PR25896 (#294)
    
    SW-241658
    
    Signed-off-by: Chendi Xue <Chendi.Xue@intel.com>

[33mcommit e99600acb0b90eb6c61513c20bbf496f20d128d4[m
Author: Chendi.Xue <chendi.xue@intel.com>
Date:   Tue Sep 30 18:27:38 2025 -0500

    [NIXL] Fix crash introduced by upstream PR #25902 (#293)
    
    [SW-241656]
    
    Previously, we copied register_kv_caches funtion and did some hack for
    hpu in hpu_nixl_connector
    
    Which is not a good approach since more and more changes happened in
    hpu_nixl_connector.
    
    Now we will reuse the original register_kv_caches and fix shape in
    hpu_model_runner.
    
    ---------
    
    Signed-off-by: Chendi Xue <Chendi.Xue@intel.com>

[33mcommit c10d05f67d9e10810314d9c29b5eca82cb0c9d5e[m
Author: Chendi.Xue <chendi.xue@intel.com>
Date:   Tue Sep 30 13:11:20 2025 -0500

    [FIX_FOR_VLLM_LATEST] fix issue introduced by PR25896 and comment out still failing tests (#292)
    
    https://github.com/vllm-project/vllm/pull/25896
    
    ---------
    
    Signed-off-by: Chendi Xue <Chendi.Xue@intel.com>

[33mcommit dedff92acb8e9a6a15fe5261602bedff6796d0fa[m
Author: Jacek Czaja <jacek.czaja@intel.com>
Date:   Tue Sep 30 12:41:57 2025 +0200

    Enable H2d(runtime scale patching) for Torch compile by default (#235)
    
    Torch compile fp8 execution does not work efficiency (huge warmup time,
    memory consumption or poor performance) without runtime scale patching
    so this PR is making runtime scale patching enabled if only torch
    compile is being used.
    
    Signed-off-by: Jacek Czaja <jczaja@habana.ai>

[33mcommit 922a18fa98a08f3d7dc04dd8849dab6800aef788[m
Author: Wuxun Zhang <wuxun.zhang@intel.com>
Date:   Tue Sep 30 09:16:36 2025 +0800

    Support sequence parallel MOE after upstream #24982 (#285)
    
    After https://github.com/vllm-project/vllm/pull/24982 merged, sequence
    parallel MOE will be turned on when `enable_expert_parallel=True`,
    `tp_size > 1` and `dp_size > 1`. Since for Gaudi, there is no choice for
    `VLLM_ALL2ALL_BACKEND`, we can not easily bypass it. So this PR aims to
    support the feature.
    
    ```python
    class ParallelConfig:
    
      @property
        def use_sequence_parallel_moe(self) -> bool:
            return (envs.VLLM_ALL2ALL_BACKEND
                    in ("allgather_reducescatter", "naive",
                        "deepep_high_throughput", "deepep_low_latency")
                    and self.enable_expert_parallel
                    and self.tensor_parallel_size > 1
                    and self.data_parallel_size > 1)
    
    ```
    
    Update:
    No hard requirement on https://github.com/vllm-project/vllm/pull/25828
    
    ---------
    
    Signed-off-by: Wuxun Zhang <wuxun.zhang@intel.com>

[33mcommit 669062f10dc6bce7e7361cebd518b750018c487e[m
Author: Soila Kavulya <soila.p.kavulya@intel.com>
Date:   Mon Sep 29 16:06:27 2025 -0700

    Fix deepseek FP8 weight creation due to upstream vllm change (#281)
    
    Fixes the following assertion failure during weight creation due to
    missing weight_block_size attribute in HPU MoE method
    
    ESC[1;36m(EngineCore_DP0 pid=59025)ESC[0;0m self.experts =
    SharedFusedMoE(
    ESC[1;36m(EngineCore_DP0 pid=59025)ESC[0;0m ^^^^^^^^^^^^^^^
    ESC[1;36m(EngineCore_DP0 pid=59025)ESC[0;0m File
    "/vllm/vllm/model_executor/layers/shared_fused_moe/shared_fused_moe.py",
    line 25, in __init__
    ESC[1;36m(EngineCore_DP0 pid=59025)ESC[0;0m super().__init__(**kwargs)
    ESC[1;36m(EngineCore_DP0 pid=59025)ESC[0;0m File
    "/vllm/vllm/model_executor/layers/fused_moe/layer.py", line 1140, in
    __init__
    ESC[1;36m(EngineCore_DP0 pid=59025)ESC[0;0m
    self.quant_method.create_weights(layer=self, **moe_quant_params)
    ESC[1;36m(EngineCore_DP0 pid=59025)ESC[0;0m File
    "/vllm-gaudi/vllm_gaudi/ops/hpu_fp8.py", line 83, in create_weights
    ESC[1;36m(EngineCore_DP0 pid=59025)ESC[0;0m
    super().create_weights(*args, **kwargs)
    ESC[1;36m(EngineCore_DP0 pid=59025)ESC[0;0m File
    "/vllm/vllm/model_executor/layers/quantization/fp8.py", line 497, in
    create_weights
    ESC[1;36m(EngineCore_DP0 pid=59025)ESC[0;0m assert
    self.weight_block_size is not None
    
    ---------
    
    Signed-off-by: Kavulya, Soila P <soila.p.kavulya@intel.com>

[33mcommit fdc7edff0748c54f2876ec9d3694b87f640411dd[m
Author: Chendi.Xue <chendi.xue@intel.com>
Date:   Mon Sep 29 17:24:35 2025 -0500

    {GITHUB ACTION} remove DCO block (#290)
    
    Signed-off-by: Chendi Xue <Chendi.Xue@intel.com>

[33mcommit b7a39c24efc8dd30f6b21d2f018b11c49d22f07f[m
Author: Chendi.Xue <chendi.xue@intel.com>
Date:   Mon Sep 29 17:15:28 2025 -0500

    [Fix Hourly] install UCX from source instead using builtin wheel from nixl (#289)
    
    upstream PR introduced the crash:
    https://github.com/vllm-project/vllm/pull/25380
    
     Root cause:
    
    After this PR, nixl < 0.5.1 will not work because of lacking num_threads
    argument support.
    However, pip install nixl >= 0.5.1 will also not work because wheel
    provided UCX is compiled with hard-dependency on libcuda.so
    
     Solution is to do nixl installation by build from source.
    
    Signed-off-by: Chendi Xue <Chendi.Xue@intel.com>

[33mcommit e48e761564c85da1f6b3b1ce84baf8315d81cb98[m
Author: Wuxun Zhang <wuxun.zhang@intel.com>
Date:   Tue Sep 30 06:03:26 2025 +0800

    Fix DP dummy run cfg (#284)
    
    Signed-off-by: Wuxun Zhang <wuxun.zhang@intel.com>

[33mcommit 8c20fceacc40203851af9f4a3246b3f917cadc70[m
Author: Krzysztof Smusz <ksmusz@habana.ai>
Date:   Mon Sep 29 15:54:07 2025 +0200

    Enable modification of prompt BS (#258)
    
    Enable modification of prefill BS with usage of
    `VLLM_PROMPT_BS_BUCKET_MAX` environment variable. The default size of
    prefill BS is set to 1 (remains the same as it was before the change).
    cherry-pick: https://github.com/vllm-project/vllm-gaudi/pull/224
    
    ---------
    
    Signed-off-by: Krzysztof Smusz <ksmusz@habana.ai>
    Co-authored-by: Chendi.Xue <chendi.xue@intel.com>

[33mcommit 5aea2f66ba4499c2fddcdd6f8f530fe777a31a12[m
Author: Paweł Olejniczak <pawelolejniczak92@gmail.com>
Date:   Mon Sep 29 10:58:36 2025 +0200

    Fix for negative logits (#160)
    
    Signed-off-by: Paweł Olejniczak <polejniczakx@habana.ai>
    Co-authored-by: Agata Dobrzyniewicz <160237065+adobrzyn@users.noreply.github.com>
    Co-authored-by: Artur Fierka <artur.fierka@intel.com>

[33mcommit d17e6fe4ec74a2416afa6af674012f53599620c6[m
Author: Uri Livne <ulivne@habana.ai>
Date:   Mon Sep 29 10:52:37 2025 +0300

    [test] Add yaml files for fp8 tests (#53)
    
    Signed-off-by: Uri Livne <ulivne@habana.ai>

[33mcommit be3722f1b542bf1d444c1aba389b13bc115d730e[m
Author: Vivek Goel <vivek.goel@intel.com>
Date:   Mon Sep 29 13:20:42 2025 +0530

    Update LoRA tests (#255)
    
    Update LoRA unit-tests to (1) download LLama2 model from HF Hub, (2) run
    with warm-up enabled.
    
    ---------
    
    Signed-off-by: Vivek Goel <vivek.goel@intel.com>
    Co-authored-by: Chendi.Xue <chendi.xue@intel.com>

[33mcommit 963cf960e975751f45e6385abebfd3e6c84cf687[m
Author: Chendi.Xue <chendi.xue@intel.com>
Date:   Fri Sep 26 17:57:01 2025 -0500

    {GITHUB ACTION} Add update stable commit action (#282)
    
    Signed-off-by: Chendi Xue <Chendi.Xue@intel.com>

[33mcommit e363acccc4fbf8d19ab81bb2c53d07bb468e517c[m
Author: Chendi.Xue <chendi.xue@intel.com>
Date:   Fri Sep 26 17:51:42 2025 -0500

    [FIX_FOR_VLLM_LATEST] FIX_HOURLY_by_skip_embedding due upstream 25738 (#280)
    
    Signed-off-by: Chendi Xue <Chendi.Xue@intel.com>

[33mcommit 687ec7b2cca698004d839dd1762af8597ed8f190[m
Author: Chendi.Xue <chendi.xue@intel.com>
Date:   Fri Sep 26 14:22:44 2025 -0500

    {GITHUB ACTION}[PRE-MERGE] switch to last good commit or main based on label (#279)
    
    Signed-off-by: Chendi Xue <Chendi.Xue@intel.com>

[33mcommit a736fb8bddae544aeb45f7be3bf9ff15021b941e[m
Author: Iryna Boiko <iboiko@habana.ai>
Date:   Fri Sep 26 11:19:14 2025 +0200

    Adding prompt context flags for linear warmup (#217)
    
    Adding prompt context flags for linear warmup
    
    ---------
    
    Signed-off-by: Iryna Boiko <iboiko@habana.ai>

[33mcommit 6e65669944f178f1ffac9fbfc172eb95a576dfd7[m
Author: Agata Dobrzyniewicz <160237065+adobrzyn@users.noreply.github.com>
Date:   Fri Sep 26 09:30:26 2025 +0200

    [Unified Attention] Bucketing and Warmup for Unified Attention (#157)
    
    Signed-off-by: Agata Dobrzyniewicz <adobrzyniewicz@habana.ai>
    Co-authored-by: Konrad Zawora <kzawora@habana.ai>

[33mcommit de45479397f1599753898a10bb3f7cec83b2a900[m
Author: Chendi.Xue <chendi.xue@intel.com>
Date:   Thu Sep 25 23:31:25 2025 -0500

    {GITHUB ACTION}[PRE_MERGE] post comments if PR failed to meet DCO or mergable requirement (#273)
    
    Signed-off-by: Chendi Xue <Chendi.Xue@intel.com>

[33mcommit 34fd5951df0d5156b18b33bfee4ff910fcfad0c9[m
Author: Chendi.Xue <chendi.xue@intel.com>
Date:   Thu Sep 25 23:06:43 2025 -0500

    {GITHUB ACTION}[PRE_MERGE] last refine to enable DCO check (#271)
    
    Signed-off-by: Chendi Xue <Chendi.Xue@intel.com>

[33mcommit ba79021ef4a541094aeec50d32cf9ba28f12bd4b[m
Author: Chendi.Xue <chendi.xue@intel.com>
Date:   Thu Sep 25 21:44:45 2025 -0500

    [GITHUB ACTION][HOURLY] add force push otherwise it failed to update (#268)
    
    Signed-off-by: Chendi Xue <Chendi.Xue@intel.com>

[33mcommit 150673af55167799f1d3fb2ef65152d0fa0bc036[m
Author: Chendi.Xue <chendi.xue@intel.com>
Date:   Thu Sep 25 21:18:53 2025 -0500

    [upstream crash] fix spec decode due to upstream 24986 (#265)
    
    also update spec_decode test, previous test seems gets stucked when fail
    
    ---------
    
    Signed-off-by: Chendi Xue <Chendi.Xue@intel.com>

[33mcommit 45af00cd307cdaf3258be591fc82ba4b82660b42[m
Author: Chendi.Xue <chendi.xue@intel.com>
Date:   Thu Sep 25 21:16:21 2025 -0500

    [GITHUB ACTION] [PRE_COMMIT] pre-check before start actual CI (#270)
    
    Signed-off-by: Chendi Xue <Chendi.Xue@intel.com>

[33mcommit 073287dd14a7a5cbbb7216d7ebe5bc7e1d89e845[m
Author: Chendi.Xue <chendi.xue@intel.com>
Date:   Thu Sep 25 21:11:57 2025 -0500

    remove DCO check (#269)
    
    Signed-off-by: Chendi Xue <Chendi.Xue@intel.com>

[33mcommit d106a8dcafbf32288da173ff83f2838364c5f982[m
Author: Chendi.Xue <chendi.xue@intel.com>
Date:   Thu Sep 25 20:55:13 2025 -0500

    [GITHUB ACTION] quick fix for last update to pre-merge (#267)
    
    Signed-off-by: Chendi Xue <Chendi.Xue@intel.com>

[33mcommit e11e3a27682ec5fac31f066db220407280f322c6[m
Author: Chendi.Xue <chendi.xue@intel.com>
Date:   Thu Sep 25 20:47:33 2025 -0500

    [GITHUB ACTION] update pre-merge to block CI for not ready PR (#266)
    
    Signed-off-by: Chendi Xue <Chendi.Xue@intel.com>

[33mcommit 934f622c24d749687411e370cb81d9d29b71225d[m
Author: Chendi.Xue <chendi.xue@intel.com>
Date:   Thu Sep 25 19:58:31 2025 -0500

    [sw 239237] Add last good commit  based on PR257 (#262)
    
    original PR #257 by @bmyrcha
    After rebase and asking gemini to do refactoring, codes change is too
    big, add a new PR to avoid force push to original
    
    ---------
    
    Signed-off-by: Chendi Xue <Chendi.Xue@intel.com>
    Co-authored-by: Bartosz Myrcha <bmyrcha@habana.ai>

[33mcommit dbb90e7e72fc3461fd6dc52ffd5e5fd2b6891d24[m
Author: Chendi.Xue <chendi.xue@intel.com>
Date:   Thu Sep 25 16:31:33 2025 -0500

    [HOURLY RUN] update the scripts and action to run in seperate job (#261)
    
    Current hourly runs everything in single job, problem:
    1. early failure will cause other checks skipping
    2. hard to identify failing
    
    This PR:
    1. put all checks in individual job
    
    ---------
    
    Signed-off-by: Chendi Xue <Chendi.Xue@intel.com>

[33mcommit 0751bf688a4ad7c4e5eb09014c3996f7295f8af3[m
Author: Chendi.Xue <chendi.xue@intel.com>
Date:   Thu Sep 25 14:04:13 2025 -0500

    Fix crash introduced by 25489 -  cause PD fail (#260)
    
    https://github.com/vllm-project/vllm/pull/25489
    
    Signed-off-by: Chendi Xue <Chendi.Xue@intel.com>

[33mcommit 1d5b260d05cf13820aa95a56f94bb875a753f5db[m
Author: Chendi.Xue <chendi.xue@intel.com>
Date:   Thu Sep 25 13:33:34 2025 -0500

    fix crash introduced by upstream PR 25613 and PR23991 (#259)
    
    https://github.com/vllm-project/vllm/pull/23991
    https://github.com/vllm-project/vllm/pull/25613
    
    ---------
    
    Signed-off-by: Chendi Xue <Chendi.Xue@intel.com>

[33mcommit 60808d77994751d05af81702a3712090e9b0403c[m
Author: Krzysztof Smusz <ksmusz@habana.ai>
Date:   Thu Sep 25 10:45:26 2025 +0200

    Adding dynamic swap number and defragmenter warmup (#183)
    
    Introducing dynamic swap buckets to defragmenter, together with
    defragmenter warmup.
    
    Currently only a maximum of 32 blocks can be swapped of one iteration of
    a defragmenter. This change introduces a bucketing system, which asserts
    the minimal size bucket of swaps to be done in current defragmenter
    iteration based on actual number of blocks, that need to be swapped.
    Size of the buckets range from 8 swaps up to 512 swaps in a single
    defragmenter run.
    
    As the number of possible swap buckets grew from a single size bucket, a
    warmup of defragmenter has been added. Thanks to the warmup, no
    additional graph compilations connected to the defragmenter were visible
    during the inference.
    
    ---------
    
    Signed-off-by: Krzysztof Smusz <ksmusz@habana.ai>
    Co-authored-by: Marcin Swiniarski <marcin.swiniarski@intel.com>

[33mcommit 50a6cb568469ebe883a2d0bc5a1ba4861dc453e6[m
Author: Jozef Mamza <jozefx.mamza@intel.com>
Date:   Thu Sep 25 09:39:04 2025 +0200

    Enable group indexing gptq (#154)
    
    Signed-off-by: Jozef Mamza <jmamzax@habana.ai>
    Co-authored-by: Jozef Mamza <jmamzax@habana.ai>
    Co-authored-by: Marcin Swiniarski <marcin.swiniarski@intel.com>

[33mcommit f3813517ee792dffb7e39f04d2dd9a273f696e22[m[33m ([m[1;31morigin/bmyrcha/test_branch[m[33m)[m
Author: Soila Kavulya <soila.p.kavulya@intel.com>
Date:   Wed Sep 24 19:29:39 2025 -0700

    [SW-236002] Enable group indexing for compressed w4a16 format (#243)
    
    Enables group indexing for compressed w4a16 format similar to
    https://github.com/vllm-project/vllm-gaudi/pull/154
    
    Also fixes bug in per-channel compressed w4a16 for models like
    "nm-testing/tinyllama-oneshot-w4a16-channel-v2"
    
    ---------
    
    Signed-off-by: Chendi Xue <Chendi.Xue@intel.com>
    Signed-off-by: Kavulya, Soila P <soila.p.kavulya@intel.com>
    Signed-off-by: Jimin Ha <jimin.ha@intel.com>
    Co-authored-by: Chendi.Xue <chendi.xue@intel.com>
    Co-authored-by: Jimin Ha <jimin.ha@intel.com>

[33mcommit 79be752159212b7290c1540a1634c40c8d0e352a[m
Author: Wuxun Zhang <wuxun.zhang@intel.com>
Date:   Thu Sep 25 09:29:18 2025 +0800

    skip dp padding sync in set_forward_context (#226)
    
    Currently we always do dp padding for prefill bs/prompt length/context
    blocks and decode bs, hence there is no need to sync dp in during
    `set_forward_context`.
    
    ---------
    
    Signed-off-by: Wuxun Zhang <wuxun.zhang@intel.com>

[33mcommit d6611751fa4df6c598e32daf1c0645c42813f279[m
Author: Jimin Ha <jimin.ha@intel.com>
Date:   Wed Sep 24 17:17:24 2025 -0700

    Add HPUMultiHeadAttention with FusedSDPA (#249)
    
    MultiheadAttention is often used in vision models. Replace naive SDPA
    with FusedSDPA implementation to reduce memory usage and improve
    performance in MultiHeadAttention operations.
    
    This is part of effort for Gemma3 enablement (SW-234444)
    
    For example, with 18 images, we get OOM (27G+ is used for processing the
    vision embedding)
    With this fix: Memory is used much less(less than 1G is used for the
    same case)
    
    ```
    image_embeds:torch.Size([18, 256, 5376])
     Before  get_multimodal_embeddings: 27.56 GiB
     After get_multimodal_embeddings: 27.09 GiB
    ```
    
    Signed-off-by: Jimin Ha <jimin.ha@intel.com>
    Co-authored-by: Chendi.Xue <chendi.xue@intel.com>

[33mcommit 9c7d6d62b729e110fbebf29fcd00b1b38ba05563[m
Author: Chendi.Xue <chendi.xue@intel.com>
Date:   Wed Sep 24 18:01:26 2025 -0500

    Fix crash due to PR 25541 (#252)
    
    https://github.com/vllm-project/vllm/pull/25541
    
    ---------
    
    Signed-off-by: Chendi Xue <Chendi.Xue@intel.com>

[33mcommit 63b8af2ac325fa3454e85e5ebc7254ffc8a36c27[m
Author: Chendi.Xue <chendi.xue@intel.com>
Date:   Wed Sep 24 16:58:40 2025 -0500

    update CI file to use my PR code (#254)
    
    After switch to use pull_request_target, codes is checkout from main not
    pr, use this PR to fix
    
    Signed-off-by: Chendi Xue <Chendi.Xue@intel.com>

[33mcommit bb259fb0cb6b34314809c2996bae12505c0005cf[m
Author: Chendi.Xue <chendi.xue@intel.com>
Date:   Wed Sep 24 15:52:39 2025 -0500

    another PR for HF_TOKEN (#251)
    
    Signed-off-by: Chendi Xue <Chendi.Xue@intel.com>

[33mcommit 50cce480fc6b790a3c4c569a3f659b65768025d9[m
Author: Chendi.Xue <chendi.xue@intel.com>
Date:   Wed Sep 24 15:10:26 2025 -0500

    add hf_token for CI (#248)
    
    Enable Gemma3 CI for test.
    
    ---------
    
    Signed-off-by: Chendi Xue <Chendi.Xue@intel.com>

[33mcommit b8b1b8aad5f1ee428fc7845e144295238eb16594[m
Author: Karol Damaszke <kdamaszke@habana.ai>
Date:   Wed Sep 24 21:52:08 2025 +0200

    Add fused_experts to HPUFp8MoEMethod to fix Deepseek (#228)
    
    Currently we will fail on this check in vLLM:
    https://github.com/vllm-project/vllm/blob/main/vllm/model_executor/layers/fused_moe/layer.py#L1899
    as we don't have `fused_experts` argument at all.
    
    Signed-off-by: Karol Damaszke <kdamaszke@habana.ai>

[33mcommit 09b7837100ce86073081769b39206e1dddf8c300[m
Author: Chendi.Xue <chendi.xue@intel.com>
Date:   Wed Sep 24 11:39:13 2025 -0500

    Enable device_to_device nixl_connector support (#240)
    
    Follow SW-240337 for setup
    
    ---------
    
    Signed-off-by: Chendi Xue <Chendi.Xue@intel.com>

[33mcommit 16f478d7e7e94eff9c256aa7ea805dcc6c84e03a[m
Author: Iryna Boiko <iboiko@habana.ai>
Date:   Wed Sep 24 17:24:51 2025 +0200

    Update test owners: iboiko-habana, jkaniecki (#247)
    
    Signed-off-by: Iryna Boiko <iboiko@habana.ai>

[33mcommit 1385e8ca04f48f26ffa3491a80aefcd3cfa05e32[m
Author: Vivek Goel <vivek.goel@intel.com>
Date:   Wed Sep 24 19:38:21 2025 +0530

    Align to lora_manager changes in upstream (#244)
    
    Update plugin LoRA code to align to following PRs in upstream,
    - [Core] Modify the initialization parameters of the lora manager #25249
    - [Core] Remove tokenizer group in vLLM #24078
    
    Signed-off-by: Vivek Goel <vivek.goel@intel.com>

[33mcommit 6a7ccc90c5b97a39b80b09e39442768c85dfe697[m
Author: Karol Damaszke <kdamaszke@habana.ai>
Date:   Wed Sep 24 09:33:34 2025 +0200

    Remove sync point from _prepare_sampling (#204)
    
    Using `.copy_()` operator triggers unwanted sync points between prompts.
    Instead, we can make async copy of CPU tensor and assign it to the
    destination.
    
    ---------
    
    Signed-off-by: Karol Damaszke <kdamaszke@habana.ai>

[33mcommit 0336b6a82d01856cb4ea6b51908db3237041cc86[m
Author: Chendi.Xue <chendi.xue@intel.com>
Date:   Tue Sep 23 19:52:14 2025 -0500

    [FIX][upstream crash]Fix due upstream change 25510 (#241)
    
    https://github.com/vllm-project/vllm/pull/25510
    
    Signed-off-by: Chendi Xue <Chendi.Xue@intel.com>

[33mcommit 45f3f5ded0f291d337d58a6db3088c3da5fc7d1a[m
Author: Harish Subramony <hsubramony@habana.ai>
Date:   Tue Sep 23 15:43:29 2025 -0700

    Enable p2d2 for nixl (#237)
    
    enable p2d2 for nixl
    tp > 1 for prefill and decode
    
    ---------
    
    Signed-off-by: Harish Subramony <hsubramony@habana.ai>

[33mcommit 04c8263292c051adb92a694fbce9d2543be03adc[m
Author: Chendi.Xue <chendi.xue@intel.com>
Date:   Tue Sep 23 15:03:14 2025 -0500

    [FIX][Upstream caused crash] Fix crash caused by upstream PR 25184 (#238)
    
    Signed-off-by: Chendi Xue <Chendi.Xue@intel.com>

[33mcommit dbf656d139b984637a2ab95475a5cb300d584a7f[m
Author: Chendi.Xue <chendi.xue@intel.com>
Date:   Tue Sep 23 10:37:59 2025 -0500

    use vllm intree API to enable synced_model_load, #25126 (#208)
    
    with https://github.com/vllm-project/vllm/pull/25126, we can do
    synced_model_loading in through platform interface now
    
    Have set VLLM_WEIGHT_LOAD_FORCE_SYNC=true, so all models will do
    synced_model_load by default
    
    Signed-off-by: Chendi Xue <Chendi.Xue@intel.com>

[33mcommit a2f6b715502ca027678e85020cb5a231b6dadd6a[m
Author: Patryk Wolsza <patryk.wolsza@intel.com>
Date:   Tue Sep 23 16:59:52 2025 +0200

    V0.10.2 docker updates / benchmark serving section (#191) - cherry-pick (#200)
    
    Moving fix for the benchmark section from release branch to main.
    
    Signed-off-by: PatrykWo <patryk.wolsza@intel.com>

[33mcommit a0bbe78f442d5c5e26b383e83b944619d63a5c08[m
Author: Jimin Ha <jimin.ha@intel.com>
Date:   Mon Sep 22 18:07:09 2025 -0700

    Update the script fix for gemma-3-4b test  (#225)
    
    Fixing the test model file name to gemma-3-4b-it.yaml and also upload
    the model file.
    
    Test is failing at
    https://github.com/vllm-project/vllm-gaudi/actions/runs/17926460012/job/50985856988?pr=150
    due to missing file after
    https://github.com/vllm-project/vllm-gaudi/pull/150.
    ```
    Testing gemma-3-4b-it
    VLLM_SKIP_WARMUP=true PT_HPU_LAZY_MODE=1 VLLM_USE_V1=1 python -u vllm-gaudi/tests/models/language/generation/generation_mm.py --model-card-path vllm-gaudi/tests/full_tests/model_cards/gemma-3-27b-it.yaml
    ...
    FileNotFoundError: [Errno 2] No such file or directory: 'vllm-gaudi/tests/full_tests/model_cards/gemma-3-27b-it.yaml'
    ```
    
    ---------
    
    Signed-off-by: Jimin Ha <jimin.ha@intel.com>

[33mcommit 481b163a5ae23edb7939521f7dbff34deea0a6a3[m
Author: Jimin Ha <jimin.ha@intel.com>
Date:   Mon Sep 22 16:13:11 2025 -0700

    Enable interleaved sliding window for gemma3 (#150)
    
    Enable interleaved sliding window for gemma3, this provides initial
    Gemma3 enablement for V1 with functional support.
        - Add prefill mask creation based on sliding_window
        - Add decode window_block operation for sliding_window
        - Port implementation from v0 Gemma3 code
        - Add prefix-prefill block calculation with sliding_window
    
    The following items are remaining work that needs to be ported from V0
    with further analysis:
        - Multimodal warmup support
    - Gemma3 model file optimization for the vision tower to split images
    into buckets
        - FusedSDPA with window_size support for longer prompts
        - Split_qkv and MultiHeadAttention with FusedSDPA
    
    `PT_HPU_LAZY_MODE=1 VLLM_SKIP_WARMUP=True lm_eval --model vllm
    --model_args
    pretrained=google/gemma-3-4b-it,tensor_parallel_size=1,distributed_executor_backend=mp,trust_remote_code=true,max_model_len=4096,dtype=bfloat16,max_num_seqs=128
    --tasks gsm8k --batch_size 128 --num_fewshot 5`
    
    |Tasks|Version| Filter |n-shot| Metric | |Value | |Stderr|
    
    |-----|------:|----------------|-----:|-----------|---|-----:|---|-----:|
    |gsm8k| 3|flexible-extract| 5|exact_match|↑ |0.6778|± |0.0129|
    | | |strict-match | 5|exact_match|↑ |0.6755|± |0.0129|
    
    ---------
    
    Signed-off-by: Jimin Ha <jimin.ha@intel.com>
    Signed-off-by: Mohit Deopujari <mdeopujari@habana.ai>
    Co-authored-by: Mohit Deopujari <mdeopujari@habana.ai>

[33mcommit e76373ce681ef233291789131a0db44a78ada7e4[m
Author: Wuxun Zhang <wuxun.zhang@intel.com>
Date:   Mon Sep 22 23:53:39 2025 +0800

    Fix DP dummy run crash for P/D  (#194)
    
    - Skip kvtransfer load/save for dummy run
    - For dummy run, `SchedulerOutput` is created without
    `kv_connector_metadata`
    - Fix real prefill batch calculation
    
    ---------
    
    Signed-off-by: Wuxun Zhang <wuxun.zhang@intel.com>

[33mcommit d930dea9c28f0bd32fcdc48e73e06dfb2d7e43fc[m
Author: Agata Dobrzyniewicz <160237065+adobrzyn@users.noreply.github.com>
Date:   Mon Sep 22 15:57:07 2025 +0200

    [BUGFIX] Fix after PR25332 & 25321 & 25366 (#215)
    
    Culprit commit : https://github.com/vllm-project/vllm/pull/25332 and
    https://github.com/vllm-project/vllm/pull/25321 and
    https://github.com/vllm-project/vllm/pull/25366
    
    ---------
    
    Signed-off-by: Agata Dobrzyniewicz <adobrzyniewicz@habana.ai>
    Signed-off-by: Konrad Zawora <kzawora@habana.ai>
    Co-authored-by: Konrad Zawora <kzawora@habana.ai>

[33mcommit 82408b0de88fb5df225cf06da55ad0847e72c198[m
Author: Konrad Zawora <kzawora@habana.ai>
Date:   Mon Sep 22 10:15:55 2025 +0200

    Update project URLs in docs/README.md

[33mcommit 5963cba2dcda453975da155bbab00ba37b683a4e[m
Author: Konrad Zawora <kzawora@habana.ai>
Date:   Mon Sep 22 10:12:26 2025 +0200

    Update README.md

[33mcommit 8fcf4667c46cd07d84158b97633d80f68b7a9c17[m
Author: Konrad Zawora <kzawora@habana.ai>
Date:   Mon Sep 22 10:03:04 2025 +0200

    Update .readthedocs.yaml

[33mcommit 9a4f248ac992237bb42ac5fe9ad6490011214576[m
Author: Konrad Zawora <kzawora@habana.ai>
Date:   Mon Sep 22 10:00:59 2025 +0200

    Create .readthedocs.yaml (#219)

[33mcommit df5dfda12b627af5bdce4b89770d314e55cdc0ee[m
Author: Chendi.Xue <chendi.xue@intel.com>
Date:   Sat Sep 20 01:13:30 2025 -0500

    Move hourly to aicf-gaudi2-07 (#211)
    
    Signed-off-by: Chendi Xue <Chendi.Xue@intel.com>

[33mcommit 5ce8e5c20aa40a103e7ac118252188f8c094ad58[m
Author: Chendi.Xue <chendi.xue@intel.com>
Date:   Fri Sep 19 18:47:04 2025 -0500

    update HOURLY docker image and move DP to separate test run (#209)
    
    Signed-off-by: Chendi Xue <Chendi.Xue@intel.com>

[33mcommit 2c89440ac1f38b570c74c50b60f4a6f54bb28125[m
Author: Soila Kavulya <soila.p.kavulya@intel.com>
Date:   Fri Sep 19 14:04:00 2025 -0700

    [SW-236002] Support compressed int4 w4a16 format (#193)
    
    Add support for compressed int4 w4a16 format for dense and MoE models
    [SW-236002]
    
    ---------
    
    Signed-off-by: Kavulya, Soila P <soila.p.kavulya@intel.com>

[33mcommit 28366277b53c026d984d0ba1483cb8bbb29b59a6[m
Author: Michal Adamczyk <madamczyk@habana.ai>
Date:   Fri Sep 19 18:51:28 2025 +0200

    Unified mixed batches (#196)
    
    Signed-off-by: Michal Adamczyk <michal.adamczyk@intel.com>
    Co-authored-by: Agata Dobrzyniewicz <160237065+adobrzyn@users.noreply.github.com>

[33mcommit 04c82d5c7b5c7813fc4b17bf8f086aea45adc9ac[m
Author: Kamil Kaczor <kamil.kaczor@intel.com>
Date:   Fri Sep 19 16:01:52 2025 +0200

    Fix swap in defragmentator (#182)
    
    Signed-off-by: Kamil Kaczor <kamil.kaczor@intel.com>
    Co-authored-by: Agata Dobrzyniewicz <160237065+adobrzyn@users.noreply.github.com>

[33mcommit 0f56cae0236f0c25550b6145bff2ec7126850467[m
Author: Wuxun Zhang <wuxun.zhang@intel.com>
Date:   Fri Sep 19 15:07:37 2025 +0800

    fix block bucket size for DP+contiguous PA (#171)
    
    This can be reproduced when increasing input reqs.
    
    Signed-off-by: Wuxun Zhang <wuxun.zhang@intel.com>

[33mcommit 7c7ee9ccf569992a3acf82bfb8f3c99764d21109[m
Author: Thomas Atta-Fosu <thomas.atta-fosu@intel.com>
Date:   Thu Sep 18 14:06:20 2025 -0700

    [SW-240630] Qwen3-30B-MoE: Flatten post-attn seqs and restore model output shape (#176)
    
    - After [#24772](https://github.com/vllm-project/vllm/pull/24772) an
    assertion on input shape in `Qwen3MoeSparseMoeBlock` exposes a design
    compatibility [issue
    #24806](https://github.com/vllm-project/vllm/issues/24806): attn output
    is expected to flatten `batch_size x seqlen`
    - This PR aligns the attention output as such, and restores the output
    shape of the forward pass afterwards
    
    ---------
    
    Signed-off-by: attafosu <thomas.atta-fosu@intel.com>
    Co-authored-by: Chendi.Xue <chendi.xue@intel.com>

[33mcommit 5a4e0ec6cca10b7510e26bb4b6ed4e86a2da515d[m
Author: Agata Dobrzyniewicz <160237065+adobrzyn@users.noreply.github.com>
Date:   Thu Sep 18 20:32:09 2025 +0200

    [BUGFIX] Fix hourly after PR#22772 (#197)
    
    Culprit commit: https://github.com/vllm-project/vllm/pull/22772
    
    ---------
    
    Signed-off-by: Agata Dobrzyniewicz <adobrzyniewicz@habana.ai>

[33mcommit 963170cb5d05da1181f52d1920dff68576974050[m
Author: Tianmu Li <tianmu.li@intel.com>
Date:   Wed Sep 17 13:23:45 2025 -0700

    Cache token ids on device for async_scheduling (#184)
    
    Signed-off-by: Tianmu Li <tianmu.li@intel.com>
    Co-authored-by: Chendi.Xue <chendi.xue@intel.com>

[33mcommit 322bb1cf91eb26d55b84572de8195aa96ca6c1be[m
Author: Wuxun Zhang <wuxun.zhang@intel.com>
Date:   Thu Sep 18 02:22:18 2025 +0800

    Fix dp sync after upstream change #24105 (#179)
    
    - fix behavior of Lazy + `enforce_eager` in which case hpu graph is NOT
    used
    - disable device group for dp sync when hpu graph is used
    - enable DP CI test again
    
    ---------
    
    Signed-off-by: Wuxun Zhang <wuxun.zhang@intel.com>
    Co-authored-by: Chendi.Xue <chendi.xue@intel.com>

[33mcommit a3dce5ceca1047fa4197312f94fa74322cbaa163[m
Author: Agata Dobrzyniewicz <160237065+adobrzyn@users.noreply.github.com>
Date:   Wed Sep 17 19:23:50 2025 +0200

    CI fix (#186)
    
    https://github.com/vllm-project/vllm/pull/24795 and
    https://github.com/vllm-project/vllm/pull/24615 and
    https://github.com/vllm-project/vllm/pull/24078
    
    ---------
    
    Signed-off-by: Agata Dobrzyniewicz <adobrzyniewicz@habana.ai>

[33mcommit e677dd1f7531e79d1406b435b0409e7160f98bc2[m
Author: PatW <patryk.wolsza@intel.com>
Date:   Wed Sep 17 01:31:59 2025 +0200

    Fix in docker compose functionality for v1-plugin (#185)
    
    Signed-off-by: PatrykWo <patryk.wolsza@intel.com>

[33mcommit b94548aec4d9710ade9ea8aee0b19d74fd855ced[m
Author: Thomas Atta-Fosu <thomas.atta-fosu@intel.com>
Date:   Tue Sep 16 16:24:12 2025 -0700

    Bug fix: hpu mrope (#167)
    
    - HPU Mrope implementation had a bug which was exposed by
    https://github.com/vllm-project/vllm/pull/24444
    - Initial workaround was to use the default implementation:
    https://github.com/vllm-project/vllm-gaudi/pull/162
    - This PR fixes the bug in the HPU mrope
    
    ---------
    
    Signed-off-by: attafosu <thomas.atta-fosu@intel.com>
    Co-authored-by: Chendi.Xue <chendi.xue@intel.com>

[33mcommit b6a1c7c1dfe6494a95fe53a35c66d5ed8c71e339[m
Author: Xinyu Chen <xinyu1.chen@intel.com>
Date:   Wed Sep 17 00:06:19 2025 +0800

    Support Ray distributed executor (#169)
    
    Signed-off-by: Xinyu Chen <xinyu1.chen@intel.com>
    Co-authored-by: Chendi.Xue <chendi.xue@intel.com>

[33mcommit d37a2ee33c9bf6e89cac5463e5cad807a3ed5178[m
Author: Xinyu Chen <xinyu1.chen@intel.com>
Date:   Tue Sep 16 10:29:31 2025 +0800

    Introduce VLLM_SCALE_ADJUSTMENT (#164)
    
    Introduce VLLM_SCALE_ADJUSTMENT to speed up the weight loading for
    pre-converted model on g2
    
    Signed-off-by: Xinyu Chen <xinyu1.chen@intel.com>
    Co-authored-by: Chendi.Xue <chendi.xue@intel.com>

[33mcommit 05b0de53108187c697d2b4f2929e25c0f4c09e97[m
Author: Tianmu Li <tianmu.li@intel.com>
Date:   Mon Sep 15 17:25:43 2025 -0700

    Added fix for VLLM_WEIGHT_LOAD_FORCE_SYNC (#173)
    
    Signed-off-by: Tianmu Li <tianmu.li@intel.com>
    Co-authored-by: Chendi.Xue <chendi.xue@intel.com>

[33mcommit dca67192887952d3912b7c00ca96e782fb68c958[m
Author: Tianmu Li <tianmu.li@intel.com>
Date:   Mon Sep 15 15:11:31 2025 -0700

    Fully overlap model execution (#134)
    
    Dependent on https://github.com/vllm-project/vllm/pull/23569
    
    ---------
    
    Signed-off-by: Tianmu Li <tianmu.li@intel.com>
    Co-authored-by: Chendi.Xue <chendi.xue@intel.com>

[33mcommit 1d3731bda056ff9eca578d77024d98c364f12c99[m
Author: Chendi.Xue <chendi.xue@intel.com>
Date:   Mon Sep 15 15:45:11 2025 -0500

    [BUG][Disable CI] Disable DP test due recent upstream change failed HPU DP (#177)
    
    Signed-off-by: Chendi Xue <Chendi.Xue@intel.com>

[33mcommit dcad6a08118f05d4a2680031c054d3b430370c16[m
Author: Chendi.Xue <chendi.xue@intel.com>
Date:   Mon Sep 15 14:00:56 2025 -0500

    [CI FIX] Fix issue introduced by upstream #24745 (#174)
    
    Signed-off-by: Chendi Xue <Chendi.Xue@intel.com>

[33mcommit d778af061d24e7c6f93c281e9f75af79909d3370[m
Author: Chendi.Xue <chendi.xue@intel.com>
Date:   Mon Sep 15 12:55:37 2025 -0500

    [CI FIX]Fix issue introduced by upstream PR #23974 (#172)
    
    [PR #23974](https://github.com/vllm-project/vllm/pull/23974) updated the
    vllm.v1.core.kv_cache_utils.get_kv_cache_configs api.
    
    Signed-off-by: Chendi Xue <Chendi.Xue@intel.com>

[33mcommit 98e89bd1ae254319d204df22c850b8a08ff864b9[m
Author: Thomas Atta-Fosu <thomas.atta-fosu@intel.com>
Date:   Mon Sep 15 07:47:03 2025 -0700

    [TEMP-WA] Skip Qwen3-30B-A3B in tests - Bug introduced in upstream #24772  (#168)
    
    - After [#24772](https://github.com/vllm-project/vllm/pull/24772),
    there's a bug in `Qwen3MoeSparseMoeBlock` forward call (assertion for 2d
    max hidden_state dims)
    - An issue has been created pending resolution
    [#24806](https://github.com/vllm-project/vllm/issues/24806)
    - Re-enable once resolved in upstream
    
    Signed-off-by: attafosu <thomas.atta-fosu@intel.com>

[33mcommit 21426a37f301bd29aebe6c4eb72d623ed5f369ca[m
Author: Agata Dobrzyniewicz <160237065+adobrzyn@users.noreply.github.com>
Date:   Mon Sep 15 16:19:57 2025 +0200

    TESTOWNERS update (#165)
    
    lmk if you want to be added
    
    ---------
    
    Signed-off-by: Agata Dobrzyniewicz <adobrzyniewicz@habana.ai>
    Co-authored-by: Chendi.Xue <chendi.xue@intel.com>
    Co-authored-by: Konrad Zawora <kzawora@habana.ai>

[33mcommit 39ac27a7378caa583d902e8c59b95d4eddd78e3a[m
Author: Chendi.Xue <chendi.xue@intel.com>
Date:   Fri Sep 12 16:20:17 2025 -0500

    [BUGFIX] warmup failed after PR104, propose fix in this PR (#148)
    
    1. skip any warmup seq_len is larger than max_num_tokens.
    PR104 will prepare warmup seq_len which is bigger than max_num_tokens,
    which will fail the warmup when preparing inputs.
    2. add a warmup sharegpt perf test
    
    Signed-off-by: Chendi.Xue <chendi.xue@intel.com>

[33mcommit 505d5cfdde6b6eabb545bf9d44c58c2644ec8a68[m
Author: Chendi.Xue <chendi.xue@intel.com>
Date:   Fri Sep 12 14:55:41 2025 -0500

    [HOURLY FIX] For upstream PR-24548 changes (#166)
    
    Fix for upstream PR - https://github.com/vllm-project/vllm/pull/24548
    
    ---------
    
    Signed-off-by: Chendi.Xue <chendi.xue@intel.com>

[33mcommit f956f7da24c70165a290ee7afd68e6543d01a6ae[m
Author: Michal Muszynski <141021743+mmuszynskihabana@users.noreply.github.com>
Date:   Fri Sep 12 13:44:37 2025 +0200

    Allow building vllm-plugin docker with upstream torch (#155)
    
    To create vllm-plugin docker for RHEL 9.6 with torch package taken from
    upstream we need to modify 'FROM' directive - image should be based on
    pytorch-upstream-installer.
    
    'TORCH_TYPE_SUFFIX' arg will have one of two values:
    - empty string (default)
    - 'upstream-'
    
    Signed-off-by: Michal Muszynski <mmuszynski@habana.ai>
    Co-authored-by: Konrad Zawora <kzawora@habana.ai>

[33mcommit 903f3ad740472658d09f7cd97a22a5e1260b0f7d[m
Author: Artur Fierka <artur.fierka@intel.com>
Date:   Fri Sep 12 13:39:17 2025 +0200

    Reenabling llama4 models (#128)
    
    Signed-off-by: Artur Fierka <artur.fierka@intel.com>
    Co-authored-by: Konrad Zawora <kzawora@habana.ai>

[33mcommit bc79342231148445595b1012af80efb3e0101d03[m
Author: Chendi.Xue <chendi.xue@intel.com>
Date:   Fri Sep 12 06:37:41 2025 -0500

    [BUGFIX] qwen2.5-vl failed after PR24444, provide a temp solution (#162)
    
    The reason qwen2.5-vl failed after PR24444 is because that HPU is kept
    using forward_native for MRotaryEmbedding
    After PR24444 merged, it firstly time will go forward_oot instead of
    forward_native, while forward_oot is not implemented correctly.
    
    Temporary switch to use forward_native in this PR
    
    ---------
    
    Signed-off-by: Chendi.Xue <chendi.xue@intel.com>
    Co-authored-by: Konrad Zawora <kzawora@habana.ai>

[33mcommit 2e8989c749fb627bbbb2d14a121be79c035ba023[m
Author: Chendi.Xue <chendi.xue@intel.com>
Date:   Fri Sep 12 06:37:05 2025 -0500

    [Feature][SpecDecode][Part2] Eagle3,MTP enabling, accept_rate improvement (#142)
    
    Spec Decode Part2 - [SW-234434]
    1. enabled eagle3 and MTP
    2. enable draft model to execute prefill, previously with prefill missed
    for draft model, first few draft token will not be accepted
    3. did token by token comparison with CUDA to enable the correctness
    4. did perf profiling to reduce overhead
    ---
    
    Changes in this PR:
    For hpu_model_runner,
    1. most of changes are residing in spec_decode specific functions,
    including prepare_spec_decodes_input, propose_draft_token_ids, etc...
    2. changes to general code path:
    2.1 In prepare_decode_inputs: move token_ids_devices and
    positions_device before prepare_spec_decode_inputs
    2.2 in execute_model, added 4 lists for
    non_flattened_hidden_states_prefills, aux_hidden_states_prefills,
    aux_hidden_states_prefills, aux_hidden_states_prefills, which are used
    by spec_decode propose_draft_token; Meanwhile, updated
    execute_model_generic to return corresponding values.
    2.3 Move spec_decode propose_draft_token_ids to very end of
    execute_model so we can use postprocessed_sampled_token_ids and updated
    self.input_batch.input_ids_cpu
    
    ---
    
    validations:
    1. Tested with sharedgpt 1000 prompts with this PR + disable spec
    decode, no obvious new graphs added
    2. verified with eagle, eagle3,mtp, ngram; up to 1.2x speedup observed.
    => details updated in jira ticket
    
    ----
    TODO for 1.24:
    1. warmup for spec decode
    4. multi draft token enabling
    5. deepseek_mtp acc_rate is not as good as v0 (no clue of the reason)
    
    ---------
    
    Signed-off-by: Chendi.Xue <chendi.xue@intel.com>
    Co-authored-by: Konrad Zawora <kzawora@habana.ai>

[33mcommit 56158a3f3008e363789448fb63a3b3648802411c[m
Author: Yi Liu <yi4.liu@intel.com>
Date:   Fri Sep 12 10:07:59 2025 +0800

    Re-quantize FP8 model with INC (#114)
    
    ```bash
    QUANT_CONFIG=vllm-gaudi/tests/models/language/generation/inc_dynamic_quant.json VLLM_HPU_FORCE_CHANNEL_FP8=false  \
    HABANA_VISIBLE_DEVICES=all VLLM_CONTIGUOUS_PA=False VLLM_SKIP_WARMUP=true PT_HPU_LAZY_MODE=1 VLLM_USE_V1=1 \
    VLLM_SKIP_WARMUP=true VLLM_CONTIGUOUS_PA=False PT_HPU_LAZY_MODE=1 \
    lm_eval   --model vllm --tasks gsm8k --num_fewshot 5 --batch_size 128 \
    --model_args "pretrained=/mnt/disk8/Qwen/Qwen3-8B-FP8,tensor_parallel_size=1,trust_remote_code=true,max_model_len=4096,dtype=bfloat16"
    ```
    ```bash
    vllm (pretrained=/mnt/disk8/Qwen/Qwen3-8B-FP8,tensor_parallel_size=1,trust_remote_code=true,max_model_len=4096,dtype=bfloat16), gen_kwargs: (None), limit: None, num_fewshot: 5, batch_size: 128
    |Tasks|Version|     Filter     |n-shot|  Metric   |   |Value |   |Stderr|
    |-----|------:|----------------|-----:|-----------|---|-----:|---|-----:|
    |gsm8k|      3|flexible-extract|     5|exact_match|↑  |0.8817|±  |0.0089|
    |     |       |strict-match    |     5|exact_match|↑  |0.8749|±  |0.0091|
    
    ```
    
    ---------
    
    Signed-off-by: yiliu30 <yi4.liu@intel.com>
    Co-authored-by: Chendi.Xue <chendi.xue@intel.com>

[33mcommit 563111e85b9842c9a4ddbff617d7e70391f048a2[m
Author: Harish Subramony <hsubramony@habana.ai>
Date:   Thu Sep 11 17:29:07 2025 -0700

    update nixl version in requirements (#163)
    
    Signed-off-by: Harish Subramony <hsubramony@habana.ai>

[33mcommit dc0d049669cdd399d32932f1bd05bf1c253b105e[m
Author: Harish Subramony <harish.subramony@intel.com>
Date:   Thu Sep 11 15:22:20 2025 -0700

    initial port for nixl  (#100)
    
    port nixl
    
    ---------
    
    Signed-off-by: Harish Subramony <hsubramony@habana.ai>
    Signed-off-by: Chendi.Xue <chendi.xue@intel.com>
    Co-authored-by: Chendi.Xue <chendi.xue@intel.com>

[33mcommit 56c4577300e9a3f9f5e5d7a59f4df12946261cf9[m
Author: Chendi.Xue <chendi.xue@intel.com>
Date:   Thu Sep 11 16:05:24 2025 -0500

    [Quick fix for CI]fix CI break on Qwen2.5-vl and update docker image (#161)
    
    Signed-off-by: Chendi.Xue <chendi.xue@intel.com>

[33mcommit 50f990662809d8cb5203b6dd3bddc3f98b223190[m
Author: Wuxun Zhang <wuxun.zhang@intel.com>
Date:   Fri Sep 12 00:08:08 2025 +0800

    Fix dummy decode input for DP (#151)
    
    For DP, dummy decode input data will be created with
    `schedulerOutput=None`, this is to skip prepare spec_decode_inputs in
    this case.
    
    ---------
    
    Signed-off-by: Wuxun Zhang <wuxun.zhang@intel.com>
    Co-authored-by: Chendi.Xue <chendi.xue@intel.com>
    Co-authored-by: Konrad Zawora <kzawora@habana.ai>

[33mcommit 6603379b9f81ff48d220340da50b3bc3737dd9c0[m
Author: Konrad Zawora <kzawora@habana.ai>
Date:   Thu Sep 11 15:07:51 2025 +0200

    [CI] Jenkins false positive bugfix (#159)
    
    This PR fixes current Schrödinger's CI pipelines - it makes failing
    pipelines fail (failures reported as false positives are now true
    negatives), and it also makes failing pipelines pass (former false
    positives are now true positives due to adjusted tolerances). Basically,
    if you break something == CI pipeline will fail as it should, and
    pipelines that used to be broken, are now not broken.
    
    ---------
    
    Signed-off-by: Konrad Zawora <kzawora@habana.ai>

[33mcommit fa223d5685225ca02ec8f67d30dcf56c13c85a42[m
Author: Karol Damaszke <kdamaszke@habana.ai>
Date:   Thu Sep 11 14:40:20 2025 +0200

    Patch FusedMoE forward to avoid dynamo recompilations (#158)
    
    Currently there are dynamo recompilations for each layer, due to
    `layer_name` arg passed to the forward function:
    ```
    (Worker pid=79578) [rank0]:W0910 15:26:29.372000 79578 torch/_dynamo/convert_frame.py:1016] [3/8] torch._dynamo hit config.recompile_limit (8)
    (Worker pid=79578) [rank0]:W0910 15:26:29.372000 79578 torch/_dynamo/convert_frame.py:1016] [3/8]    function: 'forward' (vllm/vllm/model_executor/models/mixtral.py:230)
    (Worker pid=79578) [rank0]:W0910 15:26:29.372000 79578 torch/_dynamo/convert_frame.py:1016] [3/8]    last reason: 3/7: self._modules['block_sparse_moe']._modules['experts'].layer_name == 'model.layers.7.block_sparse_moe.experts'
    (Worker pid=79578) [rank0]:W0910 15:26:29.372000 79578 torch/_dynamo/convert_frame.py:1016] [3/8] To log all recompilation reasons, use TORCH_LOGS="recompiles".
    (Worker pid=79578) [rank0]:W0910 15:26:29.372000 79578 torch/_dynamo/convert_frame.py:1016] [3/8] To diagnose recompilation issues, see https://pytorch.org/docs/main/torch.compiler_troubleshooting.html.
    ```
    
    It causes huge perf drop once using torch.compile instead of lazy mode
    (~5x worse perf) -- on the traces we can observe a lot of
    `transpose_mme` and `broadcast_nd` blocks, that are between all MME
    nodes:
    <img width="353" height="167" alt="image"
    src="https://github.com/user-attachments/assets/343ae137-20d0-447c-b687-387eefe19e41"
    />
    
    
    To avoid it, I proposed a similar solution we used to have in vllm-fork
    ([FusedMoe.__init__()](https://github.com/HabanaAI/vllm-fork/blob/habana_main/vllm/model_executor/layers/fused_moe/layer.py#L866)
    and
    [FusedMoE.forward()](https://github.com/HabanaAI/vllm-fork/blob/habana_main/vllm/model_executor/layers/fused_moe/layer.py#L1442))
    -- using `FusedMoE.forward_impl()` function for the cases where
    `dp_size` is equal to 1.
    
    ---------
    
    Signed-off-by: Karol Damaszke <kdamaszke@habana.ai>

[33mcommit c82cdcf6d743b09894c15dbd59f2a912bdcb673c[m
Author: Konrad Zawora <kzawora@habana.ai>
Date:   Thu Sep 11 10:56:06 2025 +0200

    Add TESTOWNERS (#153)
    
    Signed-off-by: Konrad Zawora <kzawora@habana.ai>

[33mcommit 5acf426bad59574bf2fe87f23c8a276078799ed5[m
Author: Wuxun Zhang <wuxun.zhang@intel.com>
Date:   Thu Sep 11 00:30:13 2025 +0800

    Add DP into CI (#146)
    
    Signed-off-by: Wuxun Zhang <wuxun.zhang@intel.com>
    Co-authored-by: Agata Dobrzyniewicz <160237065+adobrzyn@users.noreply.github.com>

[33mcommit c520a1f331c72732dd05410810bc41051c956803[m
Author: Kacper Pietkun <kpietkun@habana.ai>
Date:   Wed Sep 10 14:14:14 2025 +0200

    Enable sampler compilation (#95)
    
    Signed-off-by: Kacper Pietkun <kpietkun@habana.ai>

[33mcommit b5a197d9f92b5965caaea3f94683678004edf040[m
Author: Artur Fierka <artur.fierka@intel.com>
Date:   Wed Sep 10 10:42:10 2025 +0200

    Update CODEOWNERS (#135)
    
    Add @afierka-intel user
    
    Signed-off-by: Artur Fierka <artur.fierka@intel.com>

[33mcommit 5139bf6e2a859cd74b28f32dda25e0fd2e606236[m
Author: Chendi.Xue <chendi.xue@intel.com>
Date:   Tue Sep 9 22:29:42 2025 -0500

    [FIX HOURLY]Remove DP test from Hourly (#147)
    
    Signed-off-by: Chendi.Xue <chendi.xue@intel.com>

[33mcommit 472c8f45de798d102749c19f5ffbd42841697cd9[m
Author: Konrad Zawora <kzawora@habana.ai>
Date:   Tue Sep 9 21:19:46 2025 +0200

    Increase allowed line length to 120 + reformat accordingly (#130)
    
    Signed-off-by: Konrad Zawora <kzawora@habana.ai>

[33mcommit a2bcfca5a4154af394b2f0c82f802bcc76c51ac5[m
Author: Wuxun Zhang <wuxun.zhang@intel.com>
Date:   Wed Sep 10 02:41:49 2025 +0800

    Add data parallel support (#80)
    
    This is to add data parallel support for V1 gaudi plugin.
    
    - [x] add dp aware padding
    - [x] use all_gather and reduce_scatter
    - [x] add data parallel example
    
    ---------
    
    Signed-off-by: Wuxun Zhang <wuxun.zhang@intel.com>
    Co-authored-by: Konrad Zawora <kzawora@habana.ai>
    Co-authored-by: Agata Dobrzyniewicz <160237065+adobrzyn@users.noreply.github.com>

[33mcommit c2a0171e14540159335f15de03dbb54cdab4f5f5[m
Author: Vivek Goel <vgoel@habana.ai>
Date:   Tue Sep 9 20:18:41 2025 +0530

    Add support for LoRA (#51)
    
    Signed-off-by: Vivek <vgoel@habana.ai>

[33mcommit a9702fe1b0d52b9dd30ffab436f0cec4e145c74a[m
Author: Krzysztof Smusz <ksmusz@habana.ai>
Date:   Tue Sep 9 15:09:39 2025 +0200

    Introducing sampler warmup as separate warmup step (#131)
    
    Warming up the sampler with different configurations removes graph
    recompilations of bigger sampler graphs seen within the actual
    execution. As tested with example workloads and batch sizes, the only
    recompilations left from the sampler are from minor graphs, which have
    minimal influence to the execution time.
    
    The warmup of the sampler takes around 1-3 seconds, depending on the
    buckets' batch sizes to be warmed up.
    
    Additionally, removed the situation, where the warmup method is called
    twice (seen as duplicated prints within the warmup phase but with empty
    warmed up buckets, as these have all been already warmed up).
    
    ---------
    
    Signed-off-by: Krzysztof Smusz <ksmusz@habana.ai>

[33mcommit 257b8d860123df6672b3ab3b1a57be4bed0027bf[m
Author: Michal Adamczyk <michal.adamczyk@intel.com>
Date:   Tue Sep 9 13:15:03 2025 +0200

    Experimental support for Unified Attention (#133)
    
    Introduces a new attention backend - Unified Attention to handle both
    prefills and decodes (and potentially in the future mixed batches).
    * To enable run with VLLM_UNIFIED_ATTN=true
    * Unified Attention by default implies contiguous_pa and merged_prefill,
    but one can disable them by specifying their respective flags
    (VLLM_CONTIGOUS_PA=false or VLLM_MERGED_PREFILL=f)
    
    ---------
    
    Signed-off-by: Michal Adamczyk <michal.adamczyk@intel.com>

[33mcommit 60c9a3ed25e70b9dce7d8b22dc36c2d9947f2d9d[m
Author: Agata Dobrzyniewicz <160237065+adobrzyn@users.noreply.github.com>
Date:   Tue Sep 9 10:24:52 2025 +0200

    [Merged Prefill] Warmup for merged prefill (#104)
    
    This PR introduces warmup for merged prefill but also changes warmup
    design a little bit:
    - separate get cfg and get range functions in strategies
    - strategies will not handle filtering buckets now
    - bucketing manager will create buckets from 3 ranges (bs, query, ctx)
    and filter out not wanted buckets based on filtering map
    
    ---------
    
    Signed-off-by: Agata Dobrzyniewicz <adobrzyniewicz@habana.ai>

[33mcommit dea1fcf38177534f072c17da7c02b4c526d3feec[m
Author: Vivek Goel <vivek.goel@intel.com>
Date:   Tue Sep 9 12:54:35 2025 +0530

    Update CODEOWNERS file (#143)
    
    Signed-off-by: Vivek <vgoel@habana.ai>

[33mcommit 5327e512457cdb4fe767b35fbad9cacc8c2145a3[m
Author: Spurthi Lokeshappa <slokeshappa@habana.ai>
Date:   Mon Sep 8 13:41:38 2025 -0700

    Enable embedding feature (#141)
    
    This PR adds support to hpu_model_ruuner to execute pooling models.
    Note : Warm up is not yet enabled for pooling.
    
    ---------
    
    Signed-off-by: slokesha <slokeshappa@habana.ai>

[33mcommit 65869ce5157f9400519f7d6953d86f97482aeacd[m
Author: Kacper Pietkun <kpietkun@habana.ai>
Date:   Mon Sep 8 14:56:20 2025 +0200

    Add tests for custom op registration (#109)
    
    I added two tests for testing custom op registration.
    Additionally, in `vllm_gaudi/ops/__init__.py`, I wrapped imports into a
    function. I did it because currently, if someone imported custom
    operator (before ops registration) for example `from
    vllm_gaudi.ops.hpu_layernorm import HPURMSNorm`, then all other custom
    ops would be register as an unexpected side effect. With that change
    only `HPURMSNorm` will be registered in such case.
    
    ---------
    
    Signed-off-by: Kacper Pietkun <kpietkun@habana.ai>
    Co-authored-by: Michał Kuligowski <michal.kuligowski@intel.com>

[33mcommit aae8e9604268b05cedb85fac72f34420e321cc17[m
Author: Taran Iyengar <taran.iyengar@intel.com>
Date:   Mon Sep 8 03:25:06 2025 -0700

    Fix warmup break when max decode bucket bs > max num seq (#107)
    
    Signed-off-by: taran2210 <taran.iyengar@intel.com>
    Co-authored-by: Michał Kuligowski <michal.kuligowski@intel.com>
    Co-authored-by: Agata Dobrzyniewicz <160237065+adobrzyn@users.noreply.github.com>

[33mcommit 69d4ad3c1814e9afd8acce10bcf7b59b67ed5388[m
Author: Michal Gawarkiewicz <michal.gawarkiewicz@intel.com>
Date:   Mon Sep 8 09:55:29 2025 +0200

    Update CODEOWNERS (#144)

[33mcommit 8a359c7bc9c44904901a5a502275163b2a7dfc4a[m
Author: Agata Dobrzyniewicz <160237065+adobrzyn@users.noreply.github.com>
Date:   Mon Sep 8 08:47:50 2025 +0200

    [Bugfix] Remove reqs without logits - merge prefill case (#137)
    
    Sometimes reqs that don't have returning toks get mixed up with the rest
    prefills in merged prefill case - we want to remove them from sampling
    
    ---------
    
    Signed-off-by: Agata Dobrzyniewicz <adobrzyniewicz@habana.ai>

[33mcommit acc45076123cdc2dcca40a675e578001584e107b[m
Author: Agata Dobrzyniewicz <160237065+adobrzyn@users.noreply.github.com>
Date:   Fri Sep 5 12:51:27 2025 +0200

    Revert "Enable embedding feature" (#140)
    
    Reverts vllm-project/vllm-gaudi#120

[33mcommit 64f2142d040f0c8fecde0fc8464a3e67703e35a5[m
Author: Spurthi Lokeshappa <spurthi.lokeshappa@intel.com>
Date:   Thu Sep 4 14:51:58 2025 -0700

    Enable embedding feature (#120)
    
    This PR adds support to hpu_model_ruuner to execute pooling models.
    
    ---------
    
    Signed-off-by: slokesha <slokeshappa@habana.ai>

[33mcommit f23198aa63739fe57e987f320b23244d429d9ff6[m
Author: PatW <patryk.wolsza@intel.com>
Date:   Thu Sep 4 14:39:28 2025 +0200

    Merging vllm docker implementation to vllm-gaudi (v1) (#125)
    
    Set of commits for the vllm docker
    
    ---------
    
    Signed-off-by: PatrykWo <patryk.wolsza@intel.com>

[33mcommit 4a59ed4fccbf4df8591b97376c770c3cc6a0a673[m
Author: Marcin Swiniarski <marcin.swiniarski@intel.com>
Date:   Thu Sep 4 13:33:38 2025 +0200

    Disable warmup for defragmentator (#132)
    
    Updating states in defragmentator on dummy data is redundant and we
    should avoid it.
    Right now, doing warmup on defragmentator will also cause a crash in
    case of contigious pa due to
    https://github.com/vllm-project/vllm-gaudi/pull/126
    
    Signed-off-by: Marcin Swiniarski <marcin.swiniarski@intel.com>

[33mcommit 353f81eb45a9216d6c937a5cb7dba93dd61a7b74[m
Author: Agata Dobrzyniewicz <160237065+adobrzyn@users.noreply.github.com>
Date:   Wed Sep 3 11:25:47 2025 +0200

    [Bugfix] Warmup with continuous PA (#126)
    
    Signed-off-by: Agata Dobrzyniewicz <adobrzyniewicz@habana.ai>

[33mcommit e8f817f2875ca0d988b203c791c0fcf8a5682b1e[m
Author: Chendi.Xue <chendi.xue@intel.com>
Date:   Wed Sep 3 01:33:44 2025 -0500

    [BUG fix] Fix spec_decode introduced long graph compilation issue (#127)
    
    Smusz, Krzysztof reported performance regression introduced by spec
    decode PR
    Fixed in this PR
    
    Signed-off-by: Chendi.Xue <chendi.xue@intel.com>

[33mcommit e3dd6a67308d75813678e61dbf6b374b83888d9e[m
Author: Konrad Zawora <kzawora@habana.ai>
Date:   Mon Sep 1 12:13:02 2025 +0200

    Remove test_load_model_weights_inplace  (#48)
    
    This test can take 2-3 minutes for initializing copied model runner
    fixture, and is skipped anyway - no point in wasting time for it.
    
    ---------
    
    Signed-off-by: Konrad Zawora <kzawora@habana.ai>

[33mcommit 596d33576259d5a0253da5eb722ea569270d99a0[m
Author: Chendi.Xue <chendi.xue@intel.com>
Date:   Fri Aug 29 17:01:48 2025 -0500

    [FIX HOURLY Failure] transformer 4.56.0 is not compatible with INC (#117)
    
    Signed-off-by: Chendi.Xue <chendi.xue@intel.com>

[33mcommit b3cf2e261407e44b447ad59430fc0f052caeda29[m
Author: Chendi.Xue <chendi.xue@intel.com>
Date:   Fri Aug 29 16:12:17 2025 -0500

    fix qwen3-30B-A3B-FP8 - The number of dims cannot be packed into CompleteArgumentSpec:65535  (#113)
    
    This PR is on top of #112
    
    With the fix, we can run qwen3-30B-A3B-FP8 successfully on G2 and G3 now
    
    ```
    
     VLLM_SKIP_WARMUP=true VLLM_CONTIGUOUS_PA=False PT_HPU_LAZY_MODE=1 lm_eval   --model vllm --tasks gsm8k --num_fewshot 5 --batch_size 128 --model_args "pretrained=Qwen/Qwen3-30B-A3B-FP8,tensor_parallel_size=1,trust_remote_code=true,max_model_len=4096,dtype=bfloat16"
    
    ```
    
    2025-08-29:06:12:48 INFO [loggers.evaluation_tracker:280] Output path
    not provided, skipping saving results aggregated
    vllm
    (pretrained=Qwen/Qwen3-30B-A3B-FP8,tensor_parallel_size=1,trust_remote_code=true,max_model_len=4096,dtype=bfloat16),
    gen_kwargs: (None), limit: None, num_fewshot: 5, batch_size: 128
    |Tasks|Version| Filter |n-shot| Metric | |Value | |Stderr|
    
    |-----|------:|----------------|-----:|-----------|---|-----:|---|-----:|
    |gsm8k| 3|flexible-extract| 5|exact_match|↑ |0.8711|± |0.0092|
    | | |strict-match | 5|exact_match|↑ |0.8908|± |0.0086|
    
    Signed-off-by: Chendi.Xue <chendi.xue@intel.com>

[33mcommit 0fbea5561c919ede0e6179428e257336ef35447b[m
Author: Chendi.Xue <chendi.xue@intel.com>
Date:   Fri Aug 29 12:21:00 2025 -0500

    Enable Spec Decode for HPU v1 - Part1(basic workflow + eagle) (#81)
    
    Design concept:
    
    with spec decode, each decode request might have more than 1 tokens
    * leads to the decode num_tokens range from [batch_size, batch_size *
    (num_decode_tokens+1)]
    * temp solution: always assuming worst case, so we will use the
    max_num_tokens
    * change to HPU_model_runner prepare_decodes_input, all shape will be
    [padded_batch_size, (num_draft_tokens + 1)]
    
    workflow:
    
    ```
    prompt path:
     => [draft_model] => draft_tokens
    
    decode path:
    => (combine input_tokens, draft_tokens) => [prepare_inputs] => (padded_input_tokens) => [target_model]
        => (target_tokens, bonus_tokens) => [reject_sampler]
        => output_tokens (combination of draft_token + bonus_tokens) => [draft_model]
        => (new_draft_tokens, output_tokens) => update to input_batch data structure
    
    # input_tokens: shape is [num_decodes, 1]
    # draft_tokens: shape is [num_decode * num_draft_tokens, 1]
    # combine input_tokens, draft_tokens is with dynamic shape, range from num_decodes to num_decode * (num_draft_tokens+1)
    # padded_input_tokens: shape is [num_decode * (num_draft_tokens+1), 1] => same to positions, slot_mapping, block_groups
    # output_tokens: shape is [num_decodes * (num_drafte_tokens + 1), 1]
    # new_draft_tokens: shape is [num_decode, num_drafte_tokens]
    ```
    
    Design Doc:
    <img width="2113" height="837" alt="image"
    src="https://github.com/user-attachments/assets/fa2e3176-4008-470c-acdf-823da23a38bf"
    />
    
    Jira: SW-234434
    
    Updated on WW35.2:
    
    This PR is working on Eagle and NGRAM at this moment
    For Eagle, only support num_spec_decode_token = 1
    
    ```
    PT_HPU_LAZY_MODE=1 python tests/full_tests/spec_decode.py --task eagle --osl 512
    ```
    ```
    ================ spec_eagle =================
    acc_counts: [3331, 0]
    acc_rate: 0.3735142408611796
    num_draft_tokens: 8918
    num_drafts: 8918
    ---
    Prompt: Hello, my name is
    Generated text:  [Name]. I am a [Your Profession/Student] and I am here to learn more about [Topic/Industry]. I am excited to be a part of this [Event/Community] and I am looking forward to connecting with others who'...'
    ---
    Prompt: The president of the United States is
    Generated text:  the head of state and government of the United States, and is the highest-ranking official in the country. The president is responsible for executing the laws of the United States, and is also the co'...'
    ---
    Prompt: The capital of France is
    Generated text:  Paris, which is located in the north-central part of the country. Paris is the most populous city in France and is known for its stunning architecture, art museums, fashion, and romantic atmosphere. '...'
    ---
    Prompt: The future of AI is
    Generated text:  bright, but it's not without its challenges. Here are some of the key challenges that AI faces in the future:
    1. Explainability and Transparency: As AI systems become more complex and autonomous, it''...'
    ---
    Prompt: San Francisco is know for its
    Generated text:  vibrant arts and culture scene, and the city is home to a wide range of museums, galleries, and performance venues. Here are some of the top arts and culture attractions in San Francisco:
    1. de Young'...'
    ---
    Prompt: Facebook was created in 2004 by
    Generated text:  Mark Zuckerberg, along with his college roommates and fellow Harvard University students Eduardo Saverin, Andrew McCollum, Dustin Moskovitz, and Chris Hughes. Initially, the platform was called "Thef'...'
    ---
    Prompt: Curious George is a
    Generated text:  beloved children's book series created by H.A. and Margret Rey. The series follows the adventures of a curious and mischievous monkey named George, who lives with his friend the Man in the Yellow Hat'...'
    ---
    Prompt: Python 3.11 brings improvements to its
    Generated text:  type hinting system, including support for type hints in lambda functions and improvements to the type checker. Here are some of the key changes:
    
    1. **Type hints in lambda functions**: You can now a'...'
    =========================================
    ```
    
    
    ```
    PT_HPU_LAZY_MODE=1 python tests/full_tests/spec_decode.py --task ngram --osl 512
    ```
    ```
    ================= spec_ngram =================
    acc_counts: [1452, 0]
    acc_rate: 0.18558282208588958
    num_draft_tokens: 7824
    num_drafts: 7824
    ---
    Prompt: Hello, my name is
    Generated text:  Xiaoyu, and I'm a student at the University of Science and Technology of China. I'm currently working on a research project about the application of machine learning in the field of materials science'...'
    ---
    Prompt: The president of the United States is
    Generated text:  the head of state and government of the United States. The president is the head of the executive branch of the U.S. government, and is the commander-in-chief of the United States Armed Forces. The p'...'
    ---
    Prompt: The capital of France is
    Generated text:  Paris. The capital of Germany is Berlin. The capital of Italy is Rome. The capital of Spain is Madrid. The capital of Portugal is Lisbon. The capital of Greece is Athens. The capital of Belgium is Br'...'
    ---
    Prompt: The future of AI is
    Generated text:  a topic that has been the subject of much speculation and debate. As the technology continues to evolve, it's clear that AI is going to have a significant impact on society, the economy, and the way '...'
    ---
    Prompt: San Francisco is know for its
    Generated text:  fog, but the fog is not the only thing that is fog-like. The city is also known for its fog-like "fog" in the form of a fog-like substance that is not actually fog. What is this substance? Also, what'...'
    ---
    Prompt: Facebook was created in 2004 by
    Generated text:  Mark Zuckerberg, and it has grown into a global social media platform with over 2.8 billion monthly active users. The platform allows users to create profiles, connect with friends, share content, an'...'
    ---
    Prompt: Curious George is a
    Generated text:  2015 American 3D computer-animated comedy film directed by Tom McCamus and written by David W. Zucker, and starring the titular character, Curious George, who is a monkey. The film is the first in th'...'
    ---
    Prompt: Python 3.11 brings improvements to its
    Generated text:  standard library, including the `typing` module. One of the notable changes is the introduction of the `TypeAlias` feature, which allows for the creation of type aliases in a more readable and concis'...'
    =========================================
    ```
    
    ---------
    
    Signed-off-by: Chendi.Xue <chendi.xue@intel.com>

[33mcommit 2b85c2d7aa2b05ef4877fe58dee8723c30fd41ef[m
Author: Chendi.Xue <chendi.xue@intel.com>
Date:   Fri Aug 29 09:40:56 2025 -0500

    Port G2 scaling convert from vllm-fork #1505 (#112)
    
    Now both compressed_tensor / blockfp8 gets correct result
    
    * block fp8 model
    ```
    VLLM_SKIP_WARMUP=true VLLM_CONTIGUOUS_PA=False PT_HPU_LAZY_MODE=1 \
    lm_eval   --model vllm --tasks gsm8k --num_fewshot 5 --batch_size 128 \
    --model_args "pretrained=Qwen/Qwen3-8B-FP8,tensor_parallel_size=1,trust_remote_code=true,max_model_len=4096,dtype=bfloat16"
    ```
    
    vllm
    (pretrained=Qwen/Qwen3-8B-FP8,tensor_parallel_size=1,trust_remote_code=true,max_model_len=4096,dtype=bfloat16),
    gen_kwargs: (None), limit: None, num_fewshot: 5, batch_size: 128
    |Tasks|Version| Filter |n-shot| Metric | |Value | |Stderr|
    
    |-----|------:|----------------|-----:|-----------|---|-----:|---|-----:|
    |gsm8k| 3|flexible-extract| 5|exact_match|↑ |0.8772|± |0.0090|
    | | |strict-match | 5|exact_match|↑ |0.8741|± |0.0091|
    
    
    * compressed-tensor
    ```
    VLLM_SKIP_WARMUP=true VLLM_CONTIGUOUS_PA=False PT_HPU_LAZY_MODE=1 \
    lm_eval \
      --model vllm --tasks gsm8k --num_fewshot 5 --batch_size 128 \
      --model_args "pretrained=RedHatAI/Qwen3-8B-FP8-dynamic,tensor_parallel_size=1,trust_remote_code=true,max_model_len=4096,dtype=bfloat16"
    ```
    2025-08-29:05:34:04 INFO [loggers.evaluation_tracker:280] Output path
    not provided, skipping saving results aggregated
    vllm
    (pretrained=RedHatAI/Qwen3-8B-FP8-dynamic,tensor_parallel_size=1,trust_remote_code=true,max_model_len=4096,dtype=bfloat16),
    gen_kwargs: (None), limit: None, num_fewshot: 5, batch_size: 128
    |Tasks|Version| Filter |n-shot| Metric | |Value | |Stderr|
    
    |-----|------:|----------------|-----:|-----------|---|-----:|---|-----:|
    |gsm8k| 3|flexible-extract| 5|exact_match|↑ |0.8810|± |0.0089|
    | | |strict-match | 5|exact_match|↑ |0.8757|± |0.0091|
    
    Signed-off-by: Chendi.Xue <chendi.xue@intel.com>

[33mcommit a6a4ce9ef3916e3153345f5308c9dc03146422c7[m
Author: Chendi.Xue <chendi.xue@intel.com>
Date:   Thu Aug 28 20:20:02 2025 -0500

    fix an argument issue introduced by recent vllm upstream and add CI (#111)
    
    Signed-off-by: Chendi.Xue <chendi.xue@intel.com>

[33mcommit 9a17bda5a739d71217d83c68441f5151f538e56f[m
Author: Chendi.Xue <chendi.xue@intel.com>
Date:   Thu Aug 28 11:49:53 2025 -0500

    Fix the failing introduced by upstream 22685 (#110)
    
    root cause:
    upstream added a new get_compressed_expert_map which used dynamic shape.
    => monkeypatch with different impl
    
    Now, bring the QWEN3-30B EP=2 test back
    
    Signed-off-by: Chendi.Xue <chendi.xue@intel.com>

[33mcommit 9a958379e2c0aac52d5c2bd72f2b8b02ff3b6ef7[m
Author: Chendi.Xue <chendi.xue@intel.com>
Date:   Wed Aug 27 16:37:29 2025 -0500

    fix upstream PR 23749 (#108)
    
    Signed-off-by: Chendi.Xue <chendi.xue@intel.com>

[33mcommit 97c5be06c0196fc64140b207f21a9e3e134d05be[m
Author: Kamil Kaczor <kamil.kaczor@intel.com>
Date:   Wed Aug 27 11:33:29 2025 +0200

    Fix decode profiling (#106)
    
    Signed-off-by: Kamil Kaczor <kamil.kaczor@intel.com>

[33mcommit 676360ebc88e7c866ab8f4314738c4d8fb0e272a[m
Author: Thomas Atta-Fosu <thomas.atta-fosu@intel.com>
Date:   Tue Aug 26 16:00:41 2025 -0700

    Fix mm encoder inputs for mix-modalities in input batch (#103)
    
    - Multimodal encoder inputs are now hashed (from upstream
    [#22711](https://github.com/vllm-project/vllm/pull/22711))
    - `SchedulerOutputs`' member `scheduled_encoder_inputs` now has only
    (representative) `req_ids` with inputs that are not hashed yet.
    - This PR fixes a scenario where `req_ids` to be processed by
    `_execute_mm_encoder` contains requests ids whose encoder inputs are
    previously hashed, and thus will throw `KeyError` exception for keys in
    `req_ids - scheduled_encoder_inputs.keys()`
    - This is common with mixed modality inputs e.g the input sequences
    passed to `generate` has some prompts associated with an image, while
    others are for a video)
    
    Signed-off-by: attafosu <thomas.atta-fosu@intel.com>

[33mcommit 2d5352e23e303c5bc0f97250f8b01d2f8b9acf36[m
Author: Marcin Swiniarski <marcin.swiniarski@intel.com>
Date:   Tue Aug 26 15:31:44 2025 +0200

    Avoid copying dynamic slice of sampling_metadata tensors (#88)
    
    Currently when batch has changed, we are refreshing sampling metadata on
    device by copying only a slice of data.
    This solution works good on GPU, but has negative impact on HPU by
    causing recompilation for each unique slice.
    Since we are copying here only a tensor of shape [batch_size], it is
    better to avoid recompilation in cost of copying little more data.
    
    Other solution could be to warmup each possible test case, but in this
    case we would have to warmup copy of factorial of each batch_size bucket
    (batch_size!), which doesn't seem like a good idea.
    
    ---------
    
    Signed-off-by: Marcin Swiniarski <marcin.swiniarski@intel.com>

[33mcommit b2e8a7da7a7afc7d836b8e280a4448d7f8dc8979[m
Author: Kamil Kaczor <kkaczor@habana.ai>
Date:   Tue Aug 26 14:00:04 2025 +0200

    Add sampler unit tests (#99)
    
    Add tests of:
    - greedy sampling
    - random sampling
    - random sampling seeded
    - random sampling + top_p/top_k/min_p
    
    ---------
    
    Signed-off-by: Kamil Kaczor <kamil.kaczor@intel.com>

[33mcommit f331b00f29383c2d3dd47f715a2877a896080a39[m
Author: Agata Dobrzyniewicz <160237065+adobrzyn@users.noreply.github.com>
Date:   Tue Aug 26 09:34:28 2025 +0200

    [UT] Fix test args for bucketing tests (#105)
    
    Signed-off-by: Agata Dobrzyniewicz <adobrzyniewicz@habana.ai>

[33mcommit c204d321e06a4248113bd8a71d5019ec78030b3a[m
Author: Agata Dobrzyniewicz <160237065+adobrzyn@users.noreply.github.com>
Date:   Mon Aug 25 14:33:18 2025 +0200

    Warmup fix - for non contiguous PA runs, don't take more context blocks than possible (#97)
    
    Signed-off-by: Agata Dobrzyniewicz <adobrzyniewicz@habana.ai>

[33mcommit fe7cd43b578c32bca6be106c547fd287cc296110[m
Author: Andrzej Kotłowski <Andrzej.Kotlowski@intel.com>
Date:   Mon Aug 25 14:05:36 2025 +0200

    Reduce number of compilations when dynamic shapes is used (#90)
    
    It fixes the issue with to many compilation for Pytorch dynamic shapes (
    when VLLM_T_COMPILE_DYNAMIC_SHAPES=1)
    It allows making dynamic shapes for registered buffers (see
    UnspecializedParamBufferSource in PyTorch) by setting dynamo config.
    It also enables dynamic_shapes_compilation by default.
    
    ---------
    
    Signed-off-by: Andrzej Kotłowski <akotlowski@habana.ai>

[33mcommit 50f7d8b7eec32021ab2464539a335f479f6b36d9[m
Author: Agata Dobrzyniewicz <160237065+adobrzyn@users.noreply.github.com>
Date:   Mon Aug 25 13:25:59 2025 +0200

    [Upstream fix] Fix after #22711 (#102)
    
    Culprit PR: https://github.com/vllm-project/vllm/pull/22711
    
    ---------
    
    Signed-off-by: Agata Dobrzyniewicz <adobrzyniewicz@habana.ai>

[33mcommit d695fa81736933233e73b31f190e2cafec344855[m
Author: Michal Adamczyk <michal.adamczyk@intel.com>
Date:   Mon Aug 25 08:49:50 2025 +0200

    Port defragmentation support from vllm-fork PR #1568 (#94)
    
    Signed-off-by: Michal Adamczyk <michal.adamczyk@intel.com>

[33mcommit bfbad711f35401c7831a9c867e6a4ff9677ee069[m
Author: Marcin Swiniarski <marcin.swiniarski@intel.com>
Date:   Fri Aug 22 11:45:23 2025 +0200

    Fix upstream PR 22668 that added additional arg to is_kv_cache_dtype_supported (#96)
    
    Fixes https://github.com/vllm-project/vllm/pull/22668 - we need to take
    one more arg.
    
    Signed-off-by: Marcin Swiniarski <mswiniarski@habana.ai>

[33mcommit b8217f69b81fb8e8aff0188c8c0e33cca8fc55e1[m
Author: Thomas Atta-Fosu <thomas.atta-fosu@intel.com>
Date:   Thu Aug 21 16:54:12 2025 -0700

    Enable multimodal support + qwen2.5-vl (#92)
    
    - Enables v1 multmodal support
    - Enables qwen2.5-vl: Support for MRope
    
    ---------
    
    Signed-off-by: attafosu <thomas.atta-fosu@intel.com>

[33mcommit a21cbc61cc3e6891236c0caac832cdfbcbd86ca5[m
Author: Agata Dobrzyniewicz <160237065+adobrzyn@users.noreply.github.com>
Date:   Thu Aug 21 16:59:57 2025 +0200

    [Upstream fix] Fix after #23262 from upstream - Make new_block_ids None if empty (#93)
    
    Culprit commit: https://github.com/vllm-project/vllm/pull/23262
    
    ---------
    
    Signed-off-by: Agata Dobrzyniewicz <adobrzyniewicz@habana.ai>

[33mcommit 5ff54ed966ce1a4c834e1da1a88b705d48da82bc[m
Author: Chendi.Xue <chendi.xue@intel.com>
Date:   Thu Aug 21 02:50:08 2025 -0500

    Fix jenkins - remove failed test and fix later / update API (#79)
    
    See jenkins is failing with all PRs, provide a quick fix here:
    
    1. remove failed test
        1. llama4 modeling is updated from upstream, failed on all CI
        2. Blockfp8 + Qwen3 show accuracy as zero, need to root cause
        3. llama4 vision is not enabled yet, skip CI
    2. update vllm engine_args
    1. upstream removed 'num_scheduler_steps' args -> update in
    .jenkins/lm-eval-harness/test_lm_eval_correctness.py accordingly
    2. upstream removed 'weights_load_device' args -> update in
    .jenkins/lm-eval-harness/test_lm_eval_correctness.py accordingly
    
    ---------
    
    Signed-off-by: Chendi.Xue <chendi.xue@intel.com>

[33mcommit efdf1d7fc70dc293a04a9cff9a138fd6be5326b4[m
Author: Chendi.Xue <chendi.xue@intel.com>
Date:   Wed Aug 20 13:00:36 2025 -0500

    remove enable_prompt_adapter in test to fix (#91)
    
    TypeError: EngineArgs.__init__() got an unexpected keyword argument
    'enable_prompt_adapter'
    
    Signed-off-by: Chendi.Xue <chendi.xue@intel.com>

[33mcommit 8400f8204a4265bf6cd58832935c47c9cd531454[m
Author: Agata Dobrzyniewicz <160237065+adobrzyn@users.noreply.github.com>
Date:   Wed Aug 20 12:06:55 2025 +0200

    Change warmup scenario for execute dummy scenario (#54)
    
    Change warmup scenario to execute dummy scenario. This way we more
    accurately simulate the real behaviour of vllm inference by executing
    the precise run that is happening in real inference but with dummy
    config that we want to warm-up during warm-up process. No need for some
    artificially create an inference scenario, as right now we are utilizing
    real execution flow
    
    ---------
    
    Signed-off-by: Agata Dobrzyniewicz <adobrzyniewicz@habana.ai>

[33mcommit 0492c5598089fe29314ceb2aeaa4ae1f67eebcb5[m
Author: Agata Dobrzyniewicz <160237065+adobrzyn@users.noreply.github.com>
Date:   Tue Aug 19 14:17:00 2025 +0200

    [Upstream fix] Fix after #23041 from upstream (#87)
    
    Culprit PR: https://github.com/vllm-project/vllm/pull/23041
    
    Signed-off-by: Agata Dobrzyniewicz <adobrzyniewicz@habana.ai>

[33mcommit 77c704f9657db1dcacbeb100ae7492e112103611[m
Author: Chendi.Xue <chendi.xue@intel.com>
Date:   Mon Aug 18 16:00:07 2025 -0500

    add commit-id to distinguish image and container for each PR (#85)
    
    Current CI does not distinguish PR during image build and container
    launch
    
    If multiple CI triggered, it will overwrite other PR's docker image.
    
    In this commit, I propose to use difference name for each PR
    docker/container.
    
    Signed-off-by: Chendi.Xue <chendi.xue@intel.com>

[33mcommit ab65f9ba2abbaf4c30f8cdb24a62c731f8bbdf4c[m[33m ([m[1;33mtag: [m[1;33mv0.10.1[m[33m, [m[1;31morigin/v0.10.1_next[m[33m)[m
Author: Kacper Pietkun <kpietkun@habana.ai>
Date:   Mon Aug 18 14:08:50 2025 +0200

    Add t.compile config (#62)
    
    Signed-off-by: Kacper Pietkun <kpietkun@habana.ai>

[33mcommit 39bce5d0437888619baefa4d3c384c66bc5da31e[m
Author: Chendi.Xue <chendi.xue@intel.com>
Date:   Sun Aug 17 15:30:20 2025 -0500

    Fix logitsProcessor change in input_batch introduced by upstream pr19912 (#84)
    
    Signed-off-by: Chendi.Xue <chendi.xue@intel.com>

[33mcommit a9accecb431b72f524fdfc2731e0457354731539[m
Author: Chendi.Xue <chendi.xue@intel.com>
Date:   Fri Aug 15 15:32:34 2025 -0500

    Fix crash caused by #20059 (#82)
    
    https://github.com/vllm-project/vllm/pull/20059, Will assume cudagraph
    is on
    => Solution: use this PR is diable
    
    Signed-off-by: Chendi.Xue <chendi.xue@intel.com>

[33mcommit 3e0c034b6a7fb2b8ceb99a6ece7fb2d56870c07f[m
Author: Chendi.Xue <chendi.xue@intel.com>
Date:   Wed Aug 13 19:17:17 2025 -0500

    enable awq/gptq based on PR56 (#78)
    
    Origin PR is at https://github.com/vllm-project/vllm-gaudi/pull/56
    
    Signed-off-by: maktukmak <mehmet.aktukmak@intel.com>
    Signed-off-by: Chendi.Xue <chendi.xue@intel.com>

[33mcommit f3a006835c783ef045836748c44086999354d507[m
Author: Tianmu Li <tianmu.li@intel.com>
Date:   Wed Aug 13 16:21:38 2025 -0700

    Enabled structured output (#68)
    
    1. Combine logits from decode and prompt
    2. Move logits to cpu
    3. Apply bitmask on cpu logits
    4. Move logits back to hpu
    All of these are only triggered when doing structured output
    
    ---------
    
    Signed-off-by: Tianmu Li <tianmu.li@intel.com>

[33mcommit 5c4e3a96c161ab75f684aa51ecb691748e4b82b5[m
Author: Chendi.Xue <chendi.xue@intel.com>
Date:   Wed Aug 13 15:33:38 2025 -0500

    Fix crash caused by upstream update PR22570 (#77)
    
    Hourly check crashed due to upstream PR 22570
    
    https://github.com/vllm-project/vllm-gaudi/actions/runs/16943144469/job/48017289064
    
    Fixed in this PR
    
    Signed-off-by: Chendi.Xue <chendi.xue@intel.com>

[33mcommit 15b173a99dd0b1eacf7316b57f7720a0fd89e9de[m
Author: Yi Liu <yi4.liu@intel.com>
Date:   Wed Aug 13 09:15:13 2025 +0800

    Add INC dynamic quant test for deepseek-v2 (#67)
    
    Add INC dynamic quant test for deepseek-v2 + tp2
    
    ---------
    
    Signed-off-by: yiliu30 <yi4.liu@intel.com>

[33mcommit 988b326de380232b77cf9af12d46808b58a20014[m
Author: Chendi.Xue <chendi.xue@intel.com>
Date:   Tue Aug 12 19:57:51 2025 -0500

    Fix error introduced by upstream PR 22714 (#76)
    
    Upstream 007dd90859cc - Yongye Zhu, 2 hours ago : [gpt-oss] Enable
    gpt-oss on ampere (#22714) changed the
    current_platform.get_attn_backend_cls API, update in this PR
    
    Signed-off-by: Chendi.Xue <chendi.xue@intel.com>

[33mcommit 89e0fe748db466fb0dc75c90b982fb5b5a3fc64c[m
Author: Jacek Czaja <jacek.czaja@gmail.com>
Date:   Tue Aug 12 10:50:59 2025 +0200

    Adjusted vllm profiler to recent changes in vllm structures (#73)
    
    This PR is adjusting interface changes of vllm profiler to updated
    changes in RequestCachedData and NewRequestData from vllm project.
    
    Without this adjustment there was a crash when asking for
    profiling(VLLM_PROFILE_DECODE=124,1024 PT_HPU_LAZY_MODE=0
    ./run_benchmark_throughput.sh):
    <img width="1570" height="976" alt="image"
    src="https://github.com/user-attachments/assets/1a8c88fa-4c3c-4b7c-9093-d72e3dfbed00"
    />
    
    Dumped traces were inspected and they seem (with this fix) contain valid
    information
    
    Signed-off-by: Jacek Czaja <jacek.czaja@intel.com>

[33mcommit a97b3a654404fcaec069a9ce5314fd1c44a042b1[m
Author: Chendi.Xue <chendi.xue@intel.com>
Date:   Thu Aug 7 21:08:03 2025 -0500

    Revert "Add INC dynamic quant test" (#66)
    
    Reverts vllm-project/vllm-gaudi#65

[33mcommit a386a6c135b0ea42ddf7b4038efe5d0ddc3184f4[m
Author: Yi Liu <yi4.liu@intel.com>
Date:   Fri Aug 8 09:34:14 2025 +0800

    Add INC dynamic quant test (#65)
    
    Add INC dynamic quant test for deepseek-v2.
    
    cc @hshen14 @thuang6
    
    Signed-off-by: yiliu30 <yi4.liu@intel.com>

[33mcommit 746cd72885f6a8084418ed0271b43f65044ffcc7[m
Author: Konrad Zawora <kzawora@habana.ai>
Date:   Thu Aug 7 16:11:21 2025 +0200

    Reduce CI acc tests to 250 GSM8K samples (#60)
    
    Signed-off-by: Konrad Zawora <kzawora@habana.ai>

[33mcommit ee2156ae22315c009e69af1786c6e2a5cd88a21e[m
Author: Chendi.Xue <chendi.xue@intel.com>
Date:   Wed Aug 6 12:49:13 2025 -0500

    update CI docker image id (#63)
    
    Signed-off-by: Chendi.Xue <chendi.xue@intel.com>

[33mcommit 9343f3513d8d7eadf35c192842c05857e52ae650[m
Author: Agata Dobrzyniewicz <160237065+adobrzyn@users.noreply.github.com>
Date:   Tue Aug 5 18:13:01 2025 +0200

    Port #301 and #313 from extension (#55)
    
    https://github.com/HabanaAI/vllm-hpu-extension/pull/301
    and
    https://github.com/HabanaAI/vllm-hpu-extension/pull/313
    
    Signed-off-by: Agata Dobrzyniewicz <adobrzyniewicz@habana.ai>

[33mcommit 079e659c5c44f236f5b0a28e49453309211fce7a[m
Author: Kacper Pietkun <kacper.pietkun00@gmail.com>
Date:   Tue Aug 5 09:17:39 2025 +0200

    Change PT_HPU_LAZY_MODE default value (#58)
    
    Signed-off-by: Kacper Pietkun <kpietkun@habana.ai>

[33mcommit 86a8acef15b5e733af4c2632a62cbc119eb86d44[m
Author: Agata Dobrzyniewicz <160237065+adobrzyn@users.noreply.github.com>
Date:   Wed Jul 30 11:06:20 2025 +0200

    Port: #282 from extension (#52)
    
    https://github.com/HabanaAI/vllm-hpu-extension/pull/282
    
    Signed-off-by: Agata Dobrzyniewicz <adobrzyniewicz@habana.ai>

[33mcommit b50b3b0ffaf61457c44c87252fe8b30a87efad97[m
Author: Konrad Zawora <kzawora@habana.ai>
Date:   Tue Jul 29 13:21:22 2025 +0200

    Enable high level profiler (#49)
    
    Ripped from https://github.com/HabanaAI/vllm-fork/pull/1501
    
    ---------
    
    Signed-off-by: Konrad Zawora <kzawora@habana.ai>

[33mcommit c9c266ef6dbb48853051a80083fa713464909e5c[m
Author: Konrad Zawora <kzawora@habana.ai>
Date:   Mon Jul 28 14:06:34 2025 +0200

    [CI] Report 10 longest unit tests above 1s (#47)
    
    Signed-off-by: Konrad Zawora <kzawora@habana.ai>

[33mcommit 3222c738ff6c32411f7dea04467cfbe209490cd1[m
Author: Konrad Zawora <kzawora@habana.ai>
Date:   Mon Jul 28 12:13:27 2025 +0200

     Add HPU model runner & HPU input batch unit tests (#44)
    
    Signed-off-by: Konrad Zawora <kzawora@habana.ai>

[33mcommit dfcbfb7d57d37952faa4dfac82bb885f3c4f7926[m
Author: Konrad Zawora <kzawora@habana.ai>
Date:   Mon Jul 28 11:13:11 2025 +0200

    Add CODEOWNERS (#45)
    
    Signed-off-by: Konrad Zawora <kzawora@habana.ai>

[33mcommit 7c75f506a889d7407abf68094e1a36a98b3937a9[m
Author: Konrad Zawora <kzawora@habana.ai>
Date:   Fri Jul 25 15:31:28 2025 +0200

    Fix API mismatch after PR 21585 (#43)
    
    Mirroring changes from https://github.com/vllm-project/vllm/pull/21585
    to HPU code
    
    Signed-off-by: Konrad Zawora <kzawora@habana.ai>

[33mcommit 0cc8bb6e2635aec21d4501359738e520dc4df32d[m
Author: Chendi.Xue <chendi.xue@intel.com>
Date:   Wed Jul 23 23:29:42 2025 -0500

    [CI] update lm_eval with lastet version for vllm args update (#42)
    
    Bring back lm_eval CI by
    1. use lm_eval latest version because of vllm API update -
    https://github.com/EleutherAI/lm-evaluation-harness/pull/3176
    2. update test_common.py to work with latest lm_eval
    
    Signed-off-by: Chendi.Xue <chendi.xue@intel.com>

[33mcommit f1d3f04c63b7693a72ed0bd8bdced2a6e06d16bc[m
Author: Chendi.Xue <chendi.xue@intel.com>
Date:   Wed Jul 23 22:11:43 2025 -0500

    [FIX_DUE_UPSTREAM]fix for upstream PR20588 (#41)
    
    https://github.com/vllm-project/vllm/pull/20588
    
    ---------
    
    Signed-off-by: Chendi.Xue <chendi.xue@intel.com>

[33mcommit 55cb5aab7d006c8ffbaea8e3de4b5580d1044345[m
Author: Konrad Zawora <kzawora@habana.ai>
Date:   Tue Jul 22 12:59:29 2025 +0200

    Restore support for kv_cache_dtype  (#40)
    
    https://github.com/vllm-project/vllm/pull/21302 got merged, we can
    re-enable kv_cache_dtype now.
    
    ---------
    
    Signed-off-by: Konrad Zawora <kzawora@habana.ai>

[33mcommit c8948623dddfaf74c9596d86abd64827443a9bfe[m
Author: Konrad Zawora <kzawora@habana.ai>
Date:   Mon Jul 21 18:31:32 2025 +0200

    [CI] Don't fetch PyTorch for building vLLM upstream (#39)
    
    This PR shortens the CI execution time significantly by using
    preinstalled PyTorch during vLLM build.
    
    ---------
    
    Signed-off-by: Konrad Zawora <kzawora@habana.ai>

[33mcommit fcbe490d84efa313f74b4d927e4d50d400697135[m
Author: Konrad Zawora <kzawora@habana.ai>
Date:   Mon Jul 21 17:40:12 2025 +0200

    Fix attention API post blocksparse deprecation (#38)
    
    Upstream PR https://github.com/vllm-project/vllm/pull/21217 changed
    attention APIs. This PR adjusts our attention implementation to the new
    API.
    
    ---------
    
    Signed-off-by: Konrad Zawora <kzawora@habana.ai>

[33mcommit 6952fef73fbbdbccd8681de87d913d03fe600307[m
Author: Chendi.Xue <chendi.xue@intel.com>
Date:   Thu Jul 17 16:35:08 2025 -0500

    [CI]v0 hpu deprecated, update CI scripts (#37)
    
    Signed-off-by: Chendi.Xue <chendi.xue@intel.com>

[33mcommit eafe5b55ceee72fc9a77b3b2e3b81c5d99ee27d7[m
Author: Konrad Zawora <kzawora@habana.ai>
Date:   Thu Jul 17 15:27:30 2025 +0200

    Add Getting Started section to README (#36)
    
    Signed-off-by: Konrad Zawora <kzawora@habana.ai>

[33mcommit d1c028364e8114deba2a1f5358d925c439050490[m
Author: Konrad Zawora <kzawora@habana.ai>
Date:   Thu Jul 17 10:58:03 2025 +0200

    Fix bs=2 prefill bucketing weirdness  (#35)
    
    ripped from: https://github.com/HabanaAI/vllm-fork/pull/1606, fixes
    weird bucketing anomaly where bs=1 prefills would be padded to bs=2 and
    trigger a recompilation
    
    Signed-off-by: Konrad Zawora <kzawora@habana.ai>

[33mcommit acac4cf0d0796b3f4bc3f7b578f2c33b49cf3859[m
Author: Chendi.Xue <chendi.xue@intel.com>
Date:   Wed Jul 16 17:22:00 2025 -0500

    [CI]update CI version to 1.22-526 and add INC unit_scale test in (#33)
    
    Signed-off-by: Chendi.Xue <chendi.xue@intel.com>

[33mcommit 719ccc4cc7ecd4f769092bdf4432aca5ebe96253[m
Author: Uri Livne <ulivne@habana.ai>
Date:   Wed Jul 16 21:57:25 2025 +0300

    Add fp8 tests and protect from invalid quant config (#28)
    
    Signed-off-by: Uri Livne <ulivne@habana.ai>
    Co-authored-by: Chendi.Xue <chendi.xue@intel.com>

[33mcommit 219c1ab7c9d11a005398d6095f87d2ddb353db10[m
Author: Konrad Zawora <kzawora@habana.ai>
Date:   Wed Jul 16 17:36:02 2025 +0200

    Fix warmup num_blocks (#31)
    
    Ripped from https://github.com/HabanaAI/vllm-fork/pull/1594/
    
    ---------
    
    Signed-off-by: Konrad Zawora <kzawora@habana.ai>

[33mcommit d1d9bbce2ff3c5d9d055792a669f7bfe1ff998de[m
Author: Konrad Zawora <kzawora@habana.ai>
Date:   Wed Jul 16 14:55:57 2025 +0200

    Fix non-prefix prefill warmup (#30)
    
    This PR fixes warmup of zero-context prefills (ripped from
    https://github.com/HabanaAI/vllm-hpu-extension/pull/293)
    
    Signed-off-by: Konrad Zawora <kzawora@habana.ai>

[33mcommit 664c4605e351769daadc7cdbf464e9865bdc8fd9[m
Author: Konrad Zawora <kzawora@habana.ai>
Date:   Wed Jul 16 14:31:06 2025 +0200

    Use vllm_gaudi.extension logger (#29)
    
    Using default vLLM logger results in logging messages being suppressed
    by default. This PR switches to vllm_gaudi.extension.logger, which wraps
    vLLM logger if available (without suppressing messages), or uses
    non-vLLM logger.
    
    ---------
    
    Signed-off-by: Konrad Zawora <kzawora@habana.ai>

[33mcommit 8ba1c8fdbc5f5a3900e511f448d5cf2672081943[m
Author: Konrad Zawora <kzawora@habana.ai>
Date:   Mon Jul 14 17:06:41 2025 +0200

    Fix bind_kv_cache import moved in #20900 (#26)
    
    Signed-off-by: Konrad Zawora <kzawora@habana.ai>

[33mcommit 8d635af7b245af33b3cff0052d47e71e350cd446[m
Author: Chendi.Xue <chendi.xue@intel.com>
Date:   Sat Jul 12 12:53:31 2025 -0500

    Fix enable_eplb missing arguments and add fp8 llama4 and qwen3 to CI (#25)
    
    Signed-off-by: Chendi.Xue <chendi.xue@intel.com>

[33mcommit 0613e6f87e3e21626f8898baf8df2fc766f4d699[m
Author: Chendi.Xue <chendi.xue@intel.com>
Date:   Fri Jul 11 15:55:41 2025 -0500

    split hourly and pre-merge ci (#24)
    
    Signed-off-by: Chendi.Xue <chendi.xue@intel.com>

[33mcommit ba07bf1fece173ecdf474906ec7cbc63e50b3443[m
Author: Chendi.Xue <chendi.xue@intel.com>
Date:   Fri Jul 11 15:27:58 2025 -0500

    Add gsm8k test to full_tests (#23)
    
    1. granite8b -> used to check HPUattn + MLP
    2. deepseek-v2-lite -> used to check MLA + MOE
    3. qwen3-MOE -> used to check HPUattn + MOE + ExpertParallel
    
    ---------
    
    Signed-off-by: Chendi.Xue <chendi.xue@intel.com>

[33mcommit 256e53bd40117ad61688fd67f41bb34bbb5a636d[m
Author: Konrad Zawora <kzawora@habana.ai>
Date:   Fri Jul 11 19:14:08 2025 +0200

    Port code from latest habana_main (10 July) (#10)
    
    Signed-off-by: Konrad Zawora <kzawora@habana.ai>
    Signed-off-by: Chendi.Xue <chendi.xue@intel.com>
    Co-authored-by: Chendi.Xue <chendi.xue@intel.com>

[33mcommit 980934d9fce09cff6ca8a9124b19a24dabd500ce[m
Author: Konrad Zawora <kzawora@habana.ai>
Date:   Fri Jul 11 16:13:19 2025 +0200

    [CI] Add models-medium stage (#21)
    
    Signed-off-by: Konrad Zawora <kzawora@habana.ai>

[33mcommit 650ce373e00145932aa966c6e160943467f251ce[m
Author: Konrad Zawora <kzawora@habana.ai>
Date:   Fri Jul 11 13:11:56 2025 +0200

    Refactor vllm_hpu -> vllm_gaudi (#2)
    
    Signed-off-by: Konrad Zawora <kzawora@habana.ai>

[33mcommit ee7d75edf4b2e56d541e7bd5f7db4a26fd856c9f[m
Author: Konrad Zawora <kzawora@habana.ai>
Date:   Fri Jul 11 13:07:37 2025 +0200

    Cancel stale pre-commit GHA jobs (#20)

[33mcommit 9882506cb2aa349f19ed3507c9e6c13b5d040c42[m
Author: Konrad Zawora <kzawora@habana.ai>
Date:   Fri Jul 11 11:55:04 2025 +0200

    Add pre-commit GHA (#1)
    
    This PR enables common pre-commit checks (ruff, yapf, mypy) and ensures
    that these checks pass.
    
    ---------
    
    Signed-off-by: Konrad Zawora <kzawora@habana.ai>

[33mcommit 54b24852e60f3e37e97c8294a49de9416db8d460[m
Author: Chendi.Xue <chendi.xue@intel.com>
Date:   Thu Jul 10 23:47:11 2025 -0500

    move hf_cache from _work to /mnt (#19)
    
    Signed-off-by: Chendi.Xue <chendi.xue@intel.com>

[33mcommit 2233bf9c806f32b587ed018d298cf8b4be5d4db6[m
Author: Chendi.Xue <chendi.xue@intel.com>
Date:   Thu Jul 10 23:35:15 2025 -0500

    update pre-merge script to use current workspace (#18)
    
    current script is not installing from wip vllm-gaudi code space. => Fix
    that
    Also rename the docker name to avoid conflict
    
    ---------
    
    Signed-off-by: Chendi.Xue <chendi.xue@intel.com>

[33mcommit 6f8619dc0ee5f86bbb6e6d34ee86870ec1396212[m
Author: Chendi.Xue <chendi.xue@intel.com>
Date:   Thu Jul 10 21:14:29 2025 -0500

    Fix deepseek v2 failing + update CI with early exit when running into failing (#17)
    
    Signed-off-by: Chendi.Xue <chendi.xue@intel.com>

[33mcommit 5032b7b42bfe9d1fc522b22f802a93f5a4a37f0d[m
Author: Konrad Zawora <kzawora@habana.ai>
Date:   Thu Jul 10 19:46:12 2025 +0200

    Auto-cancel stale GHA pre-merge jobs (#16)
    
    Signed-off-by: Konrad Zawora <kzawora@habana.ai>

[33mcommit 1e79b15027735fcecdf285df61926a766e5de288[m
Author: Konrad Zawora <kzawora@habana.ai>
Date:   Thu Jul 10 17:44:21 2025 +0200

    Add Llama3.1-405B jenkins config (#15)
    
    requires https://github.com/vllm-project/vllm-gaudi/pull/14
    
    ---------
    
    Signed-off-by: Konrad Zawora <kzawora@habana.ai>

[33mcommit 9b24a6fa4349cba49e56360732dae79d81e945c2[m
Author: Konrad Zawora <kzawora@habana.ai>
Date:   Thu Jul 10 17:43:16 2025 +0200

    Add synchronized weight loader hack for Llama 405B OOMs (#14)
    
    Signed-off-by: Konrad Zawora <kzawora@habana.ai>

[33mcommit ee1b61747dc1633892f6d35d645b88c765279d79[m
Author: Konrad Zawora <kzawora@habana.ai>
Date:   Thu Jul 10 16:18:06 2025 +0200

    Add test config for jenkins CI (#11)
    
    Signed-off-by: Konrad Zawora <kzawora@habana.ai>

[33mcommit ea29f05dcea2c940d7c776e5e2b3febbaeda22d6[m
Author: Konrad Zawora <kzawora@habana.ai>
Date:   Thu Jul 10 15:27:48 2025 +0200

    Untrack and ignore setuptools_scm-generated _version.py (#13)
    
    Signed-off-by: Konrad Zawora <kzawora@habana.ai>

[33mcommit 8e89441e299fd8371b5ba5bd61e8ba2d057030d6[m
Author: Konrad Zawora <kzawora@habana.ai>
Date:   Thu Jul 10 15:22:00 2025 +0200

    Uniproc executor segfault workaround (#12)
    
    Signed-off-by: Konrad Zawora <kzawora@habana.ai>

[33mcommit 4691f5bcfbbd0972c0f85907fbe462e8f451f94b[m
Author: Konrad Zawora <kzawora@habana.ai>
Date:   Wed Jul 9 12:24:19 2025 +0200

    Add HPU CI tests (#9)
    
    * Add HPU CI tests
    
    Signed-off-by: Konrad Zawora <kzawora@habana.ai>
    
    * oopsie wrong definition
    
    Signed-off-by: Konrad Zawora <kzawora@habana.ai>
    
    ---------
    
    Signed-off-by: Konrad Zawora <kzawora@habana.ai>

[33mcommit 83f99c54ceb02e566a5989baadfb4c222b99be85[m
Author: Chendi.Xue <chendi.xue@intel.com>
Date:   Thu Jul 3 21:11:22 2025 -0500

    [CI] update CI script (#8)
    
    * update CI script
    
    Signed-off-by: Chendi Xue <chendi.xue@intel.com>
    
    * clean up
    
    Signed-off-by: Chendi Xue <chendi.xue@intel.com>
    
    ---------
    
    Signed-off-by: Chendi Xue <chendi.xue@intel.com>

[33mcommit a4646becd590994bbbf60346f51b3e75e55938b2[m
Author: Chendi.Xue <chendi.xue@intel.com>
Date:   Thu Jul 3 17:30:08 2025 -0500

    Add hourly CI (#7)
    
    Signed-off-by: Chendi Xue <chendi.xue@intel.com>

[33mcommit 1755fdb4fc7a414f5cc40799527bf116083d06cd[m
Author: Chendi.Xue <chendi.xue@intel.com>
Date:   Thu Jul 3 14:50:23 2025 -0500

    Fix failing due to  (#16728) (#5)
    
    Signed-off-by: Chendi Xue <chendi.xue@intel.com>

[33mcommit f75ff7b9fceb99f91ebbd3d522ff9fd4fb4b876f[m
Author: Chendi.Xue <chendi.xue@intel.com>
Date:   Thu Jul 3 13:57:57 2025 -0500

    Fix CI fail hang (#6)
    
    Signed-off-by: Chendi Xue <chendi.xue@intel.com>

[33mcommit 35d46d0e8e1cb42bf62f0c4d204978abec83bbba[m
Author: Chendi.Xue <chendi.xue@intel.com>
Date:   Thu Jul 3 11:59:49 2025 -0500

    Fix hpu_model_runner based on (#20291) (#4)
    
    Signed-off-by: Chendi Xue <chendi.xue@intel.com>

[33mcommit d031d03b6e7c8a1c0335d902e4c86b98cd2da650[m
Author: Chendi.Xue <chendi.xue@intel.com>
Date:   Mon Jun 30 17:45:09 2025 -0500

    [FIX for upstream changes ]hpu_model_runner and add UT (#3)
    
    * Fix hpu_model_runner due to PR  (#20232)
    
    Signed-off-by: Chendi.Xue <chendi.xue@intel.com>
    
    * add UT in plugin and will be used by upstream test
    
    Signed-off-by: Chendi.Xue <chendi.xue@intel.com>
    
    ---------
    
    Signed-off-by: Chendi.Xue <chendi.xue@intel.com>

[33mcommit e8e173cda704685c3a8d6c6edfeb5030033708ba[m
Author: Konrad Zawora <kzawora@habana.ai>
Date:   Wed Jun 25 18:33:36 2025 +0200

    Update README.md

[33mcommit 765fda979f35ce3db461d4ba7f6abbc121b6ef60[m
Author: Konrad Zawora <kzawora@habana.ai>
Date:   Wed Jun 25 19:15:35 2025 +0300

    Initial vllm-gaudi commit
    
    Co-authored-by: Chendi.Xue <chendi.xue@intel.com>
    Co-authored-by: Michal Adamczyk <madamczyk@habana.ai>
    Signed-off-by: Konrad Zawora <kzawora@habana.ai>
