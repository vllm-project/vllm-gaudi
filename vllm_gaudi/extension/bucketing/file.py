import os.path


class FileBucketingStrategy():
    '''
    FileBucketingStrategy allows to read buckets from json file.
    Files can be passed through flags:
    - VLLM_PROMPT_BUCKETING_FILE
    - VLLM_DECODE_BUCKETING_FILE
    Valid files should have each bucket listed in new line in this order:
    (batch_size, query_length, number_of_context_blocks)
    '''
    def get_prompt_buckets(self, max_num_prefill_seqs, block_size, 
                           max_num_batched_tokens, max_model_len):
        all_buckets = read_buckets_file(True)

        # Verify buckets - remove not valid
        prompt_buckets = []
        for bucket in all_buckets:
            bs, query, ctx = bucket
            if query + ctx * block_size > max_num_batched_tokens \
                or bs > max_num_prefill_seqs
                or (bs * math.celi(max_model_len / block_size) > max_model_len):
                #TODO include conti pa
                continue
            prompt_buckets.append(bucket)

        return sorted(prompt_buckets)

    def get_decode_buckets(self, max_num_seqs, block_size, 
                           max_num_batched_tokens, max_model_len,
                           num_max_blocks):
        all_buckets = read_buckets_file(False)

        return sorted(decode_buckets)


def read_buckets_file(is_prompt):
    file_name = get_context().VLLM_PROMPT_BUCKETING_FILE if is_prompt \
                else get_context().VLLM_DECODE_BUCKETING_FILE
    phase = 'prompt' if is_prompt else 'decode'

    assert os.path.isfile(file_name), \
            "File for {phase} buckets config doesn't exist")

    all_buckets = []
    with open(file_name, "r") as f:
        for line in f:
            bucket = line.strip()
            if not bucket or not bucket[0].isdigit():
                continue
            values = [b for b in bucket.replace(",", " ").split() if b]
         
            try:
                new_bucket = list(map(int, values))
            except ValueError:
                continue

            if len(new_bucket) == 3:
                all_buckets.append(tuple(new_bucket))
            elif len(new_bucket) == 2:
                all_buckets.append(tuple(new_bucket[0], new_bucket[1], 0))
            # skip other invaid configs

    if len(all_buckets) < 1:
        logger().info(f"No buckets found in {file_name} file for {phase}")
    return all_buckets
