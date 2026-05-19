import re
import pandas as pd
from pathlib import Path


def parse_log_file(filepath):
    """Parse a single log file and extract benchmark results."""
    with open(filepath, 'r') as f:
        content = f.read()

    # Extract TP size from filename
    tp_match = re.search(r'tp(\d+)', filepath.name)
    tp_size = int(tp_match.group(1)) if tp_match else None

    # Extract model config
    model_match = re.search(r'Model config: (.+)', content)
    model_config = model_match.group(1) if model_match else "Unknown"

    results = []

    # Find all num_chunks sections
    chunk_sections = re.findall(
        r'--- Testing num_chunks=(\d+) ---.*?'
        r'Tokens\s+Time \(ms\)\s+Throughput.*?\n'
        r'-+\n(.*?)(?=\n--- Testing num_chunks=|\n===|$)', content, re.DOTALL)

    for num_chunks, data_section in chunk_sections:
        # Parse each token count result
        token_results = re.findall(r'(\d+)\s+([\d.]+)\s+([\d.]+)', data_section)

        for tokens, time_ms, throughput in token_results:
            results.append({
                'TP Size': tp_size,
                'Model': model_config,
                'Num Chunks': int(num_chunks),
                'Tokens': int(tokens),
                'Time (ms)': float(time_ms),
                'Throughput (tokens/ms)': float(throughput)
            })

    return results


def parse_summary_section(filepath):
    """Parse the summary/speedup section from log file."""
    with open(filepath, 'r') as f:
        content = f.read()

    # Extract TP size
    tp_match = re.search(r'tp(\d+)', filepath.name)
    tp_size = int(tp_match.group(1)) if tp_match else None

    summaries = []

    # Find all summary sections
    summary_sections = re.findall(
        r'--- num_chunks=(\d+) ---.*?'
        r'Tokens\s+Baseline \(ms\)\s+Chunked \(ms\)\s+Speedup\s+Recommendation.*?\n'
        r'-+\n(.*?)'
        r'>>> Recommended cutoff for num_chunks=\d+: (\d+) tokens', content, re.DOTALL)

    for num_chunks, data_section, cutoff in summary_sections:
        # Parse each comparison
        comparisons = re.findall(r'(\d+)\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)\s+(Use \w+)', data_section)

        for tokens, baseline, chunked, speedup, recommendation in comparisons:
            summaries.append({
                'TP Size': tp_size,
                'Num Chunks': int(num_chunks),
                'Tokens': int(tokens),
                'Baseline (ms)': float(baseline),
                'Chunked (ms)': float(chunked),
                'Speedup': float(speedup),
                'Recommendation': recommendation,
                'Cutoff': int(cutoff)
            })

    return summaries


def main():
    # Define log files
    log_files = [
        Path('tests/unit_tests/ops/row_log_longSeqs_tp2.log'),
        Path('tests/unit_tests/ops/row_log_longSeqs_tp4.log'),
        Path('tests/unit_tests/ops/row_log_longSeqs_tp8.log')
    ]

    # Parse all files
    all_results = []
    all_summaries = []

    for log_file in log_files:
        if log_file.exists():
            print(f"Parsing {log_file.name}...")
            all_results.extend(parse_log_file(log_file))
            all_summaries.extend(parse_summary_section(log_file))
        else:
            print(f"Warning: {log_file} not found!")

    # Create DataFrames
    df_results = pd.DataFrame(all_results)
    df_summaries = pd.DataFrame(all_summaries)

    # Create Excel writer
    output_file = 'benchmark_results.xlsx'
    with pd.ExcelWriter(output_file, engine='openpyxl') as writer:
        # Write raw results
        if not df_results.empty:
            df_results.to_excel(writer, sheet_name='Raw Results', index=False)

        # Write summary/speedup results
        if not df_summaries.empty:
            df_summaries.to_excel(writer, sheet_name='Speedup Summary', index=False)

        # Create pivot tables for easier analysis
        if not df_results.empty:
            # Pivot: Time by TP Size and Num Chunks
            pivot_time = df_results.pivot_table(values='Time (ms)',
                                                index=['Tokens'],
                                                columns=['TP Size', 'Num Chunks'],
                                                aggfunc='first')
            pivot_time.to_excel(writer, sheet_name='Time Comparison')

            # Pivot: Throughput by TP Size and Num Chunks
            pivot_throughput = df_results.pivot_table(values='Throughput (tokens/ms)',
                                                      index=['Tokens'],
                                                      columns=['TP Size', 'Num Chunks'],
                                                      aggfunc='first')
            pivot_throughput.to_excel(writer, sheet_name='Throughput Comparison')

        # Create summary pivot: Speedup by TP and Chunks
        if not df_summaries.empty:
            pivot_speedup = df_summaries.pivot_table(values='Speedup',
                                                     index=['Tokens'],
                                                     columns=['TP Size', 'Num Chunks'],
                                                     aggfunc='first')
            pivot_speedup.to_excel(writer, sheet_name='Speedup by Config')

            # Create cutoff summary
            cutoff_summary = df_summaries.groupby(['TP Size', 'Num Chunks']).agg({'Cutoff': 'first'}).reset_index()
            cutoff_summary.to_excel(writer, sheet_name='Cutoff Recommendations', index=False)

    print(f"\nResults exported to {output_file}")
    print(f"Total results parsed: {len(all_results)}")
    print(f"Total summary entries: {len(all_summaries)}")

    # Display summary statistics
    if not df_results.empty:
        print("\n=== Quick Summary ===")
        print("\nTP Sizes tested:", sorted(df_results['TP Size'].unique()))
        print("Num Chunks tested:", sorted(df_results['Num Chunks'].unique()))
        print("Token counts tested:", sorted(df_results['Tokens'].unique()))


if __name__ == "__main__":
    main()
