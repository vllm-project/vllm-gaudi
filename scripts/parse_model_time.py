import json
import sys
import os
import numpy as np


def calculate_median(data_list):
    """Helper function to calculate the median of a list."""
    if not data_list:
        return None
    sorted_list = sorted(data_list)
    n = len(sorted_list)
    mid_index = n // 2
    if n % 2 == 0:
        # Even number of elements, average the two middle ones
        return (sorted_list[mid_index - 1] + sorted_list[mid_index]) / 2
    else:
        # Odd number of elements, return the middle one
        return sorted_list[mid_index]


def get_longest_tid(synchronize_events):
    duration_time = 0
    longest_tid = None
    for tid, sync_data in synchronize_events.items():
        if 'total_time' in sync_data and sync_data[
                'total_time'] > duration_time:
            longest_tid = tid
            duration_time = sync_data['total_time']
    return longest_tid


def get_closest_event(model_ts, events):
    closest_event = None
    min_diff = float('inf')
    for sync_event in events:
        sync_ts = sync_event.get("ts")
        if sync_ts is not None:
            diff = abs(sync_ts - model_ts)
            if diff < min_diff and sync_event.get("dur") > 1000:
                min_diff = diff
                closest_event = sync_event
    return closest_event


def process_json_events(input_file_path, exclude_tids):
    """
    Loads a JSON file, filters events, and associates 'synchronizeEvent'
    events with the closest 'model_decode_*' or 'model_prompt' events.

    Args:
        input_file_path (str): The path to the input JSON file.
        output_file_path (str): The path to save the processed JSON file.
    """
    output_file_path = input_file_path.replace('.json', '_processed.json')
    # if output_file_path exist
    if not os.path.exists(output_file_path):
        try:
            with open(input_file_path) as f:
                data = json.load(f)
        except FileNotFoundError:
            print(f"Error: Input file not found at '{input_file_path}'")
            return
        except json.JSONDecodeError:
            print(f"Error: Could not decode JSON from '{input_file_path}'")
            return

        # Filter events into separate lists
        synchronize_events = {}
        model_events = []
        model_forward_events = []

        event_list = data.get('traceEvents', data) if isinstance(
            data, dict) else data

        if not isinstance(event_list, list):
            print("Error: JSON data is not in the expected list format.")
            return

        for event in event_list:
            event_name = event.get("name", "")
            if event_name == "synchronizeEvent (accel0)":
                tid = event.get("tid")
                if tid not in synchronize_events:
                    synchronize_events[tid] = {'total_time': 0, 'events': []}
                synchronize_events[tid]['events'].append(event)
                synchronize_events[tid]['total_time'] += event.get("dur", 0)
            elif event_name.startswith("decode") or event_name.startswith(
                    "prefill"):
                model_events.append(event)
            elif event_name.startswith("model_forward"):
                model_forward_events.append(event)

        if not synchronize_events:
            print("Warning: No 'synchronizeEvent (accel0)' events found.")

        if not model_events:
            print(
                "Warning: No 'model_decode_*' or 'model_prompt' events found.")

        tids = list(synchronize_events.keys())
        tids = [tid for tid in tids if tid not in exclude_tids]
        if len(tids) > 1:
            tid = get_longest_tid(synchronize_events)
            if tid is None:
                print(
                    "Error: No valid 'synchronizeEvent (accel0)' events found."
                )
                return
        else:
            tid = tids[0]
        synchronize_events = synchronize_events[tid]['events']
        # For each model event, find the closest synchronizeEvent
        for model_event in model_events:
            event_name = model_event.get("name", "")
            model_ts = model_event.get("ts")
            if model_ts is None:
                continue
            closest_sync_event = get_closest_event(model_ts,
                                                   synchronize_events)
            if closest_sync_event:
                # Add the closest sync event to the model event's dictionary
                model_event['duration'] = closest_sync_event.get("dur", 0)
            closest_modelfwd_event = get_closest_event(model_ts,
                                                       model_forward_events)
            if closest_modelfwd_event:
                model_event['forward_name'] = closest_modelfwd_event.get(
                    "name", None)

        # save model_event to output file
        try:
            with open(output_file_path, 'w') as f:
                json.dump(model_events, f, indent=4)
            print(f"Processed events saved to '{output_file_path}'")
        except OSError as e:
            print(
                f"Error saving processed events to '{output_file_path}': {e}")
            return
    else:
        # load from output file
        try:
            with open(output_file_path) as f:
                model_events = json.load(f)
            print(f"Loaded processed events from '{output_file_path}'")
        except FileNotFoundError:
            print(f"Error: Processed file not found at '{output_file_path}'")
            return
        except json.JSONDecodeError:
            print(f"Error: Could not decode JSON from '{output_file_path}'")
            return

    # generate a summary
    # event_name, event_average_duration based on event_name
    summary = {}
    for event in model_events:
        event_name = event.get("name", "")
        duration = event.get("duration", 0)
        forward_name = event.get("forward_name", None)
        if event_name not in summary:
            summary[event_name] = {}
        if forward_name not in summary[event_name]:
            summary[event_name][forward_name] = []
        summary[event_name][forward_name].append(duration)

    # Calculate average duration for each event
    new_summary = {}
    for event_name, forward_events in summary.items():
        new_summary[event_name] = {}
        for forward_name, durations in forward_events.items():
            metrics = {
                'count': len(durations),
                'average_duration':
                np.mean(durations) / 1000 if durations else 0,
                'duration_list': durations,
            }
            new_summary[event_name][forward_name] = metrics

    # Save the summary to a JSON file
    print("Summary of events:")
    print("Total events processed:", len(new_summary.keys()))
    print("Event Name, Count, Average Duration (ms)")
    # new_summary sort by name
    new_summary = dict(sorted(new_summary.items(), key=lambda item: item[0]))
    for event_name, forward_events in new_summary.items():
        print(f"Event Name: {event_name}")
        forward_events = dict(
            sorted(forward_events.items(),
                   key=lambda item: get_number_from_string(item[0])))
        for forward_name, metrics in forward_events.items():
            print(f"{forward_name}, {metrics['count']}, \
                {metrics['average_duration']:.3f}")


def get_number_from_string(s):
    import re
    numbers = re.findall(r'\d+', s)
    return int(numbers[0]) * 10 + int(numbers[1])


if __name__ == "__main__":
    # How to run:
    # python your_script_name.py input.json output.json
    if len(sys.argv) < 2:
        print("Usage: python process_events.py <input_json_file>")
        sys.exit(1)

    input_path = sys.argv[1]
    exclude_tids = sys.argv[2].split(',') if len(sys.argv) > 2 else []
    process_json_events(input_path, exclude_tids)
