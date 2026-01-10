# SPDX-License-Identifier: Apache-2.0
import os
import shutil
import sys
from pathlib import Path


def shutil_copy(source_file, destination_dir):
    try:
        src_path = Path(source_file)
        dst_dir_path = Path(destination_dir)

        dst_path = dst_dir_path / src_path.name

        # Ensure the destination directory exists
        dst_dir_path.mkdir(parents=True, exist_ok=True)

        shutil.copy(src_path, dst_path)
        print(f"[Info] File '{source_file}' saved at '{dst_path}'")

    except FileNotFoundError:
        print(f"Error: The source file '{source_file}' was not found.")
    except PermissionError:
        print(f"Error: Permission denied. Cannot access '{source_file}' or write to '{destination_dir}'.")
    except shutil.SameFileError:
        print("Error: Source and destination files are the same.")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")


class ScriptGenerator:

    def __init__(self,
                 template_script_path,
                 output_script_path,
                 variables,
                 log_dir="logs",
                 dry_run_dir="/local/",
                 varlist_conf_path=None):
        self.template_script_path = template_script_path
        self.varlist_conf_path = varlist_conf_path
        self.output_script_path = output_script_path
        self.variables = variables
        self.log_dir = log_dir
        self.dry_run_dir = dry_run_dir
        self.log_file = os.path.join(self.log_dir,
                                     f"{os.path.splitext(os.path.basename(self.output_script_path))[0]}.log")

    def generate_script(self, vars_dict):
        """
        Generate the script from a template, 
        replacing placeholders with environment variables.
        """
        with open(self.template_script_path) as f:
            template = f.read()
        # Create our output list
        if self.varlist_conf_path:
            output_dict = {}
            with open(self.varlist_conf_path) as var_file:
                for line in var_file:
                    param = line.strip()
                    output_dict[param] = vars_dict[param]
            export_lines = "\n".join([f"export {k}={v}" for k, v in output_dict.items()])
        else:
            export_lines = "\n".join([f"export {k}={v}" for k, v in vars_dict.items()])
        script_content = template.replace("#@VARS", export_lines)
        with open(self.output_script_path, 'w') as f:
            f.write(script_content)

    def make_script_executable(self):
        """
        Make the output script executable.
        """
        os.chmod(self.output_script_path, 0o755)

    def print_script(self):
        """
        Print the generated script for debugging.
        """
        print(f"\n===== Generated {self.output_script_path} =====")
        with open(self.output_script_path) as f:
            print(f.read())
        print("====================================\n")

    def create_and_run(self):
        self.generate_script(self.variables)
        self.make_script_executable()
        self.print_script()

        # Run the generated script and redirect output to log file
        print(f"Starting script, logging to {self.log_file}")
        try:
            os.makedirs(self.log_dir, exist_ok=True)
        except Exception:
            print(f"Error: could not create {self.log_dir}.")

        if os.environ.get("DRY_RUN") == '1':
            shutil_copy(self.output_script_path, self.dry_run_dir)
            print(f"[INFO] This is a dry run to save the command line file {self.output_script_path}.")
            sys.exit(0)
        else:
            os.execvp("bash", ["bash", self.output_script_path])
