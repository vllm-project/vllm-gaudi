# SPDX-License-Identifier: Apache-2.0
import os
import shutil
import sys
import time

class ScriptGenerator:

    def __init__(self, template_script_path, output_script_path, variables, mode, log_dir="logs", varlist_conf_path=None):
        self.template_script_path = template_script_path
        self.varlist_conf_path = varlist_conf_path
        self.output_script_path = output_script_path
        self.variables = variables
        self.log_dir = log_dir
        self.mode = mode
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
        os.makedirs(self.log_dir, exist_ok=True)
        if (os.environ.get("DRYRUN_SERVER")=='1' and self.mode=='server') or \
        (os.environ.get("DRYRUN_BENCHMARK")=='1' and self.mode=='benchmark'):
            print(f"[INFO] This is a dry run to save the command line file {self.output_script_path}.")
            shutil.copy(self.output_script_path, f"/local/{self.mode}/")
            print(f"[INFO] The command line file {self.output_script_path} saved at .cd/{self.mode}/{self.output_script_path}")            
            try:
                while True:
                    print("[INFO] Press Ctrl+C to exit.")                    
                    time.sleep(60)
            except KeyboardInterrupt:
                print("Exiting cmd mode.")
                sys.exit(0)
        else:
            os.execvp("bash", ["bash", self.output_script_path])

