# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


import json
import os
import subprocess
from imaginaire.utils import log
from gradio_deployment.fwk.gradio_util import get_output_folder, get_outputs


class GradioCLIApp:
    """Most basic server using an existing CLI model.
    The parameter validation is left to the CLI app.
    This server has no communication channel with the workers, so no errors are reported."""

    def __init__(self, num_workers: int = 8, checkpoint_dir: str = "checkpoints", out_dir: str = "outputs"):
        self.num_workers = num_workers
        self.checkpoint_dir = "checkpoints/nvidia/Cosmos-Transfer2-Private/model.pt"
        self.process = None
        self.out_dir = out_dir
        self._setup_environment()

    def _setup_environment(self):
        self.env = os.environ.copy()

    def infer_dict(self, args: dict, output_dir=None):
        if output_dir is None:
            output_dir = get_output_folder(self.output_dir)
        else:
            os.makedirs(output_dir, exist_ok=True)

        log.info(f"Starting {self.num_workers} worker processes with torchrun")
        cli_cmd = "cosmos_transfer2/inference/inference_vid2vid_control_batch.py"

        log.debug(json.dumps(args, indent=2))
        # Save arguments to JSON file in output_dir
        os.makedirs(output_dir, exist_ok=True)
        args_file = os.path.join(output_dir, "inference_args.json")
        with open(args_file, "w") as f:
            json.dump(args, f, indent=2)
        log.info(f"Saved inference arguments to {args_file}")

        torchrun_cmd = [
            "torchrun",
            f"--nproc_per_node={self.num_workers}",
            "--nnodes=1",
            "--node_rank=0",
            cli_cmd,
            f"--ckpt_path={self.checkpoint_dir}",
            f"--num_gpus={self.num_workers}",
            f"--controlnet_specs={args_file}",
            f"--save_root={output_dir}",
        ]

        log.info(f"Running command: {' '.join(torchrun_cmd)}")

        # Launch worker processes
        try:
            self.process = subprocess.Popen(
                torchrun_cmd,
                env=self.env,
                text=True,
                bufsize=1,
                universal_newlines=True,
            )

            # Wait for the process to complete
            return_code = self.process.wait()

            if return_code == 0:
                log.info("Inference completed successfully")
            else:
                log.error(f"Inference failed with return code: {return_code}")
                raise subprocess.CalledProcessError(return_code, torchrun_cmd)

        except Exception as e:
            log.error(f"Error running inference: {e}")
            raise e

        return get_outputs(output_dir)

    def infer(
        self,
        request_text,
        output_folder=None,
    ):
        try:
            request_data = json.loads(request_text)
        except json.JSONDecodeError as e:
            return None, f"Error parsing request JSON: {e}\nPlease ensure your request is valid JSON."

        return self._infer(request_data, output_folder)
