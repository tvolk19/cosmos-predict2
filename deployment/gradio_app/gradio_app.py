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

from deployment.gradio_app.gradio_util import get_output_folder, get_outputs
from deployment.server.generic.model_server import ModelServer
from deployment.server.generic.model_worker import create_worker_pipeline
from imaginaire.utils import log


class GradioApp:
    def __init__(self, cfg):
        if cfg.num_gpus == 1:
            self.pipeline, self.validator = create_worker_pipeline(cfg)
        else:
            self.pipeline = ModelServer(cfg)
            _, self.validator = create_worker_pipeline(cfg, create_model=False)
        self.cfg = cfg

    def infer(
        self,
        request_text,
    ):
        output_folder = get_output_folder(self.cfg.output_dir)

        try:
            request_data = json.loads(request_text)
        except json.JSONDecodeError as e:
            return None, f"Error parsing request JSON: {e}\nPlease ensure your request is valid JSON."

        try:
            log.info(f"Model parameters: {json.dumps(request_data, indent=4)}")

            args_dict = self.validator.parse_and_validate(request_data)
            args_dict["output_dir"] = output_folder

            self.pipeline.infer(args_dict)

        except Exception as e:
            log.error(f"Error during inference: {e}")
            return None, f"Error: {e}"

        return get_outputs(output_folder)
