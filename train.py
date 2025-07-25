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

import os
import subprocess
import hydra
import torch
import torch.multiprocessing as mp
import torch.distributed as dist
from omegaconf import DictConfig, OmegaConf
from threedgrut.utils.logger import logger
from threedgrut.utils.timer import timing_options

OmegaConf.register_new_resolver("int_list", lambda l: [int(x) for x in l])

# # Uncomment the following lines to enable debug timing
# timing_options.active = True
# timing_options.print_enabled = True


def get_git_revision_hash() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"]).decode("ascii").strip()
    except Exception:
        return "Unknown"


def _run(rank: int, world_size: int, conf: DictConfig) -> None:
    if world_size > 1:
        dist.init_process_group("nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)
    from threedgrut.trainer import Trainer3DGRUT
    trainer = Trainer3DGRUT(conf, device=torch.device(f"cuda:{rank}"), report_hook=None)
    trainer.run_training()
    if world_size > 1:
        dist.destroy_process_group()


@hydra.main(config_path="configs", version_base=None)
def main(conf: DictConfig) -> None:
    logger.info(f"Git hash: {get_git_revision_hash()}")
    logger.info(f"Compiling native code..")
    world_size = getattr(conf, "world_size", 1)
    if world_size > torch.cuda.device_count():
        world_size = torch.cuda.device_count()
    if world_size > 1:
        os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
        os.environ.setdefault("MASTER_PORT", "29500")
        mp.spawn(_run, args=(world_size, conf), nprocs=world_size)
    else:
        _run(0, 1, conf)


if __name__ == "__main__":
    main()
