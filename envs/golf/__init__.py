"""Golf swing task registration. Entry points stay strings so importing this stays cheap."""

import gymnasium as gym

from . import agents

gym.register(
    id="Isaac-Golf-Swing-Direct",
    entry_point=f"{__name__}.golf_env:GolfEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.golf_env_cfg:GolfEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:GolfPPORunnerCfg",
        "default_agent": "rsl_rl",
    },
)
