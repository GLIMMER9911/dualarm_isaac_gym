import isaacgymenvs
import torch
import yaml
from isaacgymenvs.learning.amp_models import ModelAMPContinuous
from isaacgymenvs.learning.amp_network_builder import AMPBuilder
from rl_games.algos_torch import model_builder, torch_ext


class RlGamesPolicy:
    """
    Plays a policy trained with RlGames.
    """
    def __init__(self, checkpoint_file, base_config,
                 training_config, critic_num_obs,
                 critic_state_types=None, actor_state_types=None):

        self.base_config = base_config
        self.training_config = training_config

        self.robot_num_dofs = self.base_config["env"]["numActions"]

        model_builder.register_network('amp', lambda **kwargs : AMPBuilder())
        model_builder.register_model('continuous_amp', lambda network, **kwargs : ModelAMPContinuous(network))

        train_params = self.training_config["params"]
        config = train_params["config"]
        builder = model_builder.ModelBuilder()
        config['network'] = builder.load(train_params)
        network = config['network']
        # The actor (policy) and critic network are built with the same size.
        # A mask is used to zero privilege observations for the actor.
        self.critic_num_obs = critic_num_obs
        obs_shape = (self.critic_num_obs,)
        num_agents = 1
        obs_shape = torch_ext.shape_whc_to_cwh(obs_shape)
        net_config = {
            'actions_num' : self.robot_num_dofs,
            'input_shape' : obs_shape,
            'num_seqs' : num_agents,
            'value_size': 1,
            'normalize_value': config["normalize_value"],
            'normalize_input': config["normalize_input"],
        }
        task_type = self.base_config["task"]["task_type"]
        amp_obs_types = self.base_config["task"][task_type]["ampObservation"]

        self.critic_state_types = critic_state_types
        self.actor_state_types = actor_state_types

        net_config['amp_input_shape'] = self._calculate_obs_size(amp_obs_types, buffer_size =self.base_config["env"]["numAMPObsSteps"])
        net_config['actor_mask'] = self._get_actor_obs_mask()

        self.model = network.build(net_config)
        self.model.load_state_dict(torch.load(checkpoint_file)['model'])
        self.model.eval()

    def _get_actor_obs_mask(self):
        # Creates a mask to mask out the privilege observations
        if self.base_config["name"] == "MazeAMP":
            mask = torch.ones(self.critic_num_obs)
        else:
            from isaacgymenvs.tasks.amp.punyo_amp_base import get_actor_obs_mask
            mask = get_actor_obs_mask(
                    self.critic_state_types,
                    self.actor_state_types,
                    self.robot_num_dofs,
            )

        return mask

    def _calculate_obs_size(self, obs_types, buffer_size=1):
        if self.base_config["name"] == "MazeAMP":
            from isaacgymenvs.tasks.amp.maze_amp_base import get_obs_size_per_obs_type
        else:
            from isaacgymenvs.tasks.amp.punyo_amp_base import get_obs_size_per_obs_type

        size = 0
        for obs_type in obs_types:
            size += get_obs_size_per_obs_type(obs_type, self.robot_num_dofs)
        return (size * int(buffer_size),)

    @torch.no_grad()
    def step(self, processed_obs):

        input_dict = {
            'is_train': False,
            'prev_actions': None,
            'obs' : processed_obs,
            'rnn_states' : None
        }
        # NOTE: For deterministic execution, use self.model(input_dict)["mus"]
        action = self.model(input_dict)["actions"]
        value = self.model(input_dict)["values"]
        return action.numpy(), False, value.numpy()
