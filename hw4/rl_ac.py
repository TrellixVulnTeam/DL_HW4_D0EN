import gym
import torch
import torch.nn as nn
import torch.nn.functional

from .rl_pg import TrainBatch, PolicyAgent, VanillaPolicyGradientLoss


class AACPolicyNet(nn.Module):
    def __init__(self, in_features: int, out_actions: int, **kw):
        """
        # TODO: Tweak values?
        Create a model which represents the agent's policy.
        :param in_features: Number of input features (in one observation).
        :param out_actions: Number of output actions.
        :param kw: Any extra args needed to construct the model.
        """
        super().__init__()
        #  Implement a dual-head neural net to approximate both the
        #  policy and value. You can have a common base part, or not.
        # ====== YOUR CODE: ======
        layers1 = (256, 128)
        nns1 = (nn.ReLU(), nn.ReLU())

        layers2 = (128, 64)
        nns2 = (nn.ReLU(), nn.ReLU())

        if kw[layers1]:
            layers1 = kw[layers1]

        if kw[nns1]:
            nns1 = kw[nns1]

        if kw[layers2]:
            layers2 = kw[layers2]

        if kw[nns2]:
            nns2 = kw[nns2]

        linear_layers1 = (nn.Linear(in_features, layers1[0]), nn.Linear(layers1[0], layers1[1]),
                         nn.Linear(layers1[1], out_actions))

        linear_layers2 = (nn.Linear(in_features, layers2[0]), nn.Linear(layers2[0], layers2[1]),
                         nn.Linear(layers2[1], 1))

        self.p_model = nn.Sequential(*[linear_layers1[0], nns1[0], linear_layers1[1], nns1[1],
                                     linear_layers1[2]])
        self.v_model = nn.Sequential(*[linear_layers2[0], nns2[0], linear_layers2[1], nns2[1],
                                     linear_layers2[2]])
        # ========================

    def forward(self, x):
        """
        :param x: Batch of states, shape (N,O) where N is batch size and O
        is the observation dimension (features).
        :return: A tuple of action values (N,A) and state values (N,1) where
        A is is the number of possible actions.
        """
        #  Implement the forward pass.
        #  calculate both the action scores (policy) and the value of the
        #  given state.
        # ====== YOUR CODE: ======
        action_scores = self.p_model(x)
        state_values = self.v_model(x)
        # ========================

        return action_scores, state_values

    @staticmethod
    def build_for_env(env: gym.Env, device="cpu", **kw):
        """
        Creates a A2cNet instance suitable for the given environment.
        :param env: The environment.
        :param kw: Extra hyperparameters.
        :return: An A2CPolicyNet instance.
        """
        # ====== YOUR CODE: ======
        in_features = env.observation_space.shape[0]
        out_features = env.action_space.n

        net = AACPolicyNet(in_features, out_features, **kw)
        # ========================
        return net.to(device)


class AACPolicyAgent(PolicyAgent):
    def current_action_distribution(self) -> torch.Tensor:
        # ====== YOUR CODE: ======
        res = self.p_net(self.curr_state)
        actions_proba = res[0].softmax(dim=0)
        # ========================
        return actions_proba


class AACPolicyGradientLoss(VanillaPolicyGradientLoss):
    def __init__(self, delta: float):
        """
        Initializes an AAC loss function.
        :param delta: Scalar factor to apply to state-value loss.
        """
        super().__init__()
        self.delta = delta

    def forward(self, batch: TrainBatch, model_output, **kw):

        # Get both outputs of the AAC model
        action_scores, state_values = model_output

        # TODO: Calculate the policy loss loss_p, state-value loss loss_v and
        #  advantage vector per state.
        #  Use the helper functions in this class and its base.
        # ====== YOUR CODE: ======
        loss_v = self._value_loss(batch, state_values)
        advantage = self._policy_weight(batch, state_values)
        loss_p = self._policy_loss(batch, action_scores, advantage)
        # ========================

        loss_v *= self.delta
        loss_t = loss_p + loss_v
        return (
            loss_t,
            dict(
                loss_p=loss_p.item(),
                loss_v=loss_v.item(),
                adv_m=advantage.mean().item(),
            ),
        )

    def _policy_weight(self, batch: TrainBatch, state_values: torch.Tensor):
        #  Calculate the weight term of the AAC policy gradient (advantage).
        #  Notice that we don't want to backprop errors from the policy
        #  loss into the state-value network.
        # ====== YOUR CODE: ======
        advantage = torch.sub(batch.q_vals, state_values.detach().view(-1), alpha=1)
        # ========================
        return advantage

    def _value_loss(self, batch: TrainBatch, state_values: torch.Tensor):
        # ====== YOUR CODE: ======
        qvals = batch.q_vals
        values = state_values.squeeze()

        loss_v = nn.MSELoss()(values, qvals)
        # ========================
        return loss_v
