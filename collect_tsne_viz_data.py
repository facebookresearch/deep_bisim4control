import torch
import numpy as np
import os
import dmc2gym

import utils
from train import parse_args, make_agent

root = '/checkpoint/amyzhang/pixel-pets/walker_walk/bisim_videokinetics/seed_3/model'
actor_path = os.path.join(root, 'actor_880000.pt')
critic_path = os.path.join(root, 'critic_880000.pt')

args = parse_args()
args.domain_name = 'walker'
args.task_name = 'walk'
args.image_size = 84
args.seed = 1
args.agent = 'bisim'
args.encoder_type = 'pixel'
args.action_repeat = 2
args.img_source = 'video'
args.num_layers = 4
args.num_filters = 32
args.hidden_dim = 1024
args.resource_files = '/datasets01/kinetics/070618/400/train/driving_car/*.mp4'
args.total_frames = 5000
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

env = dmc2gym.make(
    domain_name=args.domain_name,
    task_name=args.task_name,
    resource_files=args.resource_files,
    img_source=args.img_source,
    total_frames=args.total_frames,
    seed=args.seed,
    visualize_reward=False,
    from_pixels=(args.encoder_type == 'pixel'),
    height=args.image_size,
    width=args.image_size,
    frame_skip=args.action_repeat
)
env = utils.FrameStack(env, k=args.frame_stack)
agent = make_agent(
    obs_shape=env.observation_space.shape,
    action_shape=env.action_space.shape,
    args=args,
    device=device
)

agent.actor.load_state_dict(torch.load(actor_path))
agent.critic.load_state_dict(torch.load(critic_path))

obs = env.reset()
done = False
obses = [obs]
values = []
embeddings = []
for step in range(20000):
    if done:
        obs = env.reset()
        done = False
    with torch.no_grad():
        action = agent.sample_action(obs)
        values.append(min(agent.critic(torch.Tensor(obs).to(device).unsqueeze(0), torch.Tensor(action).to(device).unsqueeze(0))).item())
        embeddings.append(agent.critic.encoder(torch.Tensor(obs).unsqueeze(0).to(device)).cpu().numpy())
    obs, reward, done, _ = env.step(action)
    obses.append(obs)

    if step % 20 == 0:
        print(reward, step, values[-1])
    
dataset = {
    'obs': obses,
    'values': values,
    'embeddings': embeddings
}
torch.save(dataset, 'train_dataset.pt')