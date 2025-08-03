# Example: Training with SAC agent

This example shows how to train using the SAC (Soft Actor-Critic) agent with skrl.

## Running SAC Training

To train using SAC, use the following command:

```bash
python scripts/skrl/train.py --task=t1 --algorithm=SAC --num_envs=1024 --timesteps=100000
```

Key parameters for SAC:
- `--algorithm=SAC`: Specifies the SAC algorithm
- `--num_envs`: Number of parallel environments (SAC can handle many environments)
- `--timesteps`: Total training timesteps (SAC typically needs more than on-policy methods)

## SAC Configuration

The SAC configuration is located at:
- Direct task: `booster_lab/tasks/direct/booster_lab/agents/skrl_sac_cfg.yaml`
- Manager-based task: `booster_lab/tasks/manager_based/booster_lab/agents/skrl_sac_cfg.yaml`

Key SAC hyperparameters:
- **batch_size**: 256 (larger batches often work better for SAC)
- **memory_size**: 25,000 (replay buffer size - optimized for high parallel envs)
- **learning_starts**: 1000 (start learning after collecting initial experience)
- **random_timesteps**: 1000 (random exploration steps)
- **polyak**: 0.005 (target network update rate)
- **actor_learning_rate**: 3e-4
- **critic_learning_rate**: 3e-4
- **entropy_learning_rate**: 3e-4

## Memory Considerations

SAC memory usage scales with: `memory_size × num_envs × observation_size × 4 bytes`

**For different environment counts:**
- **1024 envs**: ~5GB GPU memory (recommended for 16GB GPU)
- **2048 envs**: ~10GB GPU memory 
- **4096 envs**: ~20GB GPU memory (requires memory optimization)

**Memory size recommendations:**
- **Single environment**: 1,000,000 (standard)
- **1024 environments**: 50,000-100,000
- **4096+ environments**: 25,000 (current setting)

## SAC vs Other Algorithms

SAC is an off-policy algorithm with the following characteristics:

**Advantages:**
- More sample efficient than on-policy methods in many continuous control tasks
- Automatic entropy regularization for exploration
- Stable learning with replay buffer
- Works well with continuous action spaces

**Disadvantages:**
- More complex than PPO (requires 5 networks vs 2)
- Higher memory usage due to replay buffer
- Typically needs more wall-clock time per step

**When to use SAC:**
- Continuous control tasks
- When sample efficiency is important
- When you have sufficient computational resources
- For environments where exploration is challenging

## Playing/Evaluating SAC Checkpoints

To evaluate a trained SAC model:

```bash
python scripts/skrl/play.py --task=t1 --algorithm=SAC --checkpoint=path/to/checkpoint.pt
```

## Tuning Tips

1. **Replay Buffer Size**: Balance between memory usage and sample diversity
   - Single env: 1M+ for best performance
   - Many envs: 25K-100K to fit in GPU memory
2. **Batch Size**: Try 256 or 512 for better stability
3. **Learning Rates**: Start with 3e-4 for all networks
4. **Target Entropy**: Usually set automatically to -action_dim
5. **Polyak**: 0.005 is a good default for target network updates
6. **Network Size**: 256x256 networks often work well for complex tasks

## Optimal Commands for Your Hardware (64GB RAM, 16GB VRAM)

**For maximum sample efficiency (fewer envs, larger buffer):**
```bash
python scripts/skrl/train.py --task=t1 --algorithm=SAC --num_envs=1024 --max_iterations=100000
```

**For faster wall-clock training (more envs, smaller buffer):**
```bash
python scripts/skrl/train.py --task=t1 --algorithm=SAC --num_envs=4096 --max_iterations=25000
```

**For memory-constrained setups:**
```bash
python scripts/skrl/train.py --task=t1 --algorithm=SAC --num_envs=512 --max_iterations=200000
```
