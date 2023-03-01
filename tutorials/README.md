# DeepMuon Tutorials

Before seeking the mystery of science based on DeepMuon, it's important to familiarize yourself with the usage of DeepMuon. Here we provide several detailed tutorials to help you understand the basic pipeline of exploration.

The tutorials are classified into different blocks:

- Start your first experiment
  - [Customize `Dataset`](start_exp/cus_dataset.md)
  - [Customize model](start_exp/cus_model.md)
  - [Customize configuration](start_exp/cus_config.md)
  - [Start your first experiment](start_exp/start_exp.md)
- `Dataset` based on PyTorch
  - [How to add augmentation?](dataset/augment.md)
  - [What to do when adding a customized `Dataset` into DeepMuon?](dataset/dataset.md)
- Which model fits best?
  - [From zero to infinity](models/zero_inf.md)
  - [Intrinsically supported models](models/intrinsic_models.md)
- Suitable Loss Function
  - [How to create your own loss function?](loss/loss.md)
  - [Evaluation based on scores & labels](loss/evaluation.md)
- Configuration Mechanism
  - [Elements of `config.py`](config/config.md)
  - [Dynamic importing mechanism](config/import.md)
- Fully control your training/testing pipelines
  - [Data Distributed Parallel & Fully Sharded Distributed Parallel](train_test/parallel.md)
  - [16-bit float tensor & 32-bit float tensor & 64-bit double tensor](train_test/precision.md)
  - [Discriminative / Regression tasks](train_test/task.md)
  - [Gradient clip & Gradient accumulation](train_test/grad.md)
- Auto logging system
  - [Usage of Tensorboard](log/tensorboard.md)
  - [JSON log](log/json.md)
  - [Text log](log/text.md)


- Interpreting model
  - [Neuron Flow](interpret/neuron_flow.md)
  - [Model profiler](interpret/profiler.md)
  - [Attribution](interpret/attr.md)
