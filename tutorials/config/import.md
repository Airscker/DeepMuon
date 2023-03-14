# Dynamic importing mechanism

Up to now, there are many different configuration mechanisms in many different projects. Most of them require us to put the configuration file in fixed positions or we need to pack the parameters into a `class`, etc. These mechanisms are complex and inflexible. To solve these problems and provide a more direct experience of experiment planning, we used the dynamic importing mechanism in DeepMuon. You can put your configuration file anywhere, and you don't have to pack parameters. You even don't need to know how to create an experiment from zero, you just need to make your datasets, models, and loss functions available then you can start your experiments. It saves much time and energy to help you focus on more important problems.

The dynamic mechanism used by DeepMuon, actually is very simple:

```python
def import_module(module_path: str):
    '''
    ## Import python module according to the file path

    ### Args:
        - module_path: the path of the module to be imported

    ### Return:
        - the imported module
    '''
    assert module_path.endswith('.py'), f'Config file must be a python file but {module_path} is given'
    total_path = os.path.abspath(module_path)
    assert os.path.exists(total_path), f'Configuration file {total_path} does not exist. Please check the path again'
    module_spec = importlib.util.spec_from_file_location('', total_path)
    module = importlib.util.module_from_spec(module_spec)
    module_spec.loader.exec_module(module)
    return module
```

The function gets the path of the python file and imports its elements. This means that you just need to provide the file to be loaded and no more fixed position, no more packing needed. You don't have to insert the configuration file manually just like:

```python
os.environ['name']=module
```

The operation shown above means every time you create customized models, datasets, optimizers, etc. you need to insert them into the environment again and again.