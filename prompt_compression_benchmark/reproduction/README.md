**The files in this directory are adapted from https://github.com/microsoft/LLMLingua/tree/2dbdbd37aef3b4346c2feec0ff8fba7dc3d42171/experiments/llmlingua2/evaluation**

# Usage

## Compression

```
cd scripts
chmod +x compress_<benchmark>.sh
./compress_<benchmark>.sh
```

## Evaluation

```
python eval_<benchmark>.py --config scripts/eval_<benchmark>.yaml [other arguments]
```
To see other argument options, try `python eval_<benchmark>.py --help`