<p align="center">
  <img src="https://img.shields.io/badge/python-3.9+-blue">
</p>

## Feng Lab LLM Api

This repo enables inference for a variety of LLMs including Versa. 

### Supported Models
For a list of supported models please see [here](https://github.com/jjfenglab/llm-api/blob/main/constants.py#L9)

### LLM Api

To use LLM APIs add a .env file in the root folder of this directory

```bash
$ touch .env
```

Add your tokens
```bash
$ echo "OPENAI_ACCESS_TOKEN=<YOUR TOKEN>" >> .env
$ echo "HF_ACCESS_TOKEN=<YOUR TOKEN>" >> .env
$ echo "BEDROCK_ACCESS_KEY=<YOUR ACCESS KEY>" >> .env
$ echo "BEDROCK_ACCESS_TOKEN=<YOUR ACCESS TOKEN>" >> .env
$ echo "VERSA_API_KEY=<YOUR TOKEN>" >> .env
```

### Development 
To test any new changes use `test_script.py`

### Release Cutting
This repo is referenced by its release tag. When new functionality is added, you can start using it in your own repo by cutting a release and installing it in your repo

Before cutting a release please ensure new dependencies have been added to `pyproject.toml`. The package installation can be tested by running `llm-api $ pip install -e .`  
