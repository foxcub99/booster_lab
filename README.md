readme for me to write what im doing so i dont forget later.

first add asset in source/booster_lab/booster_lab/assets/t1 (idk if this is copyrighted or anything but download is here https://booster.feishu.cn/wiki/DtFgwVXYxiBT8BksUPjcOwG4n4f)

as per https://isaac-sim.github.io/IsaacLab/main/source/overview/developer-guide/template.html

with external project, you need to add this to isaaclabs path

```python -m pip install -e source/<given-project-name>``` is the command

personally i didnt install with python so I used isaaclabs python

```make isaac ARGS="-m pip install -e source/booster_lab```

line 78 of t1_env_cfg needs absolute path for some reason, this will change depending on your setup.