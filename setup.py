from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
  long_description = fh.read()

setup(name="dtil",
      version="0.0.1",
      author="Sangwon Seo",
      author_email="sangwon.seo@rice.edu",
      description="Deep Team Imitation Learner",
      long_description=long_description,
      long_description_content_type="text/markdown",
      packages=find_packages(exclude=["train"]),
      python_requires='>=3.8',
      install_requires=[
          'numpy', 'tqdm', 'scipy', 'sparse', 'torch', 'termcolor',
          'tensorboard', 'hydra-core>=1.3', 'wandb>=0.15', 'pettingzoo',
          'pygame', 'opencv-python', 'pillow', 'pandas'
      ])
