from setuptools import setup, find_packages

setup(
    name='EasyDeL',
    version='0.0.35',
    author='Erfan Zare Chavoshi',
    author_email='erfanzare82@eyahoo.com',
    description='An open-source library to make training faster and more optimized in Jax/Flax',
    url='https://github.com/erfanzar/EasyDeL',
    packages=find_packages('lib/python'),
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Developers',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
        'License :: OSI Approved :: Apache Software License',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
        'Programming Language :: Python :: 3.11',
    ],
    keywords='machine learning, deep learning, pytorch, jax, flax',
    install_requires=[
        "jax>=0.4.10",
        "jaxlib>=0.4.10",
        "flax>=0.7.1",
        "fjformer>=0.0.1",
        "numpy~=1.24.2",
        "typing>=3.7.4.3",
        "transformers>=4.33.0",
        "einops>=0.6.1",
        "optax~=0.1.7",
        "msgpack>=1.0.5",
        "ipython>=8.17.2",
        "tqdm==4.65.0",
        "datasets==2.14.3",
        "setuptools>=68.0.0",
        'torch>=2.0.1',
        "fastapi>=0.103.0",
        "gradio~=3.41.2",
        "distrax",
        "rlax",
        "wandb>=0.15.9",
        "uvicorn~=0.23.2",
        "pydantic~=2.3.0",
        "tensorboard",
        "chex>=0.1.82",
        # add any other required dependencies here
    ],
    python_requires='>=3.8',
    package_dir={'': 'lib/python'},

)
