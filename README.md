# AI_Workshop_IISER

Building AI Agents for Scientific Research

We will begin this two-day workshop by looking at how AI is currently transforming the IT industry. Drawing from my experience as a Data Scientist, I will share real-world examples of how we use AI to predict, prevent, and diagnose critical infrastructure issues in real-time. This industry context will set the stage for understanding why autonomous agents are the next major leap in software development.

Following this introduction, we will move to hands-on engineering. You will build your personal research assistant designed to streamline your literature reviews. We will cover the fundamentals of generating Reference-Backed Responses, teaching you how to ground AI outputs in real scientific data to ensure accuracy. We will also integrate voice technology, allowing you to ask questions aloud and listen to audio summaries of complex papers. You will leave with a functional application on your laptop and a clear understanding of how to build AI tools that automate your own academic workflows.

Pre-Workshop Setup Guide for workshop participants

To ensure we can jump straight into coding, please have the following ready on your laptop before arriving:

1. Software Installation

Python: Please install Python (version 3.9 or higher)
Code Editor: We recommend VS Code (Visual Studio Code) for writing and running the scripts.

2. API Keys (Free Accounts)
We will use powerful AI models that require "keys" to access. Please sign up for free accounts at the links below and save your API Keys in a text file (we will need them during the workshop):

Groq Cloud: This gives us access to the "Brain" (LLM) of our agent.
Sign up here: https://console.groq.com/keys

Tavily: This gives our agent the ability to "Search" the web.
Sign up here: https://tavily.com/

3. (Optional) Check Your Microphone
Since we will be building voice features, please ensure your laptop’s microphone is working.


# Environment Setup Instructions

## Prerequisites
- Install Miniconda or Anaconda from https://docs.conda.io/en/latest/miniconda.html

## Step 1: Create a new conda environment
conda create --name myproject python=3.11

## Step 2: Activate the environment
conda activate myproject

## Step 3: Install required packages
conda install numpy pandas matplotlib scikit-learn jupyter flask

## Step 4: (Optional) Install additional packages via pip if not available on conda
pip install <package-name>

## Step 5: Verify the installation
python -c "import numpy; import pandas; import matplotlib; import sklearn; import flask; print('All packages installed successfully!')"

## To deactivate the environment when done
conda deactivate

## To remove the environment if needed
conda remove --name myproject --all

## Notes
- Always activate the environment before running the project: conda activate myproject
- To export the environment for sharing: conda env export > environment.yml
- To recreate the environment from the file: conda env create -f environment.yml