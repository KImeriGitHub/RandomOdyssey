# Random Odyssey

```bash
\*                  ~.				        \*
\*           Ya...___|__..aab     .   .		\*
\*            Y$$a  Y$$o  Y$$a   (     )	\*
\*             Y$$b  Y$$b  Y$$b   `.oo'		\*
\*             :$$$  :$$$  :$$$  ( (`-'		\*
\*    .---.    d$$P  d$$P  d$$P   `.`.		\*
\*   / .-._)  d$P'"""|"""'-Y$P      `.`.	\*
\*  ( (`._) .-.  .-. |.-.  .-.  .-.   ) )	\*
\*   \ `---( $ )( $ )( $ )( $ )( $ )-' /	\*
\*    `.    `-'  `-'  `-'  `-'  `-'  .' CJ	\*
\*~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~	\*
\*ASCII Art from asciiart.eu/vehicles/boats	\*
\*~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~	\*
```

## First Time Setup

### Setting up virtual environement
On a Terminal in the main project folder /RandomOdyssey:

```bash
> python -m venv .venv
> .venv\Scripts\activate
> pip install -r .\requirements.txt
```

### Setting up a notebook
Active venv, then in the main project folder /RandomOdyssey:

```bash
> pip install ipykernel
> python -m ipykernel install --user --name=.venv
```

In the notebook, you might need to add the following to have working imports

```bash
import sys
import os
project_dir = os.path.abspath("..")
if project_dir not in sys.path:
    sys.path.append(project_dir)
```

## Update folder

### To update the folder
Run updatePythonProject.ps1
(It updates requirements.txt)

### Update only requirements
Run updateRequirements.ps1

## Run scripts
On a terminal you need to active venv first:
run active_venv.ps1

### Database update
python runUpdateDatabase.py