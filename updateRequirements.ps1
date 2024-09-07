## Install pipreqs through pips
# pip install pipreqs

pipreqs .\src --force --ignore database

Move-Item .\src\requirements.txt .\ -Force
