# Update pip to the latest version
Write-Host "Updating pip..."
python -m pip install --upgrade pip

pipreqs --force

# Update the requirements from requirements.txt
Write-Host "Updating requirements..."
pip install -r requirements.txt

$userInput = Read-Host "Press Enter to continue or type something to exit"

