# Activate the virtual environment
Write-Host "Activating the virtual environment..."
$venvPath = "venv\Scripts\Activate.ps1"

if (Test-Path $venvPath) {
    & $venvPath
} else {
    Write-Host "Virtual environment not found at $venvPath. Please ensure the venv directory exists."
}

# Update the requirements from requirements.txt
Write-Host "Updating requirements..."
pip install -r requirements.txt

Write-Host "All tasks completed successfully."

$userInput = Read-Host "Press Enter to continue or type something to exit"

