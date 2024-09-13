from src.stockGroupsService.Group_Over20Years import Group_Over20Years
from src.stockGroupsService.Group_Swiss import Group_Swiss

# Main function
def generateGroups():
        dbpath = "src/database"
        grpath = "src/stockGroups"

        Group_Over20Years(dbpath, grpath).generateYaml()
        Group_Swiss(dbpath, grpath).generateYaml()