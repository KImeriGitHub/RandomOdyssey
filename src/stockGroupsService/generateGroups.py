from stockGroupsService.GroupOver20Years import GroupOver20Years
from stockGroupsService.GroupSwiss import GroupSwiss
from stockGroupsService.GroupManager import GroupManager

def generateGroups():
    dbPath = "src/database"
    groupPath = "src/stockGroups"

    group_criteria = [
        GroupOver20Years(),
        GroupSwiss(),
        # Add other group criteria here
    ]

    manager = GroupManager(databasePath=dbPath, stockGroupPath=groupPath, groupCriteria=group_criteria)
    manager.generateGroups()