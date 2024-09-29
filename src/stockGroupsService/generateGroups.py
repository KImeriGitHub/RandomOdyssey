from stockGroupsService.GroupOver20Years import GroupOver20Years
from stockGroupsService.GroupSwiss import GroupSwiss
from stockGroupsService.GroupManager import GroupManager
from stockGroupsService.GroupSwissOver20Years import GroupSwissOver20Years

def generateGroups():
    dbPath = "src/database"
    groupPath = "src/stockGroups"

    group_criteria = [
        GroupOver20Years(),
        GroupSwiss(),
        GroupSwissOver20Years(),
    ]

    manager = GroupManager(databasePath=dbPath, stockGroupPath=groupPath, groupCriteria=group_criteria)
    manager.generateGroups()