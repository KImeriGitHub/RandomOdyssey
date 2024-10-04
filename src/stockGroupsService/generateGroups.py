from stockGroupsService.GroupOver20Years import GroupOver20Years
from stockGroupsService.GroupSwiss import GroupSwiss
from stockGroupsService.GroupManager import GroupManager
from stockGroupsService.GroupSwissOver20Years import GroupSwissOver20Years
from stockGroupsService.GroupAmericanOver20Years import GroupAmericanOver20Years


def generateGroups():
    dbPath = "src/database"
    groupPath = "src/stockGroups"

    groupClasses = [
        GroupOver20Years(),
        GroupSwiss(),
        GroupSwissOver20Years(),
        GroupAmericanOver20Years(),
    ]

    manager = GroupManager(databasePath=dbPath, stockGroupPath=groupPath, groupClasses = groupClasses)
    manager.generateGroups()