from stockGroupsService.GroupOver20Years import GroupOver20Years
from stockGroupsService.GroupSwiss import GroupSwiss
from stockGroupsService.GroupManager import GroupManager
from stockGroupsService.GroupSwissOver20Years import GroupSwissOver20Years
from stockGroupsService.GroupAmericanOver20Years import GroupAmericanOver20Years
from stockGroupsService.GroupSnP500 import GroupSnP500
from stockGroupsService.GroupSnP500Over20Years import GroupSnP500Over20Years


def generateGroups():
    dbPath = "src/database"
    groupPath = "src/stockGroups"

    groupClasses = [
        GroupOver20Years(),
        GroupSwiss(),
        GroupSwissOver20Years(),
        GroupAmericanOver20Years(),
        GroupSnP500(),
        GroupSnP500Over20Years(),
    ]

    manager = GroupManager(databasePath=dbPath, stockGroupPath=groupPath, groupClasses = groupClasses)
    manager.generateGroups()